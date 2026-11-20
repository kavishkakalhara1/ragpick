"""GraphRAG: a knowledge graph of entities and relations for multi-hop retrieval.

**Ingest.** Every chunk is passed through an extractor that returns entities and
relations: an LLM (``extractor: llm``) or an offline heuristic (``heuristic``: product
codes, IDs and capitalised names, related when they share a sentence). The graph
(entities, relations, and which chunks support each) is persisted to
``<index_dir>/graph/graph.json``.

**Query (local search).**

1. entities named in the question are matched against the graph;
2. the graph is expanded ``hops`` steps from them, so "the warranty of the product
   that needs a C-wire" can reach chunks that never mention the question's words;
3. chunks are scored by how close their entities are (1.0 seed, 0.5 one hop, ...) and
   fused with hybrid BM25 + vector search by Reciprocal Rank Fusion;
4. a short **facts** context ("LM-300 → requires → C-wire") is put first so the
   LLM sees the connections explicitly.

**Access control.** Chunks are filtered like everywhere else, and a fact is shown only
if at least one chunk *visible to the caller* supports it, so the graph never leaks
relations learned from restricted documents.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ragpick.components.generators import split_sentences
from ragpick.components.keyword import ENGLISH_STOPWORDS, tokenize
from ragpick.components.retrievers import HybridRetriever, reciprocal_rank_fusion
from ragpick.errors import ConfigError
from ragpick.filters import matches_filters
from ragpick.interfaces import Embedder, Generator, KeywordIndex, TextCompleter, VectorStore
from ragpick.models import Chunk, RetrievedChunk
from ragpick.registry import register
from ragpick.tracing import set_attributes, span

logger = logging.getLogger(__name__)

GRAPH_FILE = "graph.json"

EXTRACT_SYSTEM = (
    "You extract a knowledge graph from text. Respond with JSON only: "
    '{"entities": [{"name": "...", "type": "...", "description": "..."}], '
    '"relations": [{"source": "...", "target": "...", "relation": "..."}]}'
)

EXTRACT_PROMPT = """Extract the important entities (products, components, error codes, people,
organisations, places, policies, plans) and the relations between them from the text.
Use short relation phrases in lowercase (e.g. "requires", "is covered by", "costs").
Only use facts stated in the text.

Text:
{text}"""

SUMMARY_SYSTEM = (
    "You summarise a cluster of related entities from a document collection. "
    "Write 2-4 plain sentences describing the theme and the key facts. Use only the "
    "information given."
)

SUMMARY_PROMPT = """Entities: {entities}

Relations:
{relations}

Excerpts:
{excerpts}

Summarise this cluster's theme and key facts."""

_GLOBAL_RE = re.compile(
    r"\b(main|key|major|common|recurring)\s+(themes?|topics?|issues?|points?|trends?)\b|"
    r"\b(overview|summari[sz]e|summary|overall|in general|across (all|the)|big picture)\b",
    re.IGNORECASE,
)

_GLOBAL_TERMS = frozenset(
    {
        "main", "key", "major", "common", "recurring", "theme", "themes", "topic", "topics",
        "issue", "issues", "point", "points", "trend", "trends", "overview", "summarize",
        "summarise", "summary", "overall", "general", "across", "big", "picture",
        "documents", "document", "corpus", "collection",
    }
)  # fmt: skip

_CODE_RE = re.compile(r"\b[A-Z]{1,6}-?\d{1,6}[A-Z]?\b")  # E-1042, LM-300, ISO27001
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")  # HVAC, SMS
_PHRASE_RE = re.compile(  # Lumen Plus, Tier 2, Internal Escalation Guide
    r"\b[A-Z][\w-]*[a-z0-9][\w-]*(?:\s+(?:[A-Z][\w-]*|\d+[A-Za-z]*))+"
)
_LEADING_WORDS = {"every", "each", "all", "the", "a", "an", "this", "these", "our", "your", "some"}
_NORM_RE = re.compile(r"[^\w\-. ]+")


def normalize(name: str) -> str:
    """Graph key for an entity name (lowercase, collapsed whitespace)."""
    return " ".join(_NORM_RE.sub(" ", name.lower()).split()).strip(" .-")


class Entity(BaseModel):
    """An extracted entity."""

    name: str
    type: str = "concept"
    description: str = ""


class Relation(BaseModel):
    """An extracted relation between two entity names."""

    source: str
    target: str
    relation: str = "related to"


class Extraction(BaseModel):
    """Entities and relations found in one chunk."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


def _sentence_entities(sentence: str) -> list[tuple[str, str]]:
    """``(name, kind)`` candidates in one sentence, longest match wins on overlap."""
    candidates: list[tuple[int, int, str, str]] = []
    for regex, kind in ((_CODE_RE, "code"), (_PHRASE_RE, "name"), (_ACRONYM_RE, "acronym")):
        for match in regex.finditer(sentence):
            name = match.group(0).strip()
            words = name.split()
            while words and words[0].lower() in _LEADING_WORDS:
                words = words[1:]
            if kind == "name" and len(words) < 2:
                continue  # single capitalised words are too noisy ("Add", "Low")
            name = " ".join(words)
            start = match.end() - len(name)
            candidates.append((start, match.end(), name, kind))
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str, str]] = []
    for start, end, name, kind in candidates:
        # codes are kept even inside longer names, so "LM-300" also matches "LM-300 Pro"
        if kind != "code" and any(start < e and s < end for s, e in taken):
            continue
        key = normalize(name)
        if len(key) < 2 or key in ENGLISH_STOPWORDS:
            continue
        taken.append((start, end))
        found.append((start, name, kind))
    return [(name, kind) for _, name, kind in sorted(found)]


def heuristic_extract(text: str, max_entities: int = 12) -> Extraction:
    """Offline extraction: codes, acronyms and multi-word names, related within a sentence."""
    entities: dict[str, Entity] = {}
    relations: list[Relation] = []
    for sentence in split_sentences(text):
        keys: list[str] = []
        for name, kind in _sentence_entities(sentence):
            key = normalize(name)
            if key not in entities and len(entities) < max_entities:
                entities[key] = Entity(name=name, type=kind, description=sentence.strip()[:200])
            if key in entities and key not in keys:
                keys.append(key)
        keys = keys[:6]
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                relations.append(
                    Relation(
                        source=entities[a].name, target=entities[b].name, relation="related to"
                    )
                )
    return Extraction(entities=list(entities.values()), relations=relations)


def parse_extraction(reply: str) -> Extraction | None:
    """Parse an LLM extraction reply; ``None`` if unusable."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return None
    try:
        return Extraction.model_validate(json.loads(match.group(0)))
    except ValueError:
        return None


class KnowledgeGraph:
    """Entities, relations and their supporting chunks; JSON-persistable."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str], dict[str, str]] = {}
        self.chunks: dict[str, Chunk] = {}
        self.communities: list[dict[str, Any]] = []
        self._adjacency: dict[str, set[str]] | None = None

    def __len__(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------- building
    def add(self, chunk: Chunk, extraction: Extraction) -> None:
        """Add one chunk's entities and relations."""
        self.chunks[chunk.id] = chunk.model_copy(update={"embedding": None})
        for entity in extraction.entities:
            key = normalize(entity.name)
            if len(key) < 2:
                continue
            node = self.nodes.setdefault(
                key, {"name": entity.name, "type": entity.type, "mentions": {}}
            )
            node["mentions"][chunk.id] = entity.description
        for rel in extraction.relations:
            a, b = normalize(rel.source), normalize(rel.target)
            if a == b or len(a) < 2 or len(b) < 2:
                continue
            for key, name in ((a, rel.source), (b, rel.target)):
                node = self.nodes.setdefault(key, {"name": name, "type": "concept", "mentions": {}})
                node["mentions"].setdefault(chunk.id, "")
            self.edges.setdefault((a, b), {})[chunk.id] = (
                rel.relation.strip().lower() or "related to"
            )
        self._adjacency = None

    def remove_docs(self, doc_ids: Iterable[str]) -> None:
        """Remove every chunk of ``doc_ids`` (and entities/relations only they supported)."""
        drop = {cid for cid, c in self.chunks.items() if c.doc_id in set(doc_ids)}
        if not drop:
            return
        for cid in drop:
            del self.chunks[cid]
        for key in list(self.nodes):
            mentions = self.nodes[key]["mentions"]
            for cid in drop & set(mentions):
                del mentions[cid]
            if not mentions:
                del self.nodes[key]
        for edge in list(self.edges):
            support = self.edges[edge]
            for cid in drop & set(support):
                del support[cid]
            if not support or edge[0] not in self.nodes or edge[1] not in self.nodes:
                del self.edges[edge]
        self._adjacency = None

    # ------------------------------------------------------------- querying
    def adjacency(self) -> dict[str, set[str]]:
        """Undirected neighbour sets."""
        if self._adjacency is None:
            adj: dict[str, set[str]] = {k: set() for k in self.nodes}
            for a, b in self.edges:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
            self._adjacency = adj
        return self._adjacency

    def match(self, question: str) -> list[str]:
        """Graph entities mentioned in ``question`` (longest names first)."""
        text = f" {normalize(question)} "
        hits = [key for key in self.nodes if f" {key} " in text]
        return sorted(hits, key=len, reverse=True)

    def neighborhood(self, seeds: list[str], hops: int) -> dict[str, int]:
        """Breadth-first distances from ``seeds`` up to ``hops``."""
        distance = dict.fromkeys(seeds, 0)
        queue = deque(seeds)
        adj = self.adjacency()
        while queue:
            node = queue.popleft()
            if distance[node] >= hops:
                continue
            for neighbour in adj.get(node, ()):
                if neighbour not in distance:
                    distance[neighbour] = distance[node] + 1
                    queue.append(neighbour)
        return distance

    def score_chunks(self, hood: dict[str, int]) -> list[tuple[str, float]]:
        """Chunks ranked by the closeness of the entities they mention."""
        scores: dict[str, float] = {}
        for key, dist in hood.items():
            weight = 1.0 / (2**dist)
            for cid in self.nodes[key]["mentions"]:
                scores[cid] = scores.get(cid, 0.0) + weight
        for (a, b), support in self.edges.items():
            if a in hood and b in hood:
                bonus = 0.5 / (2 ** max(hood[a], hood[b]))
                for cid in support:
                    scores[cid] = scores.get(cid, 0.0) + bonus
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    def facts(self, hood: dict[str, int], visible: set[str], limit: int) -> list[str]:
        """Relation sentences among ``hood`` supported by a *visible* chunk."""
        out: list[tuple[int, str]] = []
        for (a, b), support in self.edges.items():
            if a not in hood or b not in hood:
                continue
            seen = [cid for cid in support if cid in visible]
            if not seen:
                continue  # supported only by chunks the caller can't see
            relation = support[seen[0]]
            name_a, name_b = self.nodes[a]["name"], self.nodes[b]["name"]
            line = f"{name_a} → {relation} → {name_b} (source: {', '.join(sorted(seen)[:3])})"
            out.append((max(hood[a], hood[b]), line))
        out.sort(key=lambda t: (t[0], t[1]))
        return [line for _, line in out[:limit]]

    # ------------------------------------------------------------- communities
    def detect_communities(self, max_iterations: int = 20) -> list[list[str]]:
        """Group entities by deterministic weighted label propagation.

        Each entity repeatedly adopts the label with the most supporting chunks among
        its neighbours (ties → smallest label). Returns groups of two or more entities,
        largest first.
        """
        weights: dict[str, dict[str, int]] = {k: {} for k in self.nodes}
        for (a, b), support in self.edges.items():
            if a in weights and b in weights:
                weights[a][b] = weights[a].get(b, 0) + len(support)
                weights[b][a] = weights[b].get(a, 0) + len(support)
        labels = {k: k for k in self.nodes}
        for _ in range(max_iterations):
            changed = False
            for node in sorted(self.nodes):
                if not weights[node]:
                    continue
                score: dict[str, int] = {}
                for neighbour, w in weights[node].items():
                    score[labels[neighbour]] = score.get(labels[neighbour], 0) + w
                best = min(score, key=lambda lab: (-score[lab], lab))
                if score[best] > score.get(labels[node], 0) and best != labels[node]:
                    labels[node] = best
                    changed = True
            if not changed:
                break
        groups: dict[str, list[str]] = {}
        for node, label in labels.items():
            groups.setdefault(label, []).append(node)
        communities = [sorted(g) for g in groups.values() if len(g) >= 2]
        return sorted(communities, key=lambda g: (-len(g), g))

    def community_material(self, members: list[str]) -> tuple[list[str], list[str], list[str]]:
        """``(entity names, relation lines, supporting chunk ids)`` for a community."""
        member_set = set(members)
        names = [self.nodes[m]["name"] for m in members]
        relations: list[str] = []
        chunk_ids: set[str] = set()
        for m in members:
            chunk_ids.update(self.nodes[m]["mentions"])
        seen_pairs: set[frozenset[str]] = set()
        for (a, b), support in sorted(self.edges.items()):
            if a in member_set and b in member_set:
                chunk_ids.update(support)
                pair = frozenset((a, b))
                if pair in seen_pairs:  # A→B and B→A: describe the link once
                    continue
                seen_pairs.add(pair)
                relation = support[sorted(support)[0]]
                relations.append(f"{self.nodes[a]['name']} → {relation} → {self.nodes[b]['name']}")
        return names, relations, sorted(chunk_ids & set(self.chunks))

    # ------------------------------------------------------------- persistence
    def persist(self, path: Path) -> None:
        """Write ``graph.json`` into ``path``."""
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "format": 1,
            "nodes": self.nodes,
            "edges": [[a, b, support] for (a, b), support in self.edges.items()],
            "chunks": [c.model_dump(exclude={"embedding"}) for c in self.chunks.values()],
            "communities": self.communities,
        }
        (path / GRAPH_FILE).write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: Path) -> bool:
        """Load ``graph.json`` from ``path``; ``False`` if absent."""
        file = path / GRAPH_FILE
        if not file.is_file():
            return False
        data = json.loads(file.read_text(encoding="utf-8"))
        self.nodes = data["nodes"]
        self.edges = {(a, b): support for a, b, support in data["edges"]}
        self.chunks = {c["id"]: Chunk.model_validate(c) for c in data["chunks"]}
        self.communities = list(data.get("communities", []))
        self._adjacency = None
        return True


@register("retriever", "graph")
class GraphRetriever:
    """Hybrid retrieval plus knowledge-graph expansion; see the module docstring.

    Args:
        hops: graph expansion depth (1 to 3 is sensible).
        extractor: ``auto`` (LLM when the generator is a real LLM), ``llm`` or
            ``heuristic``.
        graph_weight: RRF weight of the graph ranking relative to hybrid search.
        include_facts: put a "knowledge graph facts" context first.
        max_facts: cap on fact lines.
        mode: ``local`` (entity search), ``global`` (community summaries) or ``auto``
            (global for "main themes / overview / summarise" questions).
        communities: build community summaries at ingest (needed for global search).
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        keyword_index: KeywordIndex,
        generator: Generator,
        index_dir: Path,
        k: int = 5,
        hops: int = 1,
        extractor: str = "auto",
        rrf_k: int = 60,
        graph_weight: float = 1.0,
        include_facts: bool = True,
        max_facts: int = 15,
        max_entities_per_chunk: int = 12,
        mode: str = "auto",
        communities: bool = True,
    ) -> None:
        if mode not in ("auto", "local", "global"):
            raise ConfigError(f"mode must be auto, local or global, got {mode!r}")
        if extractor not in ("auto", "llm", "heuristic"):
            raise ConfigError(f"extractor must be auto, llm or heuristic, got {extractor!r}")
        if hops < 0:
            raise ConfigError("hops must be >= 0")
        real_llm = isinstance(generator, TextCompleter) and not getattr(generator, "is_fake", False)
        if extractor == "llm" and not isinstance(generator, TextCompleter):
            raise ConfigError("extractor: llm needs a generator that supports complete()")
        self.mode = (
            "llm" if extractor == "llm" or (extractor == "auto" and real_llm) else "heuristic"
        )
        self.llm: TextCompleter | None = generator if self.mode == "llm" else None  # type: ignore[assignment]
        self.base = HybridRetriever(embedder, store, keyword_index, k=k, rrf_k=rrf_k)
        self.path = Path(index_dir) / "graph"
        self.graph = KnowledgeGraph()
        self._loaded = False
        self.k = k
        self.hops = hops
        self.rrf_k = rrf_k
        self.graph_weight = graph_weight
        self.include_facts = include_facts
        self.max_facts = max_facts
        self.max_entities = max_entities_per_chunk
        self.query_mode = mode
        self.build_communities = communities

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.graph.load(self.path)
            self._loaded = True

    def extract(self, text: str) -> Extraction:
        """Extract entities/relations from ``text`` (LLM or heuristic)."""
        if self.llm is not None:
            reply = self.llm.complete(EXTRACT_SYSTEM, EXTRACT_PROMPT.format(text=text)).text
            parsed = parse_extraction(reply)
            if parsed is not None:
                parsed.entities = parsed.entities[: self.max_entities]
                return parsed
            logger.warning("Unparseable extraction reply; using the heuristic extractor")
        return heuristic_extract(text, self.max_entities)

    # -------------------------------------------------------------- ingest
    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Extract and add ``chunks`` (replacing earlier chunks of the same documents)."""
        self._ensure_loaded()
        self.graph.remove_docs({c.doc_id for c in chunks})
        with span("graph_extract", extractor=self.mode, chunks=len(chunks)):
            for chunk in chunks:
                self.graph.add(chunk, self.extract(chunk.text))
            set_attributes(entities=len(self.graph), relations=len(self.graph.edges))
        if self.build_communities:
            self.rebuild_communities()
        self.graph.persist(self.path)

    def summarize(self, names: list[str], relations: list[str], chunk_ids: list[str]) -> str:
        """Community summary: LLM-written, or a deterministic template offline."""
        if self.llm is not None:
            excerpts = "\n".join(f"- {self.graph.chunks[cid].text[:300]}" for cid in chunk_ids[:5])
            prompt = SUMMARY_PROMPT.format(
                entities=", ".join(names[:30]),
                relations="\n".join(f"- {r}" for r in relations[:30]) or "(none)",
                excerpts=excerpts or "(none)",
            )
            text = self.llm.complete(SUMMARY_SYSTEM, prompt).text.strip()
            if text:
                return text
        listed = ", ".join(names[:12]) + (" and others" if len(names) > 12 else "")
        connections = "; ".join(relations[:8])
        summary = f"Topic covering {listed}."
        if connections:
            summary += f" Connections: {connections}."
        return summary

    def rebuild_communities(self) -> None:
        """Recompute communities; reuse summaries whose members and relations are unchanged."""
        previous = {c["signature"]: c for c in self.graph.communities}
        built: list[dict[str, Any]] = []
        reused = 0
        with span("graph_communities") as s:
            for members in self.graph.detect_communities():
                names, relations, chunk_ids = self.graph.community_material(members)
                signature = hashlib.sha256(
                    json.dumps([members, relations, chunk_ids]).encode("utf-8")
                ).hexdigest()[:16]
                if signature in previous:
                    built.append(previous[signature])
                    reused += 1
                    continue
                built.append(
                    {
                        "id": signature[:10],
                        "signature": signature,
                        "entities": names,
                        "chunks": chunk_ids,
                        "summary": self.summarize(names, relations, chunk_ids),
                    }
                )
            if s is not None:
                s.attributes.update(communities=len(built), reused=reused)
        self.graph.communities = built

    def is_global(self, query: str) -> bool:
        """Whether ``query`` should use global (community) search."""
        if self.query_mode != "auto":
            return self.query_mode == "global"
        return bool(_GLOBAL_RE.search(query))

    def global_search(
        self, query: str, k: int, filters: dict[str, Any] | None
    ) -> list[RetrievedChunk]:
        """Community summaries relevant to ``query`` that the caller may fully see."""
        terms = set(tokenize(query)) - _GLOBAL_TERMS
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for i, community in enumerate(self.graph.communities):
            chunks = [self.graph.chunks.get(cid) for cid in community["chunks"]]
            # visible only if *every* contributing chunk is: summaries mix their sources
            if not chunks or any(
                c is None or not matches_filters(c.metadata, filters) for c in chunks
            ):
                continue
            words = set(tokenize(community["summary"] + " " + " ".join(community["entities"])))
            overlap = len(terms & words) if terms else 0
            if terms and overlap == 0:
                continue
            scored.append((overlap + 0.01 * len(community["entities"]), -i, community))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        results = []
        for score, _, community in scored[:k]:
            text = (
                f"Community summary ({len(community['entities'])} entities): {community['summary']}"
            )
            chunk = Chunk(
                id=f"graph:community#{community['id']}",
                doc_id="graph:community",
                text=text,
                metadata={"synthetic": True, "entities": community["entities"][:20]},
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(score), source="graph"))
        return results

    # -------------------------------------------------------------- query
    def retrieve(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Global: community summaries; local: facts context + graph/hybrid fused chunks."""
        k = k or self.k
        self._ensure_loaded()
        if self.is_global(query):
            with span("graph_global", communities=len(self.graph.communities)) as gs:
                summaries = self.global_search(query, k, filters)
                if gs is not None:
                    gs.attributes["selected"] = len(summaries)
            if summaries:
                return summaries
        with span("graph_search", hops=self.hops) as s:
            seeds = self.graph.match(query)
            hood = self.graph.neighborhood(seeds, self.hops) if seeds else {}
            ranked = [
                (cid, score)
                for cid, score in self.graph.score_chunks(hood)
                if matches_filters(self.graph.chunks[cid].metadata, filters)
            ]
            graph_hits = [
                RetrievedChunk(chunk=self.graph.chunks[cid], score=score, source="graph")
                for cid, score in ranked[: 3 * k]
            ]
            visible = {cid for cid, _ in ranked}
            facts = self.graph.facts(hood, visible, self.max_facts) if self.include_facts else []
            if s is not None:
                s.attributes.update(
                    seeds=[self.graph.nodes[x]["name"] for x in seeds[:5]],
                    entities=len(hood),
                    chunks=len(graph_hits),
                    facts=len(facts),
                )
        hybrid = self.base.retrieve(query, max(k, 3 * k), filters)
        fused = reciprocal_rank_fusion([hybrid, graph_hits], self.rrf_k, (1.0, self.graph_weight))
        results = [
            r.model_copy(update={"source": "graph" if r.chunk.id in visible else "fused"})
            for r in fused
        ]
        if not facts:
            return results[:k]
        text = "Knowledge graph facts:\n" + "\n".join(f"- {line}" for line in facts)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        facts_chunk = Chunk(
            id=f"graph#facts-{digest}",
            doc_id="graph",
            text=text,
            metadata={
                "entities": [self.graph.nodes[x]["name"] for x in seeds[:10]],
                "synthetic": True,  # not a document: excluded from retrieval metrics
            },
        )
        return [RetrievedChunk(chunk=facts_chunk, score=1.0, source="graph"), *results[: k - 1]]
