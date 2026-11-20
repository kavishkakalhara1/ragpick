"""ragpick quickstart: fully offline, no API keys.

Run from the repository root:

    python examples/quickstart.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ragpick import Pipeline
from ragpick.selector import recommend

HERE = Path(__file__).resolve().parent


def main() -> None:
    # 1. Describe your requirements and let ragpick pick a blueprint.
    rec = recommend({"corpus_size": "small", "keyword_heavy": True, "access_control": True})
    print(f"Recommended blueprint: {rec.blueprint}")
    for reason in rec.reasons:
        print(f"  - {reason}")

    # 2. Build the pipeline. Fake components keep this offline and deterministic;
    #    swap in `openai` (or your own plugin) for real models.
    with tempfile.TemporaryDirectory() as index_dir:
        pipeline = Pipeline.from_dict(
            {
                "blueprint": rec.blueprint,
                "index_dir": index_dir,
                "components": {"embedder": {"type": "fake"}, "generator": {"type": "fake"}},
                "query": {"top_k": 3},
            }
        )

        # 3. Ingest the sample documents.
        stats = pipeline.ingest(HERE / "data")
        print(f"\nIngested {stats.documents} documents ({stats.chunks} chunks)")

        # 4. Ask questions. Restricted documents are only visible to their groups.
        for question, groups in [
            ("What does error E-1042 mean?", None),
            ("When should support escalate an E-3301 ticket to Tier 2?", None),
            ("When should support escalate an E-3301 ticket to Tier 2?", ["support"]),
        ]:
            answer = pipeline.query(question, groups=groups)
            print(f"\nQ: {question}  (groups={groups})")
            print(f"A: {answer.text}")
            print(f"   citations: {answer.citations}")


if __name__ == "__main__":
    main()
