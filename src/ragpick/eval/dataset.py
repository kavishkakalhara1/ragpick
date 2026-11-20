"""JSONL evaluation datasets.

One JSON object per line::

    {"question": "...", "ground_truth": "...", "relevant_doc_ids": ["faq.md"],
     "groups": ["support"]}

Only ``question`` is required. ``relevant_doc_ids`` enables retrieval metrics,
``groups`` runs the query as a member of those groups (access control).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from ragpick.config import format_validation_error
from ragpick.errors import ConfigError


class EvalExample(BaseModel):
    """One evaluation question."""

    model_config = ConfigDict(extra="forbid")

    question: str
    ground_truth: str | None = None
    relevant_doc_ids: list[str] | None = None
    groups: list[str] | None = None
    id: str | None = None


def load_dataset(path: str | Path) -> list[EvalExample]:
    """Load and validate a JSONL dataset; errors name the offending line.

    Blank lines and lines starting with ``#`` or ``//`` are ignored.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Dataset not found: {p}")
    examples: list[EvalExample] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{p}:{lineno}: invalid JSON ({exc.msg})") from None
        try:
            examples.append(EvalExample.model_validate(data))
        except ValidationError as exc:
            raise ConfigError(
                format_validation_error(exc, f"{p}:{lineno}", "dataset row")
            ) from None
    if not examples:
        raise ConfigError(f"Dataset {p} contains no examples")
    return examples
