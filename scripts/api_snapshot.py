"""Write (or check) tests/api_snapshot.json: ragpick's public API surface.

Usage:
    python scripts/api_snapshot.py          # regenerate after an intentional API change
    python scripts/api_snapshot.py --check  # exit 1 if the surface changed (CI)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ragpick._api_surface import build

OUT = Path(__file__).resolve().parent.parent / "tests" / "api_snapshot.json"


def main() -> int:
    """Write or check the snapshot."""
    content = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    if "--check" in sys.argv:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != content:
            print(f"{OUT} is out of date: the public API changed. If intentional, run")
            print("python scripts/api_snapshot.py and document the change in CHANGELOG.md.")
            return 1
        print(f"{OUT} is up to date")
        return 0
    OUT.write_text(content, encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
