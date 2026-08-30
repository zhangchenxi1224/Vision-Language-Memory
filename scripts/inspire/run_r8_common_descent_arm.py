"""Fail-closed Inspire controller for one R8 common-descent arm."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inspire import run_r7_gradient_balance_arm as engine


def main(argv: Sequence[str] | None = None) -> int:
    forwarded = list(argv) if argv is not None else None
    if forwarded is None:
        import sys

        forwarded = sys.argv[1:]
    return engine.main(["--protocol-revision", "r8", *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
