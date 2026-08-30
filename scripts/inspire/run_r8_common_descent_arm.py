"""Fail-closed Inspire controller for one R8 common-descent arm."""

from __future__ import annotations

from typing import Sequence

from scripts.inspire import run_r7_gradient_balance_arm as engine


def main(argv: Sequence[str] | None = None) -> int:
    forwarded = list(argv) if argv is not None else None
    if forwarded is None:
        import sys

        forwarded = sys.argv[1:]
    return engine.main(["--protocol-revision", "r8", *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
