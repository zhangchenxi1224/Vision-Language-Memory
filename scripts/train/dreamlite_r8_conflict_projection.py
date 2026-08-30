"""R8 deterministic common-descent projection hard8 diagnostic."""

from __future__ import annotations

from typing import Sequence

from scripts.train import dreamlite_r7_gradient_balance as engine


def main(argv: Sequence[str] | None = None) -> int:
    forwarded = list(argv) if argv is not None else None
    if forwarded is None:
        import sys

        forwarded = sys.argv[1:]
    return engine.main(["--protocol-revision", "r8", *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
