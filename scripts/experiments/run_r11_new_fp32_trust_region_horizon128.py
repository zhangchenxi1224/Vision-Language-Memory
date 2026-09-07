"""Run the preregistered horizon-128 variant of the FP32 trust-region control."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments import run_r11_new_fp32_trust_region as parent_runner  # noqa: E402
from vision_memory.training import r11_new_fp32_trust_region_horizon128 as horizon_core  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Reuse the audited numerical runner while replacing only its locked config facade."""
    original_core = parent_runner.core
    parent_runner.core = horizon_core
    try:
        return parent_runner.main(argv)
    finally:
        parent_runner.core = original_core


if __name__ == "__main__":
    raise SystemExit(main())
