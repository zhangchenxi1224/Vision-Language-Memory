from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.eval.r3_teacher_attribution import (  # noqa: E402
    TEACHER_CONTROLS,
    score_r3_teacher_attribution,
)


ARTIFACT_NAMES = (
    "distill_summary",
    "qa_summary",
    "distill_retrieval",
    "qa_retrieval",
    "qa_gate",
)
CLI_CONTROL_NAMES = {
    "correct": "correct",
    "shuffled": "shuffled",
    "random": "random-moment-matched",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the complete offline R3 Set8 teacher attribution package",
    )
    for cli_control in CLI_CONTROL_NAMES:
        for artifact in ARTIFACT_NAMES:
            parser.add_argument(
                f"--{cli_control}-{artifact.replace('_', '-')}",
                dest=f"{cli_control}_{artifact}",
                type=Path,
                required=True,
            )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fail-on-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero for a well-formed scientific gate failure.",
    )
    return parser.parse_args()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return dict(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, destination)


def main() -> int:
    args = parse_args()
    arms: dict[str, dict[str, Any]] = {}
    for cli_control, scientific_control in CLI_CONTROL_NAMES.items():
        arms[scientific_control] = {
            artifact: _load_json_object(getattr(args, f"{cli_control}_{artifact}")) for artifact in ARTIFACT_NAMES
        }
    if set(arms) != set(TEACHER_CONTROLS):
        raise RuntimeError("Internal teacher-control mapping drifted from the scoring contract.")
    report = score_r3_teacher_attribution(arms)
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "passed": report["passed"],
                "scientific_payload_sha256": report["scientific_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] or not args.fail_on_gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
