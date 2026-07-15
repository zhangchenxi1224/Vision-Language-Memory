from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered scientific support rule without hiding a negative result"
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-intervention-drop", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_intervention_drop <= 1.0:
        raise SystemExit("--minimum-intervention-drop must be in [0, 1].")

    scores = require_mapping(json.loads(args.scores.read_text(encoding="utf-8")), "scores")
    contrasts = require_mapping(scores.get("contrasts"), "scores.contrasts")
    diagnostics = require_mapping(scores.get("diagnostics"), "scores.diagnostics")
    dreamlite = require_mapping(diagnostics.get("dreamlite_latent"), "DreamLite diagnostics")

    checks: dict[str, bool] = {}
    evidence: dict[str, Any] = {}
    for baseline in ("query_only", "frozen_dreamlite"):
        name = f"dreamlite_latent_vs_{baseline}"
        contrast = require_mapping(contrasts.get(name), name)
        lower = contrast.get("ci_lower")
        if not isinstance(lower, (int, float)):
            raise ValueError(f"{name} has no numeric ci_lower.")
        checks[f"paired_ci_excludes_zero_vs_{baseline}"] = float(lower) > 0.0
        evidence[name] = {
            "observed_delta": contrast.get("observed_delta"),
            "ci_lower": lower,
            "ci_upper": contrast.get("ci_upper"),
        }

    for condition in ("reset", "shuffle"):
        diagnostic = require_mapping(dreamlite.get(condition), f"DreamLite {condition}")
        drop = diagnostic.get("accuracy_drop")
        if not isinstance(drop, (int, float)):
            raise ValueError(f"DreamLite {condition} has no numeric accuracy_drop.")
        checks[f"{condition}_drop_at_least_threshold"] = float(drop) >= args.minimum_intervention_drop
        evidence[condition] = diagnostic

    supported = all(checks.values())
    report = {
        "schema_version": 1,
        "scores": str(args.scores.resolve()),
        "minimum_intervention_drop": args.minimum_intervention_drop,
        "checks": checks,
        "evidence": evidence,
        "core_hypothesis_supported": supported,
        "conclusion": (
            "paired gains and state-dependence controls support the core memory-utility hypothesis"
            if supported
            else "technical chain may be valid, but the preregistered memory-utility hypothesis is not supported"
        ),
        "exit_policy": "scientific non-support is reported, not converted into a pipeline failure",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
