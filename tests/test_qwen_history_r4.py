from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.eval.qwen_history_r4 import (  # noqa: E402
    expand_reverse_cyclic_views,
    intervention_event_streams,
    synthetic_queries,
)
from scripts.eval.score_qwen_history_r4 import _smoke_gate, _transition32_gate  # noqa: E402
from vision_memory.data.r4_baseline_lockbox import build_transition32  # noqa: E402
from vision_memory.data.schema import write_jsonl  # noqa: E402
from vision_memory.eval.r4_history_representations import (  # noqa: E402
    QWEN_R4_LAST_EFFECTIVE_EVENT,
    QWEN_R4_RAW_HISTORY,
)


CHOICES = ["teal", "burgundy", "ivory", "no active preference"]


def _row(
    *,
    episode: str,
    kind: str,
    form: str,
    view: int,
    condition: str,
    target_text: str,
    correct: bool,
    pair_id: str | None = None,
    variant: str | None = None,
    donor_target_index: int | None = None,
) -> dict[str, object]:
    target_index = view
    prediction_index = target_index if correct else (target_index + 1) % 4
    prediction_text = target_text if correct else "wrong"
    identity = pair_id or episode
    return {
        "episode_id": episode,
        "query_ordinal": 0,
        "probe_role": "delayed",
        "choice_view_family": "reverse-cyclic4",
        "choice_view_index": view,
        "condition": condition,
        "subtype": kind,
        "form": form,
        "target_index": target_index,
        "target_text": target_text,
        "prediction_index": prediction_index,
        "prediction_text": prediction_text,
        "choices": list(CHOICES),
        "distractor_pair_id": pair_id,
        "distractor_variant": variant,
        "memory_text_sha256": f"memory-{identity}-{view}",
        "prompt_sha256": f"prompt-{identity}-{view}",
        "choice_mean_nll": [float(view), 1.0, 2.0, 3.0],
        "donor_target_index": donor_target_index,
    }


def _smoke_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    targets = {"set": "teal", "overwrite": "burgundy", "clear": "no active preference", "noop": "teal"}
    for kind, target in targets.items():
        pair_id = "smoke-clean-noop" if kind in {"set", "noop"} else None
        variant = "clean" if kind == "set" else "distractor" if kind == "noop" else None
        for view in range(4):
            rows.append(
                _row(
                    episode=f"smoke-{kind}",
                    kind=kind,
                    form="separate",
                    view=view,
                    condition="standard",
                    target_text=target,
                    correct=True,
                    pair_id=pair_id,
                    variant=variant,
                )
            )
    return rows


def _transition_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind in ("set", "overwrite", "clear", "noop"):
        for form in ("separate", "mixed"):
            for history in ("short", "long"):
                for replica in range(2):
                    episode = f"transition-{kind}-{form}-{history}-r{replica}"
                    target = "no active preference" if kind == "clear" else ("teal", "burgundy")[replica]
                    pair_id = f"clean-noop-{form}-{history}-r{replica}" if kind in {"set", "noop"} else None
                    variant = "clean" if kind == "set" else "distractor" if kind == "noop" else None
                    for condition in ("standard", "reset", "shuffle", "state_swap"):
                        for view in range(4):
                            donor = (view + 1) % 4 if condition == "state_swap" else None
                            row = _row(
                                episode=episode,
                                kind=kind,
                                form=form,
                                view=view,
                                condition=condition,
                                target_text=target,
                                correct=condition == "standard",
                                pair_id=pair_id,
                                variant=variant,
                                donor_target_index=donor,
                            )
                            if condition == "state_swap":
                                row["prediction_index"] = donor
                                row["prediction_text"] = "donor"
                            rows.append(row)
    return rows


def test_smoke4_last_effective_gate_covers_four_kinds_and_exact_noop() -> None:
    report = _smoke_gate(_smoke_rows(), method=QWEN_R4_LAST_EFFECTIVE_EVENT)
    assert report["passed"] is True
    assert report["data_readability_required"] is True
    assert report["correct"] == 16
    assert report["clean_noop"]["all_fields_exact"] == 4


def test_transition32_gate_includes_all_locked_causal_controls() -> None:
    rows = _transition_rows()
    report = _transition32_gate(rows, method=QWEN_R4_LAST_EFFECTIVE_EVENT)
    assert report["passed"] is True
    assert report["state_swap"] == {"donor_answers": 32, "count": 32}
    assert report["reset"]["drop_from_standard"] == 128
    assert report["shuffle"]["drop_from_standard"] == 128
    assert report["clean_noop"]["all_fields_exact"] == 32
    assert set(report["checks"]) == {
        "overall",
        "positions",
        "event_kinds",
        "mixed",
        "cells",
        "rotation",
        "state_swap_donor",
        "reset_drop",
        "shuffle_drop",
        "clean_noop",
    }


def test_transition32_raw_performance_is_nonblocking_but_reported() -> None:
    report = _transition32_gate(_transition_rows(), method=QWEN_R4_RAW_HISTORY)
    assert report["performance_only"] is True
    assert report["data_readability_required"] is False
    assert report["checks"]["clean_noop"] is True


@pytest.mark.parametrize("control", ["reset", "shuffle"])
def test_transition32_gate_detects_missing_causal_drop(control: str) -> None:
    rows = _transition_rows()
    for row in rows:
        if row["condition"] == control:
            row["prediction_index"] = row["target_index"]
            row["prediction_text"] = row["target_text"]
    report = _transition32_gate(rows, method=QWEN_R4_LAST_EFFECTIVE_EVENT)
    assert report["passed"] is False
    assert report["checks"][f"{control}_drop"] is False


def test_transition32_gate_detects_nonexact_last_effective_noop() -> None:
    rows = _transition_rows()
    corrupted = copy.deepcopy(rows)
    target = next(
        row
        for row in corrupted
        if row["condition"] == "standard" and row["distractor_variant"] == "distractor"
    )
    target["choice_mean_nll"] = [99.0, 98.0, 97.0, 96.0]
    report = _transition32_gate(corrupted, method=QWEN_R4_LAST_EFFECTIVE_EVENT)
    assert report["passed"] is False
    assert report["checks"]["clean_noop"] is False


def test_generated_transition32_state_swap_donors_are_present_and_different() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "transition32.jsonl"
        write_jsonl(path, build_transition32())
        items = [item for item in synthetic_queries(path, None) if item.metadata["probe_role"] == "delayed"]
    assert len(items) == 32
    interventions = intervention_event_streams(items, condition="state_swap", seed=20260722)
    for item, intervention in zip(items, interventions, strict=True):
        own_target = item.choices[item.target_index]
        assert intervention.donor_target_text is not None
        assert intervention.donor_target_text != own_target
        views = expand_reverse_cyclic_views(item, intervention)
        assert len(views) == 4
        assert all(view.donor_target_index is not None for view in views)
