import json

from scripts.reporting.collect_official_alignment import collect


def test_collector_exposes_constant_state_failure_per_group(tmp_path):
    phase = tmp_path / "train/trained"
    phase.mkdir(parents=True)
    rows = [{"condition": "matched", "question_id": gold, "gold": gold,
             "prompt_id": "original_open", "raw": "ambient",
             "scorer": {"strict_correct": gold == "ambient",
                        "answer_followed_immediately_by_eos": gold == "ambient"}}
            for gold in ("ambient", "jazz", "no active preference")]
    (phase / "generations.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    summary = collect(tmp_path)["phases"]["trained"]
    assert summary["cells"]["matched/original_open"]["answer_eos"] == 1
    groups = summary["conditional_groups"]
    assert len(groups) == 3
    assert groups["ambient"]["cells"]["matched/original_open"]["answer_eos"] == 1
    assert groups["jazz"]["cells"]["matched/original_open"]["answer_eos"] == 0
    assert groups["no active preference"]["matched_raw"] == {"ambient": 1}
