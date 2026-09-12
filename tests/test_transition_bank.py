import copy
import json
from collections import Counter
from pathlib import Path

from scripts.experiments.official_transition_bank import assemble_bank, transition_plan, NOOP_EVENT


def test_transition_bank_requires_source_to_disambiguate_identical_noop_events():
    parent = json.loads((Path(__file__).resolve().parents[1] /
        "reports/official-alignment-results-20260913/state-bank-complete-manifest.json").read_text())
    before = copy.deepcopy(parent)
    sources = {state: {"source_kind": "sealed_rgb_1024", "source_image_path": state + ".png",
                       "source_latent_path": state + ".pt"} for state in ("ambient", "jazz", "clear")}
    bank = assemble_bank(parent, sources)
    assert parent == before
    assert len(transition_plan()) == len(bank["groups"]) == len(bank["teachers"]) == 15
    assert len({g["question_id"] for g in bank["groups"]}) == 15
    assert len({g["semantic_question_id"] for g in bank["groups"]}) == 1
    assert Counter(g["target_state"] for g in bank["groups"]) == {"ambient": 5, "jazz": 5, "clear": 5}
    noops = [g for g in bank["groups"] if g["operation"] == "noop"]
    assert {g["event_text"] for g in noops} == {NOOP_EVENT}
    assert len({g["answer"] for g in noops}) == 3
    assert all(g["source_state"] == g["target_state"] and g["source_kind"] == "sealed_rgb_1024" for g in noops)
    original_hashes = {t["latent_sha256"] for t in parent["teachers"]}
    assert {t["latent_sha256"] for t in bank["teachers"]} == original_hashes
    originals = {t["teacher_id"]: t for t in parent["teachers"]}
    for teacher in bank["teachers"]:
        original = originals[teacher["reused_parent_teacher"]]
        assert (teacher["latent_path"], teacher["latent_file_sha256"]) == (original["latent_path"], original["latent_file_sha256"])
