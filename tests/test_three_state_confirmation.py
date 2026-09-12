import copy
import json
from pathlib import Path

import pytest

from scripts.probes.official_three_state_confirmation import confirmation_plan, validate_development_rows, validate_guidance_gate
from vision_memory.training.latent_bank_unet import stable_seed


def bank():
    return json.loads((Path(__file__).resolve().parents[1] /
        "reports/official-alignment-results-20260913/state-bank-complete-manifest.json").read_text())


def valid_rows():
    rows = []
    for group in bank()["groups"]:
        for i in range(8):
            for prompt, query in group["question_variants"].items():
                rows.append({"condition": "matched", "question_id": group["question_id"],
                    "noise_seed": stable_seed(20260913, "heldout-evaluation-noise", i),
                    "prompt_id": prompt, "query": query, "gold": group["answer"],
                    "generated_token_ids": [42, 151645], "scorer": {"gold_token_ids": [42],
                        "strict_correct": True, "answer_followed_immediately_by_eos": True}})
    return rows


def test_confirmation_pairs_new_noises_and_does_not_train_on_withheld_event_wording():
    plan = confirmation_plan(bank())
    assert plan == confirmation_plan(bank())
    trained = [c for c in plan if c["style"] == "trained"]
    assert len(trained) == 3 and all(c["seeds"] == trained[0]["seeds"] for c in trained)
    assert sum(len(c["seeds"]) for c in plan) == 72
    training_events = {g["event_text"] for g in bank()["groups"]}
    assert all(c["event"] not in training_events and c["seeds"] == trained[0]["seeds"][:4]
               for c in plan if c["style"] != "trained")
    previous = {stable_seed(20260913, "official-full-confirmation-noise-v1", i) for i in range(16)}
    assert previous.isdisjoint(trained[0]["seeds"])


def test_confirmation_gate_rejects_missing_duplicate_failed_and_overgenerated_cells():
    rows = valid_rows()
    validate_development_rows(rows, bank())
    with pytest.raises(ValueError, match="Incomplete"):
        validate_development_rows(rows[:-1], bank())
    with pytest.raises(ValueError, match="duplicate"):
        validate_development_rows(rows + [rows[0]], bank())
    changed = copy.deepcopy(rows)
    changed[-1]["scorer"]["strict_correct"] = False
    with pytest.raises(ValueError, match="every state"):
        validate_development_rows(changed, bank())
    changed = copy.deepcopy(rows)
    changed[-1]["generated_token_ids"] = [42, 99, 151645]
    with pytest.raises(ValueError, match="every state"):
        validate_development_rows(changed, bank())


def test_cfg1_confirmation_requires_native_control_and_new_noise_namespace():
    old = {s for c in confirmation_plan(bank()) for s in c['seeds']}
    new = {s for c in confirmation_plan(bank(), guidance_scale=1.0) for s in c['seeds']}
    assert len(new)==16 and old.isdisjoint(new)
    identity={'probe_commit':'1e4acd2bcd9cbf9f0b2878e51694c17bebe06110',
        'parent_result_sha256':'parent','checkpoint_sha256':'weights','condition_style':'native',
        'guidance_scale':1.0,'image_guidance_scale':1.0,'inference_steps':28,'optimizer_updates':0,'trainable_scope':'full_unet'}
    validate_guidance_gate({'identity':identity},'parent','weights')
    for key,value in [('condition_style','training_raw'),('checkpoint_sha256','other'),('optimizer_updates',1)]:
        with pytest.raises(ValueError,match='native zero-update'):
            validate_guidance_gate({'identity':{**identity,key:value}},'parent','weights')
