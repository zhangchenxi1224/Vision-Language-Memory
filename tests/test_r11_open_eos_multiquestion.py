"""Meaningful contract/recovery tests; no model or GPU runtime required."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch

from scripts.experiments import run_r11_open_eos_multiquestion as mq


PANEL = mq.ROOT / "configs/experiments/r11_open_eos_multiquestion_targets.json"


def real_panel():
    return json.loads(PANEL.read_text(encoding="utf-8"))


def test_real_sixteen_question_panel_has_independent_groups_and_no_answer_leak():
    panel = mq.validate_panel(real_panel())
    assert len(panel["targets"]) == 16
    assert len({x["semantic_group_id"] for x in panel["targets"]}) == 16
    assert {x["scorer_metadata"]["gold"] for x in panel["targets"]} == {"green", "juice", "jazz", "linen", "pasta", "no active preference"}
    assert panel["control_spec"]["donor_gold"] == "ambient"


@pytest.mark.parametrize("change", ["gold", "query_leak", "duplicate_group", "trained_rewrite", "donor_gold"])
def test_invalid_panel_fails_before_runtime(change):
    panel = real_panel()
    target = panel["targets"][0]
    if change == "gold":
        target["scorer_metadata"]["gold"] = "forged answer"
    elif change == "query_leak":
        target["inputs"]["original_open"] += " " + target["scorer_metadata"]["gold"]
    elif change == "duplicate_group":
        panel["targets"][1]["semantic_group_id"] = target["semantic_group_id"]
    elif change == "trained_rewrite":
        target["inputs"]["new_rewrite_open"] = target["inputs"]["original_open"]
    else:
        panel["control_spec"]["donor_gold"] = target["scorer_metadata"]["gold"]
    with pytest.raises(ValueError):
        mq.validate_panel(panel)


def source_fixture(tmp_path):
    query = {"text": "What meal is preferred? Choose exactly one option.",
             "choices": ["pasta", "rice", "soup", "no active preference"], "target_index": 0}
    episode = {"episode_id": "source-1", "semantic_group_id": "group-1", "split": "train", "turns": [{"query": query}]}
    source = tmp_path/"train.jsonl"
    source.write_text(json.dumps(episode)+"\n", encoding="utf-8")
    target = {"target_index":0, "segment_id":"question-1", "semantic_group_id":"group-1",
              "source_episode_id":"source-1", "source_record_index":0, "source_turn_id":0,
              "source_episode_sha256":mq.canonical_hash(episode), "source_query_sha256":mq.canonical_hash(query),
              "source_prefix":episode["turns"], "source_prefix_is_audit_only_not_reader_input":True,
              "original":{"query":query["text"],"choices":query["choices"],"answer_index":0},
              "inputs":{"original_open":"What meal is preferred?", "paraphrase_open":"Name the meal preference.",
                        "new_rewrite_open":"Which meal is currently stored?"},
              "scorer_metadata":{"gold":"pasta","aliases":[]}}
    checkpoint=tmp_path/"donor.pt"
    torch.save({"step":256,"latent_fp32":torch.zeros(1,4,128,128)}, checkpoint)
    raw=tmp_path/"donor_generations.jsonl"
    raw.write_text("\n".join(json.dumps({"run_id":"seed-00-B","step":256,"condition":"matched",
        "prompt_id":prompt,"raw":"ambient","strict_correct":True}) for prompt in ["original_open","paraphrase_open"])+"\n")
    panel={"selection":{"source_dataset_path":str(source),"source_dataset_sha256":mq.replay.sha256_file(source),
                         "split":"train","method":"fixture"},"targets":[target],
           "control_spec":{"donor_checkpoint":str(checkpoint),"donor_checkpoint_sha256":mq.replay.sha256_file(checkpoint),
               "donor_gold":"ambient","donor_source_run_id":"seed-00-B","donor_source_raw_generations":str(raw),
               "donor_source_raw_generations_sha256":mq.replay.sha256_file(raw)}}
    return panel


def test_real_source_binding_detects_forged_export_even_when_dataset_sha_matches(tmp_path):
    panel=source_fixture(tmp_path)
    mq.validate_panel(panel)
    donor,audit=mq.verify_source(panel)
    assert donor.shape==(1,4,128,128) and audit["success_evidence_count"]==2
    bad=copy.deepcopy(panel)
    bad["targets"][0]["original"]["query"]="Forged source wording"
    with pytest.raises(ValueError,match="do not match real source"):
        mq.verify_source(bad)


def test_donor_label_evidence_and_file_hash_are_independently_required(tmp_path):
    panel=source_fixture(tmp_path)
    raw=Path(panel["control_spec"]["donor_source_raw_generations"])
    raw.write_text(raw.read_text().replace('"ambient"','"jazz"'))
    with pytest.raises(ValueError,match="behavioral evidence changed"):
        mq.verify_source(panel)
    panel["control_spec"]["donor_source_raw_generations_sha256"]=mq.replay.sha256_file(raw)
    with pytest.raises(ValueError,match="not behaviorally verified"):
        mq.verify_source(panel)


def completed_fixture(tmp_path):
    directory=tmp_path/"completed"
    directory.mkdir()
    target=real_panel()["targets"][0]
    binding=mq.run_binding(target,0,"A","plan-hash","initial-hash")
    mq.replay.write_json(directory/"manifest.json",binding)
    mq.replay.write_json(directory/"terminal.json",{"status":"completed","optimizer_steps":256})
    for step in range(1,257):
        mq.replay.append_jsonl(directory/"metrics.jsonl",{"step":step})
    for step in mq.STEPS:
        path=directory/"checkpoints"/f"step-{step:03d}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save({"step":step},path)
        mq.replay.append_jsonl(directory/"checkpoint_index.jsonl",{"step":step,"path":str(path.relative_to(directory)),
                                                                 "file_sha256":mq.replay.sha256_file(path)})
        for condition in (mq.CONDITIONS if step==256 else ["matched"]):
            for prompt in mq.PROMPTS:
                mq.replay.append_jsonl(directory/"raw_generations.jsonl",{
                    **{k:binding[k] for k in ["target_index","segment_id","seed","arm"]},
                    "step":step,"condition":condition,"prompt_id":prompt,
                    "prompt_exposed_in_training":prompt=="original_open","decoding":"raw_greedy_32_original_eos"})
    mq.write_inventory(directory)
    return directory,binding


def test_hash_verified_completed_reuse_and_partial_fail_closed(tmp_path):
    directory,binding=completed_fixture(tmp_path)
    assert len(mq.verify_completed_run(directory,binding))==27
    (directory/"metrics.jsonl").write_text((directory/"metrics.jsonl").read_text()+"\n")
    with pytest.raises(ValueError,match="SHA mismatch"):
        mq.verify_completed_run(directory,binding)
    partial=tmp_path/"partial"
    partial.mkdir()
    mq.replay.write_json(partial/"manifest.json",binding)
    with pytest.raises(ValueError,match="Partial run retained"):
        mq.verify_completed_run(partial,binding)
    assert (partial/"manifest.json").exists()


def test_reuse_rejects_same_files_bound_to_different_question(tmp_path):
    directory,binding=completed_fixture(tmp_path)
    changed={**binding,"target_index":999}
    with pytest.raises(ValueError,match="binding changed"):
        mq.verify_completed_run(directory,changed)


def test_new_A_is_fresh_and_does_not_claim_old_ambient_parity():
    target=real_panel()["targets"][0]
    a=mq.run_binding(target,0,"A","plan","same-init")
    b=mq.run_binding(target,0,"B","plan","same-init")
    assert a["initial_latent_sha256"]==b["initial_latent_sha256"]
    assert a["append_eos"] is False and b["append_eos"] is True
    assert a["eos_lambda"] is None and b["eos_lambda"]==1.
    assert a["rewrites_used_in_training"] is False
    assert "no old ambient trajectory parity claim" in a["baseline"]
