from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from vision_memory.reader.open_eos import generation_diagnostics
from vision_memory.training.latent_teacher_bank import (AuditError, SUFFIX, audit_evaluations,
    audit_direct_prompt_receipts, direct_prompt_protocol,
    campaign_progress, export_bank, geometry_description, local_artifact, validate_prompts, write_json)
from vision_memory.training.latent_bank_unet import load_teacher_bank


def prompts():
    return {key:sentence+SUFFIX for key,sentence in [
        ("original_open","What music is currently preferred?"),
        ("paraphrase_1","Which music is preferred now?"),
        ("paraphrase_2","What is the current music preference?"),
        ("paraphrase_3","Name the music currently preferred."),
        ("paraphrase_4","What music preference is in effect?")]}


def evaluated(*,direct=True,overgenerate_original=False):
    rows=[]
    for condition in ("matched","blank","fixed_donor" if direct else "donor"):
        for prompt_id,query in prompts().items():
            bad=condition != "matched"
            extra=overgenerate_original and condition == "matched" and prompt_id == "original_open"
            generated={"raw":"unknown" if bad else "ambient trance" if extra else "ambient",
                       "generated_token_ids":[8,151645] if bad else [42,99,151645] if extra else [42,151645],
                       "eos_token_ids":[151645,151643]}
            score=generation_diagnostics(generated,"ambient",[42])
            row={"optimizer_step":256,"condition":condition,"prompt_id":prompt_id,"query":query,
                 "gold":"ambient","gold_eos_appended":True,"image_sha256":condition+"-image",**generated}
            row.update({"scorer":score} if direct else score)
            rows.append(row)
    return rows


def test_fixed_two_instruction_lines_cannot_change():
    validate_prompts(prompts())
    modified=prompts()
    modified["paraphrase_1"]=modified["paraphrase_1"].replace("answer.\nAnswer","answer. Answer")
    with pytest.raises(AuditError,match="instructions"):
        validate_prompts(modified)


@pytest.mark.parametrize("direct",[True,False])
def test_endpoint_set_uses_raw_em_not_prefix_or_best_step(direct):
    result=audit_evaluations(evaluated(direct=direct,overgenerate_original=True),prompts(),"ambient",direct=direct)
    assert result["answer_prefix_correct"] and result["overgeneration"]
    assert not result["qa_pass"] and not result["robust_qa_pass"]
    assert result["blank_correct"] == result["donor_correct"] == 0


def test_duplicate_rows_missing_grid_and_fake_scoring_rejected():
    rows=evaluated()
    rows[-1]=deepcopy(rows[-2])
    with pytest.raises(AuditError,match="grid"):
        audit_evaluations(rows,prompts(),"ambient",direct=True)
    rows=evaluated(overgenerate_original=True)
    rows[0]["scorer"]["strict_correct"]=True
    with pytest.raises(AuditError,match="score"):
        audit_evaluations(rows,prompts(),"ambient",direct=True)
    rows=evaluated()
    rows[0]["gold_eos_appended"]=False
    with pytest.raises(AuditError,match="EOS"):
        audit_evaluations(rows,prompts(),"ambient",direct=True)


def test_paused_all_runs_do_not_seal_or_trigger_writer(tmp_path):
    specs=[{"run_id":"one"},{"run_id":"two"}]
    for i,spec in enumerate(specs):
        write_json(tmp_path/f"lane-{i}"/"runs"/spec["run_id"]/"terminal.json",{"status":"completed","optimizer_steps":256})
        write_json(tmp_path/f"lane-{i}"/"terminal.json",{"status":"paused_at_run_boundary"})
    progress=campaign_progress(tmp_path,"direct",specs)
    assert progress["completed"] == 2 and not progress["complete"]
    assert progress["state"] == "waiting_for_oracle"
    for i in range(2):
        write_json(tmp_path/f"lane-{i}"/"terminal.json",{"status":"completed"})
    assert not campaign_progress(tmp_path,"direct",specs)["complete"]
    write_json(tmp_path/"terminal.json",{"status":"failed"})
    assert campaign_progress(tmp_path,"direct",specs)["state"] == "failed"
    write_json(tmp_path/"terminal.json",{"status":"completed"})
    assert campaign_progress(tmp_path,"direct",specs)["complete"]


def test_frozen_completion_requires_full_oracle_stage_and_no_failed_run(tmp_path):
    specs=[{"run_id":"eos-one"}]
    write_json(tmp_path/"runs/eos-one/campaign_terminal.json",{"returncode":0})
    write_json(tmp_path/"status.json",{"state":"completed","stage":"A1"})
    assert not campaign_progress(tmp_path,"frozen",specs)["complete"]
    write_json(tmp_path/"status.json",{"state":"completed","stage":"oracle_bank_complete"})
    assert campaign_progress(tmp_path,"frozen",specs)["complete"]
    write_json(tmp_path/"runs/eos-one/campaign_terminal.json",{"returncode":1})
    progress=campaign_progress(tmp_path,"frozen",specs)
    assert progress["state"] == "failed" and not progress["complete"]


def audited(run_id,question="question-a",success=True,offset=0.):
    latent=torch.arange(65536,dtype=torch.float32).reshape(1,4,128,128)/65536+offset
    qa={"qa_pass":success,"robust_qa_pass":False,"blank_correct":1,"donor_correct":2,
        "controls_denominator":5,"answer_prefix_correct":True,"overgeneration":not success}
    return {"spec":{"run_id":run_id,"distribution":"gaussian"},"question_id":question,"answer":"ambient",
            "event_text":"The desk now prefers ambient music.","question_variants":prompts(),
            "endpoint":latent,"initial":latent*.5,"source":torch.zeros_like(latent),
            "donor":{"kind":"image","value":torch.zeros(1,3,8,8),"answer":"orange"},
            "qa":qa,"rows":[],"source_run":"/fresh/eos/"+run_id,
            "termination":{"assistant_end_token_id":151645},"models":{"reader":"locked"},"data":{"train":"locked"},
            "provenance":{"new_eos":True}}


def test_export_deduplicates_coordinates_preserves_repeats_and_missing_question_denominator(tmp_path):
    rows=[audited("first"),audited("repeat"),audited("second",offset=.1),audited("failed-question",question="question-b",success=False)]
    bank=export_bank(rows,tmp_path,route="direct",provenance={"fresh_EOS_only":True})
    assert bank["unique_teacher_count"] == 2 and bank["successful_run_count"] == 3
    assert len(bank["teachers"][0]["source_runs"]) == 2
    assert bank["planned_question_ids"] == ["question-a","question-b"]
    assert bank["excluded_question_ids"] == ["question-b"]
    assert bank["question_coverage"] == {"successful":1,"planned":2,"fraction":.5}
    assert bank["teachers"][0]["qa"]["donor_correct"] == 2
    assert bank["teachers"][0]["qa"]["blank_correct"] == 1
    loaded,tensors=load_teacher_bank(tmp_path/"manifest.json")
    assert len(tensors) == 2 and loaded["bank_status"] == "sealed"
    for teacher in bank["teachers"]:
        expected=rows[0 if teacher is bank["teachers"][0] else 2]["endpoint"]
        assert torch.equal(tensors[teacher["teacher_id"]],expected)
    assert (tmp_path/"question-a-geometry.png").is_file()


def test_all_failed_bank_is_blocked_and_cannot_train(tmp_path):
    bank=export_bank([audited("failed",success=False)],tmp_path,route="frozen",provenance={})
    assert bank["bank_status"] == "blocked_no_success" and not bank["teachers"]
    assert not bank["unet_training_started"] and bank["question_coverage"]["fraction"] == 0
    with pytest.raises(ValueError,match="sealed"):
        load_teacher_bank(tmp_path/"manifest.json")


def test_geometry_isotropic_reference_same_n_d_and_no_invented_cluster_count():
    values=np.random.default_rng(12).standard_normal((5,80))
    result=geometry_description(values)
    shifted=geometry_description(values+10)
    assert result["sample_rank_cap"] == 4 and result["cluster_count"] is None
    assert len(result["matched_n_d_isotropic_centered_pca"]["eigenvalues"]) == 5
    assert result["centered_pca"]["r95"] <= 4
    assert result["raw_pairwise_l2"]["median"] == pytest.approx(shifted["raw_pairwise_l2"]["median"])
    assert result["centered_pairwise_l2"]["median"] == pytest.approx(result["raw_pairwise_l2"]["median"])
    assert geometry_description(np.empty((0,80)))["samples"] == 0


def test_artifact_path_escape_is_rejected(tmp_path):
    root=tmp_path/"root"
    root.mkdir()
    (tmp_path/"outside").write_text("outside")
    with pytest.raises(AuditError,match="escaping"):
        local_artifact(root,"../outside")


def multiprompt_receipts():
    trained = ["original_open", "paraphrase_1", "paraphrase_2"]
    held = ["paraphrase_3", "paraphrase_4"]
    config = {"training":{"optimizer":"Adam", "steps":256, "lr":.05, "lambda_eos":1.,
                          "prompts":trained, "prompt_schedule":"round_robin_zero_based"}, "heldout_prompts":held}
    manifest = {"training_prompts":trained, "heldout_prompts":held, "prompt_schedule":"round_robin_zero_based",
                "optimizer_prompt_counts":{"original_open":86, "paraphrase_1":85, "paraphrase_2":85}}
    metrics = [{"optimizer_step":i+1, "training_prompt_id":trained[i%3]} for i in range(256)]
    rows = evaluated()
    for row in rows:
        row["question_trained"] = row["prompt_id"] in trained
    return config, manifest, metrics, rows


def test_new_direct_bank_retains_three_training_and_two_heldout_prompts():
    config, manifest, metrics, rows = multiprompt_receipts()
    protocol = audit_direct_prompt_receipts(config, manifest, metrics, rows)
    assert protocol["training_prompts"] == ["original_open", "paraphrase_1", "paraphrase_2"]
    assert protocol["heldout_prompts"] == ["paraphrase_3", "paraphrase_4"]


@pytest.mark.parametrize("bad_prompt", ["paraphrase_3", "paraphrase_2", None])
def test_new_direct_bank_rejects_heldout_leakage_schedule_drift_and_missing_receipt(bad_prompt):
    config, manifest, metrics, rows = multiprompt_receipts()
    metrics[1]["training_prompt_id"] = bad_prompt
    with pytest.raises(AuditError, match="optimizer prompt schedule"):
        audit_direct_prompt_receipts(config, manifest, metrics, rows)


def test_new_direct_bank_rejects_old_labels_and_mixed_manifest():
    config, manifest, metrics, rows = multiprompt_receipts()
    rows[1]["question_trained"] = False
    with pytest.raises(AuditError, match="label mismatch"):
        audit_direct_prompt_receipts(config, manifest, metrics, rows)
    rows[1]["question_trained"] = True
    manifest["training_prompts"] = ["original_open"]
    with pytest.raises(AuditError, match="manifest mismatch"):
        audit_direct_prompt_receipts(config, manifest, metrics, rows)


def test_legacy_direct_prompt_receipts_still_work_without_per_step_prompt_field():
    config = {"training":{"optimizer":"Adam", "steps":256, "lr":.05, "lambda_eos":1., "prompts":["original_open"]}}
    rows = evaluated()
    for row in rows:
        row["question_trained"] = row["prompt_id"] == "original_open"
    metrics = [{"optimizer_step":i+1} for i in range(256)]
    result = audit_direct_prompt_receipts(config, {"training_prompts":["original_open"]}, metrics, rows)
    assert len(result["heldout_prompts"]) == 4
    config["training"]["lambda_eos"] = 0.
    with pytest.raises(AuditError, match="training contract"):
        direct_prompt_protocol(config)
