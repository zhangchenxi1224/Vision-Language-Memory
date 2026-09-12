from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vision_memory.dreamlite.differentiable_mobile import DifferentiableDreamLiteMobileSampler
from vision_memory.repro import canonical_tensor_sha256
from vision_memory.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from vision_memory.training.latent_bank_unet import (
    INSTRUCTIONS, anchored_flow_bridge, official_flow_bridge, balanced_draw, bank_geometry, file_sha256,
    load_teacher_bank, member_split, predict_velocity, stable_seed,
)


def test_source_anchored_bridge_endpoints_and_euler_direction():
    source = torch.tensor([[[[2., 4.]]]])
    noise = torch.tensor([[[[-2., 8.]]]])
    target = torch.tensor([[[[9., -3.]]]])
    start = .5 * source + .5 * noise
    zero, velocity = anchored_flow_bridge(source, noise, target, 0.)
    state, second_velocity = anchored_flow_bridge(source, noise, target, .5)
    torch.testing.assert_close(zero, target)
    torch.testing.assert_close(state, start)
    torch.testing.assert_close(velocity, second_velocity)
    # FlowMatchEuler steps from decreasing sigma; opposite sign cannot pass.
    for sigma, next_sigma in zip((.5, .375, .25, .125), (.375, .25, .125, 0.)):
        state = state + (next_sigma - sigma) * velocity
    torch.testing.assert_close(state, target)
    torch.testing.assert_close(velocity, noise + source - 2 * target)
    with pytest.raises(ValueError):
        anchored_flow_bridge(source, noise, target, .6)


class CapturingUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.tensor(.25))
        self.observed = None

    def forward(self, model_input, **kwargs):
        self.observed = (model_input.detach(), kwargs)
        return (model_input * self.lora_A,)


def test_velocity_uses_real_sampler_width_condition_timestep_and_gradient():
    unet = CapturingUNet()
    sampler = DifferentiableDreamLiteMobileSampler(unet=unet,
        scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)), vae_scale_factor=8)
    state = torch.full((1, 4, 3, 5), 3.)
    source = torch.full_like(state, 7.)
    embeds = torch.randn((1, 4, 8))
    mask = torch.ones((1, 4), dtype=torch.long)
    pred = predict_velocity(sampler, state, source, .375, embeds, mask)
    actual, kwargs = unet.observed
    assert actual.shape == (1, 4, 3, 10)
    torch.testing.assert_close(actual[..., :5], state)
    torch.testing.assert_close(actual[..., 5:], source)
    assert kwargs["timestep"].tolist() == [375.]
    assert kwargs["added_cond_kwargs"]["time_ids"].tolist() == [[40., 24.]]
    assert kwargs["encoder_hidden_states"] is embeds
    assert kwargs["encoder_attention_mask"] is mask
    assert pred.shape == state.shape
    pred.square().mean().backward()
    assert unet.lora_A.grad is not None and unet.lora_A.grad > 0


def test_balanced_draw_does_not_average_targets_or_use_evaluation_noise():
    groups = [{"question_id": "q1", "teacher_ids": [f"a{i}" for i in range(10)]},
              {"question_id": "q2", "teacher_ids": [f"b{i}" for i in range(5)]}]
    draws = [balanced_draw(groups, 17, step) for step in range(400)]
    for offset in range(0, 400, 2):
        assert {x[0]["question_id"] for x in draws[offset:offset+2]} == {"q1", "q2"}
    for group, teacher_id, seed, sigma in draws:
        train, held = member_split(group["teacher_ids"])
        assert teacher_id in train and teacher_id not in held
        assert 0 <= sigma < 1
        assert seed not in {stable_seed(17, "heldout-evaluation-noise", i) for i in range(8)}
    assert draws == [balanced_draw(groups, 17, step) for step in range(400)]
    assert set(member_split(groups[0]["teacher_ids"])[0]) == {x[1] for x in draws if x[0]["question_id"] == "q1"}
    assert any(x[3] > .9 for x in draws)
    assert any(x[3] < .1 for x in draws)


def test_official_bridge_endpoints_derivative_and_zero_source_dependency():
    noise = torch.randn(1, 4, 3, 5, dtype=torch.float64)
    target = torch.randn_like(noise)
    start, velocity = official_flow_bridge(noise, target, 1.)
    end, _ = official_flow_bridge(noise, target, 0.)
    torch.testing.assert_close(start, noise, rtol=0, atol=0)
    torch.testing.assert_close(end, target, rtol=0, atol=0)
    state, _ = official_flow_bridge(noise, target, .63)
    nearby, _ = official_flow_bridge(noise, target, .630001)
    torch.testing.assert_close((nearby-state)/.000001, velocity, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(start - velocity, target)
    # Source never appears in the target-side API or derivative.
    import inspect
    assert set(inspect.signature(official_flow_bridge).parameters) == {"noise", "target", "sigma"}


def test_official_timestep_above_old_half_range_matches_upstream_integer_cast():
    unet = CapturingUNet()
    sampler = DifferentiableDreamLiteMobileSampler(unet=unet,
        scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)), vae_scale_factor=8)
    state = torch.ones(1,4,3,5)
    predict_velocity(sampler, state, state*7, .8769, torch.ones(1,4,8), torch.ones(1,4),
                     integer_timestep=True)
    assert unet.observed[1]["timestep"].tolist() == [876.]


def test_new_cli_defaults_to_complete_official_protocol():
    from scripts.train.train_latent_bank_unet import parser, is_official_flow, training_groups
    args = parser().parse_args(["--bank-manifest", "bank.json", "--output-dir", "out"])
    assert is_official_flow(args)
    assert (args.lora_rank, args.lr, args.gradient_accumulation_steps, args.weight_decay) == (16, 5e-5, 4, 1e-4)
    assert args.prompt_style == "official_raw"
    groups = [{"question_id":"q", "teacher_ids":[f"t{i}" for i in range(10)]}]
    selected = training_groups({"groups":groups}, "single")
    assert selected[0]["teacher_ids"] == [member_split(groups[0]["teacher_ids"])[0][0]]
    assert len(groups[0]["teacher_ids"]) == 10


def test_geometry_exposes_collapse_without_claiming_coverage_from_correctness():
    bank = torch.tensor([[[[0., 0.]]], [[[10., 0.]]], [[[0., 10.]]]])
    collapsed = torch.zeros((8, 1, 1, 2))
    result = bank_geometry(collapsed, bank, ["a", "b", "c"], ["c"])
    assert result["nearest_member_counts"] == {"a": 8, "b": 0, "c": 0}
    assert result["generated_pairwise_rms_mean"] == 0
    assert result["nearest_member_entropy"] == 0
    assert result["heldout_bank_fraction_within_radius"] == 0
    dispersed = bank_geometry(bank, bank, ["a", "b", "c"])
    assert dispersed["bank_fraction_within_radius"] == 1
    assert dispersed["nearest_member_fraction_visited"] == 1


def write_bank(tmp_path: Path) -> Path:
    tensor = torch.ones(1, 4, 128, 128)
    latent = tmp_path / "teacher.pt"
    torch.save(tensor, latent)
    prompts = {"original_open": "What music is preferred?\n" + INSTRUCTIONS,
               **{f"paraphrase_{i}": f"Which preferred music, phrasing {i}?\n" + INSTRUCTIONS for i in range(4)}}
    bank = {"schema": "latent-teacher-bank/v1", "bank_status": "sealed", "route": "direct",
        "teachers": [{"teacher_id": "t1", "question_id": "q1", "answer": "ambient", "endpoint_step": 256,
            "target": {"strict_correct": True}, "gold_eos_appended": True,
            "evaluation_generation": {"do_sample": False, "max_new_tokens": 32},
            "latent_path": str(latent), "latent_file_sha256": file_sha256(latent),
            "latent_sha256": canonical_tensor_sha256(tensor)}],
        "groups": [{"question_id": "q1", "answer": "ambient", "teacher_ids": ["t1"],
                    "event_text": "Set the desk's music preference to ambient.", "question_variants": prompts}]}
    path = tmp_path / "bank_manifest.json"
    path.write_text(json.dumps(bank), encoding="utf-8")
    return path


@pytest.mark.parametrize("mutation", ["unsealed", "wrong_prompt", "old_mcq", "no_eos", "wrong_answer", "file_tamper"])
def test_bank_fails_closed_on_protocol_or_target_tampering(tmp_path, mutation):
    path = write_bank(tmp_path)
    bank, teachers = load_teacher_bank(path)
    assert len(teachers) == 1
    if mutation == "unsealed":
        bank["bank_status"] = "partial"
    elif mutation == "wrong_prompt":
        bank["groups"][0]["question_variants"]["original_open"] = "Music?\nUse the memory image to answer. Answer with a short phrase only."
    elif mutation == "old_mcq":
        bank["teachers"][0]["evaluation_generation"]["max_new_tokens"] = 1
    elif mutation == "no_eos":
        bank["teachers"][0]["gold_eos_appended"] = False
    elif mutation == "wrong_answer":
        bank["groups"][0]["answer"] = "jazz"
    else:
        with (tmp_path / "teacher.pt").open("ab") as handle:
            handle.write(b"tamper")
    path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(ValueError):
        load_teacher_bank(path)


def test_lora_checkpoint_resume_preserves_actual_next_update_and_rng(tmp_path):
    model = CapturingUNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    state = torch.tensor([1.])
    for _ in range(2):
        optimizer.zero_grad()
        (state * model.lora_A - 2.).square().sum().backward()
        optimizer.step()
    checkpoint = tmp_path / "last.pt"
    save_training_checkpoint(checkpoint, trainable_module=model, optimizer=optimizer, epoch=0,
        episode_cursor=2, optimizer_step=2, manifest={"bank": "sealed-sha"}, trainer_state={"metrics_row": {"optimizer_step": 2}})
    expected_rng = torch.rand(3)
    optimizer.zero_grad()
    (state * model.lora_A - 2.).square().sum().backward()
    optimizer.step()
    expected = model.lora_A.detach().clone()
    replacement = CapturingUNet()
    second_optimizer = torch.optim.AdamW(replacement.parameters(), lr=.01)
    payload = load_training_checkpoint(checkpoint, trainable_module=replacement, optimizer=second_optimizer,
                                       expected_manifest={"bank": "sealed-sha"})
    assert payload["optimizer_step"] == 2
    torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
    second_optimizer.zero_grad()
    (state * replacement.lora_A - 2.).square().sum().backward()
    second_optimizer.step()
    torch.testing.assert_close(replacement.lora_A, expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        load_training_checkpoint(checkpoint, trainable_module=replacement, optimizer=second_optimizer,
                                  expected_manifest={"bank": "different"})


def test_writer_evaluation_separates_controls_and_new_seed_denominators():
    from scripts.train.train_latent_bank_unet import summarize_evaluation
    rows = []
    for kind, seeds in (("matched", [111, 222]), ("blank", [None]), ("donor", [None])):
        for seed in seeds:
            for i in range(5):
                correct = kind == "matched" and (seed == 111 or i < 4)
                rows.append({"condition": kind, "question_id": "q", "noise_seed": seed, "prompt_id": str(i),
                    "scorer": {"strict_correct": correct, "answer_prefix_token_exact": kind == "matched",
                               "answer_followed_immediately_by_eos": correct and i != 4,
                               "overgeneration": kind == "matched" and not correct}})
    summary = summarize_evaluation(rows, {})
    assert summary["matched_all_five_prompts_correct"] == 1
    assert summary["matched_all_five_prompts_answer_eos"] == 0
    assert summary["cells"]["matched/4"]["answer_eos"] == 0
    assert summary["matched_question_noise_pairs"] == 2
    assert summary["cells"]["matched/4"]["n"] == 2
    assert summary["cells"]["donor/4"]["n"] == 1
    with pytest.raises(RuntimeError, match="all five"):
        summarize_evaluation(rows[1:], {})
    with pytest.raises(RuntimeError, match="Duplicate"):
        summarize_evaluation(rows + [rows[0]], {})


def test_paired_generation_rejects_missing_seeds_and_changed_controls():
    import copy
    from scripts.train.train_latent_bank_unet import paired_evaluation
    rows = [{"question_id": "q", "condition": c, "noise_seed": 123 if c == "matched" else None,
             "prompt_id": "original_open", "query": "Music?", "gold": "ambient", "image_sha256": "pixels",
             "scorer": {"strict_correct": False}} for c in ("matched", "blank", "donor")]
    after = copy.deepcopy(rows)
    after[0]["scorer"]["strict_correct"] = True
    result = paired_evaluation(rows, after)
    assert result["exact_match_delta_count"] == 1
    with pytest.raises(RuntimeError, match="identical"):
        paired_evaluation(rows, after[:-1])
    after[-1]["image_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="control changed"):
        paired_evaluation(rows, after)
