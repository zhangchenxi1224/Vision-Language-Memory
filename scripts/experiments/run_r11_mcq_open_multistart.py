"""Preregistered paired MCQ/open-answer multistart pilot for the old R11 VAE latent.

The prerequisite endpoint replay must pass its technical checks. Every run executes
exactly 256 Adam updates; incomplete experiments cannot produce a completed summary.
No model weights are optimized, and no intermediate checkpoint is selected as best.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.experiments import run_r11_open_answer_replay as replay  # noqa: E402
from scripts.inspire.model_snapshot_manifest import verify_snapshot_binding  # noqa: E402
from scripts.train.latent_r11_vae_oracle import VAELatentOracle, _target_phase, encode_model_latent  # noqa: E402
from vision_memory.data import CYCLIC4, REVERSE_CYCLIC4  # noqa: E402
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval  # noqa: E402
from vision_memory.reader.open_answer import generate_short_answer, score_short_answer  # noqa: E402
from vision_memory.reader.qwen3vl import (  # noqa: E402
    R3_QWEN_READER_RESIZE_CONTRACT, qwen3vl_listwise_choice_ce, qwen3vl_target_only_ce,
)
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import format_mcq_query  # noqa: E402

SCHEMA = "vision_memory.r11-mcq-open-multistart.v1"
ARMS = ("mcq", "open")
PROBE_STEPS = [0, 64, 128, 192, 256]
CONDITIONS = ("matched", "blank", "fixed_donor")
PROMPTS = ("original_open", "paraphrase_open")


def load_config(path: Path, replay_config: Mapping[str, Any]) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "vision_memory.r11-mcq-open-multistart-config.v1":
        raise ValueError("Unexpected multistart config schema.")
    if config["target"] != replay_config["targets"][1]:
        raise ValueError("Pilot target must remain the preregistered old target 1.")
    donor = replay_config["targets"][2]
    if config["fixed_donor"] != {key: donor[key] for key in ("target_index", "segment_id", "artifact")}:
        raise ValueError("Fixed donor must remain old target 2.")
    init = config["initialization"]
    if init["reference_tensor_sha256"] != replay.INITIAL_LATENT_SHA256 or init["shape"] != [1, 4, 128, 128]:
        raise ValueError("Original blank reference contract changed.")
    starts = init["starts"]
    expected_starts = [{"init_id": "blank", "init_seed": None, "rho": 0.0,
                        "included_in_random_start_statistics": False}] + [
        {"init_id": f"noise-seed-{seed:02d}", "init_seed": seed, "rho": 0.1,
         "included_in_random_start_statistics": True} for seed in range(8)
    ]
    if starts != expected_starts:
        raise ValueError("Exactly one blank and eight preregistered random starts are required.")
    training = config["training"]
    optimizer = {"name": "Adam", "learning_rate": 0.05, "betas": [0.9, 0.999], "epsilon": 1e-8,
                 "weight_decay": 0.0, "amsgrad": False, "schedule": "constant", "gradient_clipping": None}
    if training["optimizer"] != optimizer or training["optimizer_steps"] != 256:
        raise ValueError("The fixed 256-step Adam protocol changed.")
    if (training["early_stopping"] or training["best_checkpoint_selection"]
            or training["primary_endpoint_step"] != 256 or training["optimization_seed"] != 0
            or not training["strict_determinism"]):
        raise ValueError("Optimization seed, fixed endpoint and strict determinism are locked.")
    arms = training["arms"]
    if set(arms) != set(ARMS) or arms["open"]["append_eos_to_target"] or arms["open"]["paraphrase_used_in_training"]:
        raise ValueError("Open training requires only the original question and gold text without EOS.")
    if (arms["mcq"]["train_choice_permutations"] != [list(p) for p in CYCLIC4]
            or arms["mcq"]["target_phase"] != _target_phase(config["target"]["segment_id"])):
        raise ValueError("Original R11 MCQ cyclic-view phase changed.")
    run_keys = [(row["init_id"], row["arm"]) for row in training["run_order"]]
    if (len(run_keys) != 18 or len(set(run_keys)) != 18
            or set(run_keys) != {(item["init_id"], arm) for item in starts for arm in ARMS}
            or any(row["run_id"] != f"{row['init_id']}-{row['arm']}" for row in training["run_order"])):
        raise ValueError("Run schedule must contain exactly both arms of all nine starts.")
    logs = config["logging"]
    if (logs["latent_save_steps"] != list(range(257)) or logs["fixed_probe_steps"] != PROBE_STEPS
            or logs["full_adam_and_rng_checkpoint_steps"] != PROBE_STEPS):
        raise ValueError("Every-step latents and five fixed probes/checkpoints are required.")
    evaluation = config["endpoint_evaluation"]
    if (evaluation["generation"] != replay.GENERATION or evaluation["prompts"] != list(PROMPTS)
            or evaluation["image_conditions"] != list(CONDITIONS)
            or evaluation["mcq_choice_permutations"] != [list(p) for p in REVERSE_CYCLIC4]):
        raise ValueError("Endpoint evaluation contract changed.")
    return config


def verify_replay_gate(summary_path: Path, replay_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    terminal = json.loads(summary_path.with_name("terminal.json").read_text(encoding="utf-8"))
    manifest = json.loads(summary_path.with_name("manifest.json").read_text(encoding="utf-8"))
    if (summary.get("technical_passed") is not True or summary.get("complete_records") is not True
            or summary.get("snapshots_unchanged") is not True or not summary.get("all_images_exactly_reproduced")
            or terminal.get("status") != "completed" or terminal.get("technical_passed") is not True):
        raise ValueError("Prerequisite replay did not complete its technical gate.")
    replay_hash = replay.sha256_file(replay_path)
    if (config["activation"]["stage1_config_sha256"] != replay_hash
            or manifest["config_sha256"] != replay_hash):
        raise ValueError("Prerequisite replay configuration hash does not match preregistration.")
    def read_rows(name):
        return [json.loads(line) for line in summary_path.with_name(name).read_text(
            encoding="utf-8").splitlines() if line.strip()]
    rebuilt = replay.aggregate_results(replay.load_config(replay_path), read_rows("generations.jsonl"),
                                       read_rows("mcq_anchor.jsonl"))
    if not rebuilt["mcq_anchor"]["all_correct"]:
        raise ValueError("All 32 old MCQ anchor views must reproduce before optimization.")
    return {"summary_path": str(summary_path.resolve()), "summary_sha256": replay.sha256_file(summary_path),
            "replay_config_sha256": replay_hash, "technical_passed": True,
            "model_snapshot_payloads": manifest["model_snapshot_payloads_start"]}


def make_initial_latent(reference: Tensor, *, seed: int | None, rho: float) -> dict[str, Any]:
    """Independent CPU FP32 noise; all RMS arithmetic in FP64; one final FP32 cast."""
    if reference.device.type != "cpu" or reference.dtype != torch.float32 or not torch.isfinite(reference).all():
        raise ValueError("Reference must be a finite CPU FP32 tensor.")
    if rho < 0 or not math.isfinite(rho) or (seed is None and rho != 0):
        raise ValueError("Blank initialization needs rho=0; noise radius must be finite and nonnegative.")
    if seed is None:
        noise = torch.zeros_like(reference)
        epsilon = noise.double()
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(reference.shape, generator=generator, device="cpu", dtype=torch.float32)
        epsilon = noise.double() / noise.double().square().mean().sqrt()
    reference_rms = reference.double().square().mean().sqrt()
    latent = (reference.double() + rho * reference_rms * epsilon).float()
    return {"reference_fp32": reference.clone(), "raw_noise_fp32": noise, "epsilon_fp64": epsilon,
            "latent_fp32": latent, "seed": seed, "rho": rho, "reference_rms": float(reference_rms),
            "latent_sha256": canonical_tensor_sha256(latent), "noise_sha256": canonical_tensor_sha256(noise),
            "epsilon_sha256": canonical_tensor_sha256(epsilon)}


def training_case(target: Mapping[str, Any], arm: str, step_zero: int) -> dict[str, Any]:
    if arm == "open":
        return {"query": target["inputs"]["original_open"], "target": target["scorer_metadata"]["gold"],
                "training_view": None, "permutation": None}
    if arm != "mcq":
        raise ValueError(f"Unknown arm {arm!r}.")
    view = (step_zero + _target_phase(target["segment_id"])) % 4
    permutation = CYCLIC4[view]
    choices = tuple(target["original"]["choices"][index] for index in permutation)
    return {"query": format_mcq_query(target["original"]["query"], choices), "choices": choices,
            "target_index": permutation.index(target["original"]["answer_index"]),
            "training_view": view, "permutation": list(permutation)}


def capture_rng() -> dict[str, Any]:
    import numpy as np
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (np_state[0], torch.from_numpy(np_state[1].astype("int64")), *np_state[2:]),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state: Mapping[str, Any]) -> None:
    import numpy as np
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np_state[1].numpy().astype("uint32"), *np_state[2:]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def save_tensor_payload(path: Path, payload: Mapping[str, Any]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(cpu_tree(payload), temporary)
    temporary.replace(path)
    return replay.sha256_file(path)


def loss_output(*, image: Tensor, target: Mapping[str, Any], arm: str, step_zero: int,
                reader: Any, processor: Any, reader_device: torch.device, require_grad: bool):
    case = training_case(target, arm, step_zero)
    common = {"model": reader, "processor": processor, "image": image[0].to(reader_device),
              "query": case["query"], "device": reader_device, "require_image_grad": require_grad,
              "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT, "deterministic_ce": True}
    if arm == "mcq":
        return qwen3vl_listwise_choice_ce(**common, choices=case["choices"], target_index=case["target_index"])
    return qwen3vl_target_only_ce(**common, target=case["target"])


def choice_margin(logits: Tensor, target_index: int) -> float:
    flat = logits.detach().float().reshape(-1)
    wrong = torch.cat((flat[:target_index], flat[target_index + 1:]))
    return float(flat[target_index] - wrong.max())


def repeated_gradient_check(*, oracle: VAELatentOracle, arm: str, target: Mapping[str, Any],
                            reader: Any, processor: Any, reader_device: torch.device) -> dict[str, Any]:
    losses, gradients = [], []
    with preserve_rng():
        state = capture_rng()
        for _ in range(2):
            restore_rng(state)
            oracle.latent_fp32.grad = None
            output = loss_output(image=oracle.image(), target=target, arm=arm, step_zero=0, reader=reader,
                                 processor=processor, reader_device=reader_device, require_grad=True)
            output.loss.backward()
            gradient = oracle.latent_fp32.grad
            if (gradient is None or not torch.isfinite(gradient).all() or not torch.isfinite(output.loss)
                    or not bool((gradient != 0).any())):
                raise RuntimeError("Initial objective must confirm finite loss and a nonzero connected latent gradient.")
            losses.append(output.loss.detach().cpu().clone())
            gradients.append(gradient.detach().cpu().clone())
            del output
        oracle.latent_fp32.grad = None
    result = {"loss_exactly_equal": torch.equal(*losses), "gradient_exactly_equal": torch.equal(*gradients),
              "loss_max_abs_difference": float((losses[0] - losses[1]).abs().max()),
              "gradient_max_abs_difference": float((gradients[0] - gradients[1]).abs().max()),
              "gradient_sha256": [canonical_tensor_sha256(item) for item in gradients]}
    if not result["loss_exactly_equal"] or not result["gradient_exactly_equal"]:
        raise RuntimeError(f"Repeated objective is not deterministic: {result}")
    return result


@torch.no_grad()
def fixed_probes(*, oracle: VAELatentOracle, target: Mapping[str, Any], reader: Any, processor: Any,
                 reader_device: torch.device, step: int) -> list[dict[str, Any]]:
    with preserve_rng():
        image = oracle.image()
        rows = []
        for view in range(4):
            step_zero = (view - _target_phase(target["segment_id"])) % 4
            output = loss_output(image=image, target=target, arm="mcq", step_zero=step_zero,
                                 reader=reader, processor=processor, reader_device=reader_device, require_grad=False)
            case = training_case(target, "mcq", step_zero)
            rows.append({"optimizer_step": step, "metric_state": "post_step" if step else "initial",
                         "probe": "mcq_forward", "view_index": view, "permutation": case["permutation"],
                         "ce": float(output.loss), "choice_logits": output.choice_logits.float().cpu().tolist(),
                         "margin": choice_margin(output.choice_logits, case["target_index"]),
                         "correct": int(output.choice_logits.argmax()) == case["target_index"]})
        output = loss_output(image=image, target=target, arm="open", step_zero=0, reader=reader,
                             processor=processor, reader_device=reader_device, require_grad=False)
        rows.append({"optimizer_step": step, "metric_state": "post_step" if step else "initial",
                     "probe": "open_gold_ce", "ce": float(output.loss),
                     "target_token_ids": output.target_ids.cpu().tolist(), "append_eos": False})
        if any(not math.isfinite(row["ce"]) for row in rows):
            raise RuntimeError("Nonfinite fixed probe.")
        return rows


@torch.no_grad()
def endpoint_rows(*, run_spec: Mapping[str, Any], image: Tensor, controls: Mapping[str, Tensor],
                  target: Mapping[str, Any], reader: Any, processor: Any,
                  reader_device: torch.device, on_generation: Any = None,
                  on_mcq: Any = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generations, mcq = [], []
    with preserve_rng():
        for condition in CONDITIONS:
            pixels = image if condition == "matched" else controls[condition]
            image_hash = canonical_tensor_sha256(pixels)
            for prompt_id in PROMPTS:
                query = target["inputs"][prompt_id]
                generated = generate_short_answer(model=reader, processor=processor, image=pixels.to(reader_device),
                                                  query=query, device=reader_device, max_new_tokens=32, do_sample=False)
                generations.append({**run_spec, "optimizer_step": 256, "condition": condition,
                                    "prompt_id": prompt_id, "query": query, "image_sha256": image_hash, **generated,
                                    "scorer": score_short_answer(generated["raw"], target["scorer_metadata"]["gold"])})
                if on_generation is not None:
                    on_generation(generations[-1])
                print(json.dumps({"stage": "endpoint_generation", "run_id": run_spec["run_id"],
                                  "condition": condition, "prompt_id": prompt_id, "raw": generated["raw"]}), flush=True)
        for view, permutation in enumerate(REVERSE_CYCLIC4):
            choices = tuple(target["original"]["choices"][index] for index in permutation)
            target_index = permutation.index(target["original"]["answer_index"])
            query = format_mcq_query(target["original"]["query"], choices)
            output = qwen3vl_listwise_choice_ce(
                model=reader, processor=processor, image=image[0].to(reader_device), query=query, choices=choices,
                target_index=target_index, device=reader_device, require_image_grad=False,
                reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT, deterministic_ce=True,
            )
            if not torch.isfinite(output.loss) or not torch.isfinite(output.choice_logits).all():
                raise RuntimeError("Nonfinite endpoint MCQ scores.")
            mcq.append({**run_spec, "optimizer_step": 256, "condition": "matched", "view_index": view,
                        "permutation": list(permutation), "query": query, "choices": list(choices),
                        "target_index": target_index, "correct": int(output.choice_logits.argmax()) == target_index,
                        "margin": choice_margin(output.choice_logits, target_index),
                        "ce": float(output.loss), "choice_logits": output.choice_logits.float().cpu().tolist(),
                        "choice_mean_nll": output.choice_mean_nll.float().cpu().tolist()})
            if on_mcq is not None:
                on_mcq(mcq[-1])
    return generations, mcq


def run_one(*, run_spec: Mapping[str, Any], initial: Mapping[str, Any], reference: Tensor,
            vae: Any, reader: Any, processor: Any, reader_device: torch.device, vae_device: torch.device,
            config: Mapping[str, Any], controls: Mapping[str, Tensor], output_dir: Path) -> dict[str, Any]:
    run_dir = output_dir / "runs" / run_spec["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "latents").mkdir()
    (run_dir / "checkpoints").mkdir()
    started = time.perf_counter()
    try:
        configure_strict_cuda_determinism(config["training"]["optimization_seed"])
        oracle = VAELatentOracle(vae=vae, initial_latent=initial["latent_fp32"].to(vae_device),
                                compute_dtype=torch.bfloat16)
        initial_hash = canonical_tensor_sha256(oracle.latent_fp32)
        if initial_hash != initial["latent_sha256"]:
            raise RuntimeError("Paired initialization changed when copied to the VAE device.")
        optimizer = torch.optim.Adam([oracle.latent_fp32], lr=0.05, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.0, amsgrad=False)
        replay.write_json(run_dir / "manifest.json", {**run_spec, "initial_latent_sha256": initial_hash,
                          "reference_sha256": canonical_tensor_sha256(reference), "optimizer_steps": 256,
                          "initial_rng_sha256": canonical_tensor_sha256(torch.get_rng_state()),
                          "gold_eos_appended": False, "target_phase": _target_phase(config["target"]["segment_id"])})
        technical = repeated_gradient_check(oracle=oracle, arm=run_spec["arm"], target=config["target"],
                                            reader=reader, processor=processor, reader_device=reader_device)
        replay.write_json(run_dir / "initial_reproducibility.json", technical)
        history: list[Tensor] = []
        metric_count, probe_count = 0, 0
        for step in range(257):
            current = oracle.latent_fp32.detach().cpu().clone()
            latent_path = run_dir / "latents" / f"step-{step:03d}.pt"
            file_hash = save_tensor_payload(latent_path, {"optimizer_step": step, "latent_fp32": current})
            replay.append_jsonl(run_dir / "latent_index.jsonl", {"optimizer_step": step,
                "path": str(latent_path.relative_to(run_dir)), "file_sha256": file_hash,
                "latent_sha256": canonical_tensor_sha256(current)})
            if step in PROBE_STEPS:
                checkpoint_path = run_dir / "checkpoints" / f"step-{step:03d}.pt"
                checkpoint_hash = save_tensor_payload(checkpoint_path, {
                    "schema": SCHEMA + ".checkpoint", **run_spec, "optimizer_step": step,
                    "latent_fp32": current, "initial_latent_fp32": initial["latent_fp32"],
                    "optimizer": optimizer.state_dict(), "rng": capture_rng(),
                })
                replay.append_jsonl(run_dir / "checkpoint_index.jsonl", {
                    "optimizer_step": step, "path": str(checkpoint_path.relative_to(run_dir)),
                    "file_sha256": checkpoint_hash})
                probes = fixed_probes(oracle=oracle, target=config["target"], reader=reader,
                                     processor=processor, reader_device=reader_device, step=step)
                for probe in probes:
                    replay.append_jsonl(run_dir / "fixed_probes.jsonl", {**run_spec, **probe})
                probe_count += len(probes)
            if step == 256:
                break
            history.append(current)
            history = history[-4:]
            optimizer.zero_grad(set_to_none=True)
            output = loss_output(image=oracle.image(), target=config["target"], arm=run_spec["arm"],
                                 step_zero=step, reader=reader, processor=processor,
                                 reader_device=reader_device, require_grad=True)
            if not torch.isfinite(output.loss):
                raise RuntimeError("Nonfinite optimization loss.")
            output.loss.backward()
            gradient = oracle.latent_fp32.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("No finite latent gradient.")
            gradient_cpu = gradient.detach().double().cpu()
            loss_value = float(output.loss.detach())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updated = oracle.latent_fp32.detach().cpu().clone()
            if not torch.isfinite(updated).all():
                raise RuntimeError("Adam produced nonfinite latent values.")
            update = updated.double() - current.double()
            metrics = {**run_spec, "optimizer_step": step + 1, "loss_before_step": loss_value,
                       "loss_state": "pre_step", "gradient_state": "pre_step", "latent_state": "post_step",
                       "gradient_l2": float(gradient_cpu.norm()),
                       "gradient_rms": float(gradient_cpu.square().mean().sqrt()),
                       "gradient_nonzero_fraction": float((gradient_cpu != 0).double().mean()),
                       "gradient_all_zero": not bool((gradient_cpu != 0).any()),
                       "actual_update_l2": float(update.norm()), "actual_update_rms": float(update.square().mean().sqrt()),
                       "four_step_update_rms": float((updated.double() - history[0].double()).square().mean().sqrt())
                       if len(history) == 4 else None,
                       "latent_rms": float(updated.double().square().mean().sqrt()),
                       "delta_from_own_start_rms": float((updated.double() - initial["latent_fp32"].double())
                                                        .square().mean().sqrt()),
                       "delta_from_shared_reference_rms": float((updated.double() - reference.double())
                                                               .square().mean().sqrt()),
                       "training_view_if_mcq": training_case(config["target"], run_spec["arm"], step)["training_view"],
                       "elapsed_seconds": time.perf_counter() - started}
            replay.append_jsonl(run_dir / "metrics.jsonl", metrics)
            metric_count += 1
            print(json.dumps({"stage": "optimizer_step", "run_id": run_spec["run_id"], "step": step + 1,
                              "loss_before_step": loss_value, "elapsed_seconds": metrics["elapsed_seconds"]}), flush=True)
            del output, gradient, gradient_cpu
        with torch.no_grad():
            endpoint_image = oracle.image().cpu()
        endpoint_hash = save_tensor_payload(run_dir / "endpoint_raw.pt", {
            "schema": SCHEMA + ".endpoint", **run_spec, "optimizer_step": 256,
            "latent_fp32": oracle.latent_fp32, "image": endpoint_image,
            "initial_latent_sha256": initial_hash,
        })
        def record_generation(row):
            replay.append_jsonl(run_dir / "generations.jsonl", row)
            replay.append_jsonl(output_dir / "generations.jsonl", row)

        def record_mcq(row):
            replay.append_jsonl(run_dir / "mcq_endpoint.jsonl", row)
            replay.append_jsonl(output_dir / "mcq_endpoint.jsonl", row)

        generations, mcq = endpoint_rows(run_spec=run_spec, image=endpoint_image, controls=controls,
                                         target=config["target"], reader=reader, processor=processor,
                                         reader_device=reader_device, on_generation=record_generation, on_mcq=record_mcq)
        replay.frozen_audit(vae, reader)
        result = {**run_spec, "status": "completed", "optimizer_steps": metric_count,
                  "probe_count": probe_count, "latent_count": 257, "checkpoint_count": 5,
                  "generation_count": len(generations), "mcq_count": len(mcq),
                  "initial_latent_sha256": initial_hash, "endpoint_file_sha256": endpoint_hash,
                  "endpoint_latent_sha256": canonical_tensor_sha256(oracle.latent_fp32),
                  "elapsed_seconds": time.perf_counter() - started}
        replay.write_json(run_dir / "terminal.json", result)
        return result
    except BaseException as error:
        replay.write_json(run_dir / "terminal.json", {**run_spec, "status": "failed", "error": str(error),
                          "error_type": type(error).__name__, "traceback": traceback.format_exc()})
        raise


def aggregate_results(config: Mapping[str, Any], completed_runs: Sequence[Mapping[str, Any]],
                      generations: Sequence[Mapping[str, Any]], mcq_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    specs = {row["run_id"]: row for row in config["training"]["run_order"]}
    if len(completed_runs) != 18 or {row["run_id"] for row in completed_runs} != set(specs):
        raise ValueError("Cannot aggregate incomplete or duplicated runs; all 18 are required.")
    for row in completed_runs:
        if (row["status"] != "completed" or row["optimizer_steps"] != 256 or row["latent_count"] != 257
                or row["probe_count"] != 25 or row["checkpoint_count"] != 5
                or row["generation_count"] != 6 or row["mcq_count"] != 4):
            raise ValueError("Every run requires all updates, checkpoints, probes and endpoint records.")
    expected_open = {(run, condition, prompt) for run in specs for condition in CONDITIONS for prompt in PROMPTS}
    keys = [(row["run_id"], row["condition"], row["prompt_id"]) for row in generations]
    if len(keys) != 108 or len(set(keys)) != 108 or set(keys) != expected_open:
        raise ValueError("Cannot aggregate incomplete, duplicate or unexpected open endpoint rows.")
    expected_mcq = {(run, view) for run in specs for view in range(4)}
    mcq_keys = [(row["run_id"], row["view_index"]) for row in mcq_rows]
    if len(mcq_keys) != 72 or len(set(mcq_keys)) != 72 or set(mcq_keys) != expected_mcq:
        raise ValueError("Cannot aggregate incomplete, duplicate or unexpected MCQ endpoint rows.")
    gold = config["target"]["scorer_metadata"]["gold"]
    for row in generations:
        if (row["optimizer_step"] != 256 or row["query"] != config["target"]["inputs"][row["prompt_id"]]
                or row["scorer"] != score_short_answer(row["raw"], gold)):
            raise ValueError("Endpoint query, checkpoint or score drifted from the locked protocol.")
    for row in mcq_rows:
        if (row["optimizer_step"] != 256 or row["permutation"] != list(REVERSE_CYCLIC4[row["view_index"]])
                or not isinstance(row["correct"], bool) or not math.isfinite(row["ce"])):
            raise ValueError("MCQ endpoint view or metrics drifted.")
    by_key, mcq_key = dict(zip(keys, generations)), dict(zip(mcq_keys, mcq_rows))
    result_by_run = {row["run_id"]: row for row in completed_runs}
    behavior = {}
    for run in specs:
        def correct(condition, prompt):
            return by_key[(run, condition, prompt)]["scorer"]["strict_correct"]
        behavior[run] = {"open_original": correct("matched", "original_open"),
                         "open_robust": all(correct("matched", prompt) for prompt in PROMPTS),
                         "mcq_all4": all(mcq_key[(run, view)]["correct"] for view in range(4)),
                         "controls": {condition: {prompt: correct(condition, prompt) for prompt in PROMPTS}
                                      for condition in ("blank", "fixed_donor")}}
    pairs = []
    blank = {}
    for start in config["initialization"]["starts"]:
        init_id = start["init_id"]
        left, right = (f"{init_id}-{arm}" for arm in ARMS)
        if result_by_run[left]["initial_latent_sha256"] != result_by_run[right]["initial_latent_sha256"]:
            raise ValueError("Paired arms did not use identical initial latent tensors.")
        pair = {"init_id": init_id, "seed": start["init_seed"], "mcq": behavior[left], "open": behavior[right]}
        if start["included_in_random_start_statistics"]:
            pairs.append(pair)
        else:
            blank = pair
    by_arm = {}
    for arm in ARMS:
        by_arm[arm] = {name: sum(pair[arm][name] for pair in pairs) / 8
                       for name in ("open_original", "open_robust", "mcq_all4")}
        by_arm[arm]["matched_minus_controls_original"] = {
            control: sum(int(pair[arm]["open_original"]) - int(pair[arm]["controls"][control]["original_open"])
                         for pair in pairs) / 8 for control in ("blank", "fixed_donor")}
    return {"schema": SCHEMA + ".summary", "complete_records": True, "formal_success": False,
            "run_count": 18, "optimizer_steps": 4608, "generation_count": 108, "mcq_count": 72,
            "random_start_count": 8, "paired_by_seed": pairs, "blank_separate": blank, "by_arm": by_arm,
            "paired_open_minus_mcq_accuracy": {name: by_arm["open"][name] - by_arm["mcq"][name]
                                               for name in ("open_original", "open_robust", "mcq_all4")},
            "scope": "single-target exploratory pilot; geometry pair counts are not independent observations"}


def pairwise_geometry(values: Tensor) -> dict[str, Any]:
    flat = values.double().reshape(values.shape[0], -1)
    squared_norm = flat.square().sum(dim=1)
    gram = flat @ flat.T
    rmse = ((squared_norm[:, None] + squared_norm[None, :] - 2 * gram).clamp_min(0) / flat.shape[1]).sqrt()
    denominator = squared_norm.sqrt()[:, None] * squared_norm.sqrt()[None, :]
    cosine = gram / denominator.clamp_min(torch.finfo(torch.float64).tiny)
    indices = torch.triu_indices(flat.shape[0], flat.shape[0], offset=1)
    selected_rmse = rmse[indices[0], indices[1]]
    valid = denominator > 0
    cosine_list = [[float(cosine[i, j]) if valid[i, j] else None for j in range(len(flat))] for i in range(len(flat))]
    valid_pairs = valid[indices[0], indices[1]]
    pair_cosines = cosine[indices[0], indices[1]][valid_pairs]
    return {"rmse_matrix": rmse.tolist(), "cosine_matrix": cosine_list,
            "mean_pairwise_squared_rmse": float(selected_rmse.square().mean()),
            "mean_pairwise_rmse": float(selected_rmse.mean()),
            "mean_pairwise_cosine": float(pair_cosines.mean()) if pair_cosines.numel() else None,
            "pair_count": selected_rmse.numel(), "independent_start_count": len(flat)}


def vector_geometry(value: Tensor, other: Tensor) -> dict[str, Any]:
    left, right = value.double().reshape(-1), other.double().reshape(-1)
    norm_product = left.norm() * right.norm()
    difference = left - right
    return {"rmse": float(difference.square().mean().sqrt()), "l2": float(difference.norm()),
            "cosine": float(left.dot(right) / norm_product) if norm_product > 0 else None}


def analyze_geometry(config: Mapping[str, Any], output_dir: Path, reference: Tensor,
                     initials: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    starts = [item["init_id"] for item in config["initialization"]["starts"]
              if item["included_in_random_start_statistics"]]
    initial_stack = torch.stack([initials[key]["latent_fp32"] for key in starts])
    initial_geometry = pairwise_geometry(initial_stack)
    denominator = initial_geometry["mean_pairwise_squared_rmse"]
    endpoints: dict[str, Tensor] = {}
    summaries = {}
    for arm in ARMS:
        curves = []
        for step in range(257):
            def read(init_id: str) -> Tensor:
                path = output_dir / "runs" / f"{init_id}-{arm}" / "latents" / f"step-{step:03d}.pt"
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload["optimizer_step"] != step:
                    raise ValueError("Geometry source step mismatch.")
                return payload["latent_fp32"]
            stack = torch.stack([read(init_id) for init_id in starts])
            blank = read("blank")
            raw = pairwise_geometry(stack)
            own_delta = pairwise_geometry(stack.double() - initial_stack.double())
            reference_delta = pairwise_geometry(stack.double() - reference.double())
            curve = {"optimizer_step": step, "raw_latent": raw, "delta_own_start": own_delta,
                     "delta_shared_reference": reference_delta,
                     "C_mean_pairwise_squared_rmse": raw["mean_pairwise_squared_rmse"],
                     "C_over_C0": raw["mean_pairwise_squared_rmse"] / denominator,
                     "per_seed": [{"init_id": key, "latent_rms": float(stack[index].double().square().mean().sqrt()),
                                   "to_reference": vector_geometry(stack[index], reference),
                                   "to_own_start": vector_geometry(stack[index], initial_stack[index]),
                                   "to_same_arm_blank": vector_geometry(stack[index], blank)}
                                  for index, key in enumerate(starts)]}
            replay.append_jsonl(output_dir / f"geometry_{arm}.jsonl", curve)
            curves.append(curve)
            if step == 256:
                endpoints[arm] = stack
        upper = torch.triu_indices(8, 8, offset=1)
        first = torch.tensor(initial_geometry["rmse_matrix"], dtype=torch.float64)[upper[0], upper[1]]
        last = torch.tensor(curves[-1]["raw_latent"]["rmse_matrix"], dtype=torch.float64)[upper[0], upper[1]]
        association = float(torch.corrcoef(torch.stack((first, last)))[0, 1]) if first.std() > 0 and last.std() > 0 else None
        summaries[arm] = {"initial": curves[0], "endpoint": curves[-1],
                          "initial_final_distance_pearson_descriptive_only": association}
    return {"by_arm": summaries, "random_start_order": starts, "independent_start_count": 8,
            "pair_count_per_arm": 28, "pairwise_observations_independent": False,
            "C_definition": "mean over unordered pairs of mean-coordinate squared distance; not square-root distance",
            "paired_cross_arm_endpoint": [{"init_id": key, **vector_geometry(endpoints["mcq"][index],
                                                                                 endpoints["open"][index])}
                                          for index, key in enumerate(starts)]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    replay_config = replay.load_config(args.replay_config)
    config = load_config(args.config, replay_config)
    prerequisite = verify_replay_gate(args.replay_summary, args.replay_config, config)
    vae_device, reader_device = torch.device(args.vae_device), torch.device(args.reader_device)
    if (vae_device.type != "cuda" or reader_device.type != "cuda" or vae_device == reader_device
            or not torch.cuda.is_available()):
        raise ValueError("Original BF16 pilot requires two distinct CUDA devices.")
    determinism = configure_strict_cuda_determinism(config["training"]["optimization_seed"])
    bindings = replay.snapshot_bindings(args)
    if bindings != prerequisite["model_snapshot_payloads"]:
        raise ValueError("Pilot model snapshot bindings differ from the technically passed replay.")
    source_paths = [Path(__file__), ROOT / "scripts/experiments/run_r11_open_answer_replay.py",
                    ROOT / "src/vision_memory/reader/open_answer.py", ROOT / "src/vision_memory/reader/qwen3vl.py",
                    ROOT / "scripts/train/latent_r11_vae_oracle.py", ROOT / "models.lock.json"]
    manifest = {"schema": SCHEMA + ".manifest", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "config_sha256": replay.sha256_file(args.config), "prerequisite": prerequisite,
                "source_sha256": {str(path.relative_to(ROOT)): replay.sha256_file(path) for path in source_paths},
                "git_commit": replay.command_output(["git", "rev-parse", "HEAD"]),
                "git_status": replay.command_output(["git", "status", "--porcelain"]),
                "runtime": replay.runtime_versions(), "determinism": determinism,
                "model_snapshot_payloads_start": bindings, "vae_device": str(vae_device),
                "reader_device": str(reader_device), "controls_cached": False,
                "primary_endpoint_step": 256, "open_target_appends_eos": False, "formal_success": False}
    replay.write_json(args.output_dir / "manifest.json", manifest)
    vae = replay.load_vae(args.dreamlite, vae_device)
    processor, reader = replay.r4._load_reader(args, reader_device, torch.bfloat16)
    reader.requires_grad_(False).eval()
    replay.frozen_audit(vae, reader)
    with torch.no_grad():
        rgb = replay.r4._initial_rgb_tensor(resolution=1024, device=vae_device, dtype=torch.bfloat16)
        reference = encode_model_latent(vae, rgb).float().cpu()
        if canonical_tensor_sha256(reference) != replay.INITIAL_LATENT_SHA256:
            raise RuntimeError("Original initial blank latent SHA256 did not reproduce.")
        donor_payload, donor_audit = replay.load_endpoint(args.old_run_root, replay_config["targets"][2])
        controls = {
            "blank": decode_model_latents_unit_interval(vae, reference.to(vae_device, dtype=torch.bfloat16),
                                                          clamp=True).cpu(),
            "fixed_donor": decode_model_latents_unit_interval(
                vae, donor_payload["latent_fp32"].to(vae_device, dtype=torch.bfloat16), clamp=True).cpu(),
        }
        if not torch.equal(controls["fixed_donor"], donor_payload["image"]):
            raise RuntimeError("Fixed donor image no longer reproduces its old endpoint.")
    replay.write_json(args.output_dir / "control_audit.json", {"donor": donor_audit,
                     "reference_sha256": canonical_tensor_sha256(reference),
                     "image_sha256": {key: canonical_tensor_sha256(value) for key, value in controls.items()}})
    save_tensor_payload(args.output_dir / "fixed_controls.pt", {"reference_fp32": reference, **controls})
    initials = {}
    (args.output_dir / "initials").mkdir()
    for spec in config["initialization"]["starts"]:
        initial = make_initial_latent(reference, seed=spec["init_seed"], rho=spec["rho"])
        initials[spec["init_id"]] = initial
        path = args.output_dir / "initials" / f"{spec['init_id']}.pt"
        file_hash = save_tensor_payload(path, {**spec, **initial})
        replay.append_jsonl(args.output_dir / "initial_index.jsonl", {**spec,
            "latent_sha256": initial["latent_sha256"], "file_sha256": file_hash})
    completed = []
    for spec in config["training"]["run_order"]:
        completed.append(run_one(run_spec=spec, initial=initials[spec["init_id"]], reference=reference,
                                 vae=vae, reader=reader, processor=processor, vae_device=vae_device,
                                 reader_device=reader_device, config=config, controls=controls, output_dir=args.output_dir))
        replay.write_json(args.output_dir / "progress.json", {"completed_runs": completed, "expected_run_count": 18})
    def read_rows(name):
        return [json.loads(line) for line in (args.output_dir / name).read_text(
            encoding="utf-8").splitlines() if line.strip()]
    summary = aggregate_results(config, completed, read_rows("generations.jsonl"), read_rows("mcq_endpoint.jsonl"))
    frozen_end = replay.frozen_audit(vae, reader)
    end_bindings = {key: verify_snapshot_binding(binding) for key, binding in bindings.items()}
    if bindings != end_bindings:
        raise RuntimeError("Model snapshot bindings changed during optimization.")
    replay.write_json(args.output_dir / "model_snapshot_verification_end.json", {
        "bindings": end_bindings, "passed": True, "frozen_parameters": frozen_end})
    geometry = analyze_geometry(config, args.output_dir, reference, initials)
    replay.write_json(args.output_dir / "geometry_summary.json", geometry)
    summary.update({"technical_passed": True, "snapshots_unchanged": True,
                    "geometry_summary": "geometry_summary.json", "elapsed_seconds": time.perf_counter() - started})
    replay.write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "replay-config", "replay-summary", "old-run-root", "dreamlite", "reader", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        summary = run(args)
    except BaseException as error:
        replay.write_json(args.output_dir / "terminal.json", {"schema": SCHEMA + ".terminal", "status": "failed",
            "technical_passed": False, "formal_success": False, "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - started})
        traceback.print_exc()
        return 1
    replay.write_json(args.output_dir / "terminal.json", {"schema": SCHEMA + ".terminal", "status": "completed",
        "technical_passed": True, "formal_success": False, "run_count": summary["run_count"],
        "optimizer_steps": summary["optimizer_steps"], "elapsed_seconds": time.perf_counter() - started})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
