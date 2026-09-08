"""Stage A worker: optimize only FP32 x_T through the complete frozen DreamLite.

The run specification is a preregistered row with stage, run_id, target_index,
seed, distribution, scale, repeat. Repeats never enter the initialization seed.
CUDA execution is deliberately separate from importable CPU contract helpers.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from vision_memory.repro import canonical_object_sha256, canonical_tensor_sha256  # noqa: E402

SCHEMA = "vision_memory.frozen-oracle-geometry-worker.v1"
STEPS = 256
CHECKPOINT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256)
LEARNING_RATE = 0.05


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json_argument(value: str) -> Any:
    return json.loads(value) if value.lstrip().startswith(("{", "[")) else json.loads(
        Path(value).read_text(encoding="utf-8-sig")
    )


def validate_run_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    required = {"stage", "run_id", "target_index", "seed", "distribution", "scale", "repeat"}
    if required - set(spec):
        raise ValueError(f"Missing run fields: {sorted(required - set(spec))}")
    result = dict(spec)
    for name in ("target_index", "seed", "repeat"):
        if name == "repeat" and result[name] is None:
            continue
        if isinstance(result[name], bool) or not isinstance(result[name], int) or result[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if result["target_index"] >= 8:
        raise ValueError("This Stage A worker only permits the eight locked F1 targets")
    if not isinstance(result["run_id"], str) or not result["run_id"] or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in result["run_id"]
    ):
        raise ValueError("run_id must be a nonempty safe identifier")
    if isinstance(result["scale"], bool) or not np.isfinite(float(result["scale"])) or float(result["scale"]) <= 0:
        raise ValueError("scale must be finite and positive")
    return result


def select_run_spec(manifest: Any, run_id: str) -> dict[str, Any]:
    rows = manifest if isinstance(manifest, list) else manifest.get("runs", manifest.get("run_specs", []))
    matches = [row for row in rows if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest run {run_id!r}, found {len(matches)}")
    return validate_run_spec(matches[0])


def initial_tensor(shape: Sequence[int], spec: Mapping[str, Any]) -> torch.Tensor:
    from vision_memory.training.frozen_oracle_geometry import make_initial_array

    if "initial_xT_path" in spec:
        if not str(spec["stage"]).startswith("A5"):
            raise ValueError("External initial_xT is reserved for preregistered Stage A5 reoptimization")
        path = Path(spec["initial_xT_path"])
        if file_sha256(path) != spec.get("initial_xT_file_sha256"):
            raise ValueError("Reoptimization initial_xT file SHA binding mismatch")
        array = np.load(path, allow_pickle=False)
    else:
        array = make_initial_array(shape, spec["distribution"], spec["seed"], float(spec["scale"]))
    value = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
    if tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
        raise ValueError("Initialization returned a wrong-shaped or nonfinite array")
    return value


def tensor_record(value: torch.Tensor) -> dict[str, Any]:
    cpu = value.detach().cpu().contiguous()
    double = cpu.double()
    if not torch.isfinite(double).all():
        raise RuntimeError("Nonfinite tensor in audited path")
    centered = double - double.mean()
    variance = float(centered.square().mean())
    return {
        "sha256": canonical_tensor_sha256(cpu), "shape": list(cpu.shape), "dtype": str(cpu.dtype),
        "mean": float(double.mean()), "rms": float(double.square().mean().sqrt()),
        "norm": float(double.norm()), "min": float(double.min()), "max": float(double.max()),
        "variance": variance, "kurtosis": float(centered.pow(4).mean()) / variance ** 2 if variance > 0 else None,
        "max_abs": float(double.abs().max()),
    }


def save_array(path: Path, value: torch.Tensor) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu = value.detach().cpu().contiguous()
    with path.with_suffix(".npy.tmp").open("wb") as handle:
        np.save(handle, cpu.numpy(), allow_pickle=False)
    path.with_suffix(".npy.tmp").replace(path)
    return {"path": str(path.resolve()), "file_sha256": file_sha256(path), **tensor_record(cpu)}


def assert_frozen_gradient_contract(oracle: Any, reader: torch.nn.Module) -> dict[str, Any]:
    trainable = [name for name, parameter in oracle.named_parameters() if parameter.requires_grad]
    if trainable != ["x_T_fp32"] or oracle.x_T_fp32.dtype != torch.float32:
        raise RuntimeError(f"Only FP32 x_T may train, observed {trainable}")
    frozen = {"dreamlite_unet": oracle.unet, "vae": oracle.vae, "text_encoder": oracle.text_encoder, "reader": reader}
    for name, module in frozen.items():
        if module.training or any(parameter.requires_grad or parameter.grad is not None for parameter in module.parameters()):
            raise RuntimeError(f"Frozen {name} is training, requires grad, or received parameter gradients")
    gradient = oracle.x_T_fp32.grad
    if gradient is None or not torch.isfinite(gradient).all() or not bool(torch.count_nonzero(gradient)):
        raise RuntimeError("The complete DreamLite/VAE/Reader path did not produce a finite nonzero x_T gradient")
    return {"only_trainable": trainable, "all_frozen_without_parameter_gradients": True,
            "gradient": tensor_record(gradient), "gradient_nonzero_fraction": float((gradient != 0).double().mean())}


def score_record(output: Any, target_index: int) -> dict[str, Any]:
    logits = output.choice_logits.detach().float().cpu().reshape(-1)
    if logits.numel() != 4 or not torch.isfinite(logits).all():
        raise RuntimeError("Reader must return four finite choice logits")
    loss = output.loss
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not torch.isfinite(loss):
        raise RuntimeError("Reader returned invalid CE")
    alternatives = torch.cat((logits[:target_index], logits[target_index + 1:]))
    margin = float(logits[target_index] - alternatives.max())
    return {"ce": float(loss.detach()), "choice_logits": logits.tolist(), "target_index_ordered": target_index,
            "predicted_index_ordered": int(logits.argmax()), "margin": margin,
            "correct": int(logits.argmax()) == target_index, "strict_correct": margin > 0.0}


def strict_endpoint_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normal = [row for row in rows if row["condition"] == "normal"]
    reset = [row for row in rows if row["condition"] == "reset"]
    for name, values in (("normal", normal), ("reset", reset)):
        if len(values) != 4 or sorted(row["view_index"] for row in values) != [0, 1, 2, 3]:
            raise ValueError(f"Endpoint needs exactly four distinct {name} views")
    return {"qa_pass": all(row["correct"] and float(row["margin"]) > 0 for row in normal),
            "normal_correct": sum(bool(row["correct"]) and float(row["margin"]) > 0 for row in normal),
            "reset_correct": sum(bool(row["correct"]) and float(row["margin"]) > 0 for row in reset),
            "reader_margin": min(float(row["margin"]) for row in normal),
            "gate": "all_4_reverse_cyclic_views_strictly_correct", "legacy_reachability_gate": None}


class PathAccumulator:
    def __init__(self) -> None:
        self.initial: torch.Tensor | None = None
        self.previous: torch.Tensor | None = None
        self.length = 0.0

    def add(self, value: torch.Tensor) -> dict[str, float | None]:
        current = value.detach().cpu().double()
        if not torch.isfinite(current).all():
            raise RuntimeError("Path geometry includes a nonfinite tensor")
        if self.initial is None:
            self.initial = current.clone()
        distance = 0.0 if self.previous is None else float((current - self.previous).norm())
        self.length += distance
        displacement = float((current - self.initial).norm())
        self.previous = current.clone()
        return {"step_distance_l2": distance, "path_length_l2": self.length, "displacement_l2": displacement,
                "tortuosity": self.length / displacement if displacement > 0 else None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "optimize", "evaluate"), default="optimize")
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path)
    group.add_argument("--run-spec")
    parser.add_argument("--run-id")
    for name in ("train", "dev", "dreamlite", "reader", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--evaluation-spec", help="JSON or path with cases:[{case_id,space,latent_path,latent_sha256}]")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.manifest:
        if not args.run_id:
            raise ValueError("--manifest requires --run-id")
        args.spec = select_run_spec(json.loads(args.manifest.read_text(encoding="utf-8-sig")), args.run_id)
    else:
        args.spec = validate_run_spec(read_json_argument(args.run_spec))
        if args.run_id and args.run_id != args.spec["run_id"]:
            raise ValueError("--run-id disagrees with --run-spec")
    if args.mode == "evaluate" and not args.evaluation_spec:
        raise ValueError("evaluate requires --evaluation-spec; no implicit endpoint selection")
    args.config_payload = json.loads(args.config.read_text(encoding="utf-8-sig"))
    args.target_index = args.spec["target_index"]
    args.seed = 0  # Runtime determinism is fixed; start RNG is separately keyed by spec.seed.
    args.strict_determinism = True
    args.pairing_seed = 0
    args.split_seed = 20260730
    from vision_memory.training.r10_alignment import R10_SELECTION_SEED
    args.schedule_seed = R10_SELECTION_SEED
    args.selected_step_count = 0
    args.gradient_mode = "full"
    return args


def _load_runtime(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.train import r11_new_frozen_dreamlite_oracle as legacy
    from vision_memory.dreamlite.conditioning import encode_latent_path_condition
    from vision_memory.training.r11_new_oracle import R11_NEW_TARGET_IDS, R11_NEW_TARGETS_PAYLOAD_SHA256
    if not torch.cuda.is_available():
        raise RuntimeError("Real Stage A execution requires CUDA; CPU mocks are not experiment evidence")
    dream_device, reader_device = torch.device(args.dreamlite_device), torch.device(args.reader_device)
    if dream_device == reader_device:
        raise ValueError("DreamLite and Reader require distinct GPUs")
    determinism = legacy.r8.configure_strict_cuda_determinism(0)
    # Use the original immutable data selection binding, not whichever local files happen to be present.
    fixed = json.loads(legacy.CONFIG_PATH.read_text(encoding="utf-8"))["fixed_data"]
    observed = {"train_sha256": file_sha256(args.train), "dev_sha256": file_sha256(args.dev)}
    if observed != fixed:
        raise RuntimeError(f"Locked data binding drifted: {observed}")
    snapshots = legacy._snapshot_bindings(args)
    data = legacy.r5._load_data(args, optimizer_steps=0)
    selected = legacy.select_f1_targets(data.train_pools)
    if tuple(target.segment_id for target in selected) != R11_NEW_TARGET_IDS or legacy.r5.canonical_sha256(
        [target.to_dict() for target in selected]
    ) != R11_NEW_TARGETS_PAYLOAD_SHA256:
        raise RuntimeError("The eight preregistered F1 targets have drifted")
    target = selected[args.target_index]
    pipe = legacy._load_pipeline(args, dream_device, torch.float32)
    processor, reader = legacy._load_reader(args, reader_device, torch.bfloat16)
    source_rgb = legacy.blank_source_rgb(device=dream_device, dtype=torch.float32)
    with torch.no_grad():
        source = legacy.encode_model_latent(pipe.vae, source_rgb)
        condition = encode_latent_path_condition(pipe, source, target.events[0].event_text)
        repeated = encode_latent_path_condition(pipe, source, target.events[0].event_text)
    if not torch.equal(condition.prompt_embeds, repeated.prompt_embeds) or not torch.equal(
        condition.attention_mask, repeated.attention_mask
    ):
        raise RuntimeError("Frozen source/event conditioner is not bitwise repeatable")
    oracle = legacy.FrozenDreamLiteOracle(
        unet=pipe.unet, scheduler=pipe.scheduler, vae=pipe.vae, text_encoder=pipe.text_encoder,
        source_latents=source, prompt_embeds=condition.prompt_embeds, prompt_attention_mask=condition.attention_mask,
        initial_x_t=initial_tensor(source.shape, args.spec).to(dream_device), compute_dtype=torch.float32,
        checkpoint_unet=True, vae_scale_factor=int(pipe.vae_scale_factor),
    )
    oracle.eval()
    reader.eval()
    for name, module in (("unet", oracle.unet), ("vae", oracle.vae), ("text_encoder", oracle.text_encoder)):
        wrong_dtype = [str(parameter.dtype) for parameter in module.parameters()
                       if parameter.is_floating_point() and parameter.dtype != torch.float32]
        if wrong_dtype:
            raise RuntimeError(f"Full FP32 DreamLite contract violated in {name}: {set(wrong_dtype)}")
    train_reader = legacy.r8.choice_reader_callable(reader=reader, processor=processor, reader_device=reader_device,
                                                   require_grad=True, deterministic_ce=True)
    eval_reader = legacy.r8.choice_reader_callable(reader=reader, processor=processor, reader_device=reader_device,
                                                  require_grad=False, deterministic_ce=True)
    with torch.no_grad():
        reset_image = legacy.decode_model_latents_unit_interval(pipe.vae, source, clamp=True)
    runtime = dict(legacy=legacy, oracle=oracle, reader=reader, pipe=pipe, target=target, source=source,
                   condition=condition, train_reader=train_reader, eval_reader=eval_reader, reset_image=reset_image,
                   snapshots=snapshots, determinism=determinism, data_binding=observed,
                   devices={"dreamlite": {"device": str(dream_device), "name": torch.cuda.get_device_name(dream_device)},
                            "reader": {"device": str(reader_device), "name": torch.cuda.get_device_name(reader_device)}})
    return runtime


def _source_bindings() -> dict[str, str]:
    # Bind the complete Python implementation tree, including dynamically imported Reader/sampler helpers.
    paths = [*sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "scripts").rglob("*.py"))]
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}


def _manifest(args: argparse.Namespace, runtime: Mapping[str, Any]) -> dict[str, Any]:
    out = args.output_dir
    target, condition = runtime["target"], runtime["condition"]
    condition_records = {
        "source_latent": save_array(out / "condition/source_latent.npy", runtime["source"]),
        "event_embedding": save_array(out / "condition/event_embedding.npy", condition.prompt_embeds),
        "event_attention_mask": save_array(out / "condition/event_attention_mask.npy", condition.attention_mask),
    }
    weights = condition.attention_mask.to(condition.prompt_embeds.dtype).unsqueeze(-1)
    pooled = (condition.prompt_embeds * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    condition_records["event_pooled_embedding"] = save_array(out / "condition/event_pooled_embedding.npy", pooled)
    boundary = runtime["legacy"]._writer_information_boundary(target)
    if not boundary["passed"]:
        raise RuntimeError("Source/event-only information boundary failed")
    # The inherited boundary audit checks source/event-only conditioning; this new
    # protocol deliberately replaces the legacy per-event noise key with PCG64 starts.
    actual_initial_keys = ["initial_xT_path", "initial_xT_file_sha256"] if "initial_xT_path" in args.spec else [
        "seed", "distribution", "scale"
    ]
    boundary["noise_key"] = actual_initial_keys
    boundary["oracle_initialization_key_names"] = actual_initial_keys
    boundary["initialization_protocol"] = "SHA-bound A5 restart" if "initial_xT_path" in args.spec else "CPU NumPy PCG64"
    result = {
        "schema": SCHEMA, "mode": args.mode, "run_spec": args.spec,
        "run_spec_sha256": canonical_object_sha256(args.spec), "config_path": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config), "config": args.config_payload,
        "input_manifest_sha256": file_sha256(args.manifest) if args.manifest else None,
        "git_commit": runtime["legacy"].r8.git_value("rev-parse", "HEAD"),
        "git_dirty": bool(runtime["legacy"].r8.git_value("status", "--porcelain")),
        "source_files_sha256": _source_bindings(), "models": runtime["snapshots"],
        "data": runtime["data_binding"], "determinism": runtime["determinism"], "devices": runtime["devices"],
        "target_segment": target.to_dict(), "task_id": target.segment_id,
        "condition": condition_records, "information_boundary": boundary,
        "initial_xT": tensor_record(runtime["oracle"].x_T_fp32),
        "contract": {"optimizer": "Adam", "optimizer_steps": STEPS, "lr": LEARNING_RATE,
                     "weight_decay": 0.0, "gradient_clipping": None, "only_trainable": "x_T_fp32",
                     "dreamlite_dtype": "float32", "reader_dtype": "bfloat16", "resolution": 1024,
                     "dreamlite_steps": 4, "effective_sigmas": [0.5, 0.375, 0.25, 0.125],
                     "checkpoint_steps": list(CHECKPOINT_STEPS), "latent_save_every_step": True,
                     "path_length_sampling": "every optimizer iterate", "repeat_in_initial_seed": False,
                     "reset": "VAE.decode(unchanged_blank_source_latent)",
                     "training_views": "legacy locked forward cyclic schedule",
                     "endpoint_views": "four reverse cyclic choice permutations",
                     "success": "all four normal views strictly correct; no tie", "legacy_gate": "not reused"},
    }
    write_json(out / "manifest.json", result)
    return result


def _reader_forward(runtime: Mapping[str, Any], image: torch.Tensor, step: int) -> tuple[Any, dict[str, Any]]:
    legacy, target = runtime["legacy"], runtime["target"]
    schedule = legacy.build_phase1a_schedule(legacy.R11_NEW_TARGET_IDS.index(target.segment_id))[step]
    permutation = schedule.permutation
    ordered = tuple(target.query.choices[index] for index in permutation)
    correct = permutation.index(target.query.target_index)
    output = runtime["train_reader"](image, legacy.r5.format_mcq_query(target.query.text, ordered), ordered, correct)
    return output, {**score_record(output, correct), "view_index": schedule.forward_cyclic_training_view,
                    "permutation": list(permutation)}


def _evaluate_image(runtime: Mapping[str, Any], image: torch.Tensor) -> list[dict[str, Any]]:
    legacy, target = runtime["legacy"], runtime["target"]
    rows = []
    with torch.no_grad():
        for name, state in (("normal", image), ("reset", runtime["reset_image"])):
            for view, permutation in enumerate(legacy.r5.REVERSE_CYCLIC4):
                ordered = tuple(target.query.choices[index] for index in permutation)
                correct = permutation.index(target.query.target_index)
                result = runtime["eval_reader"](state, legacy.r5.format_mcq_query(target.query.text, ordered), ordered, correct)
                rows.append({"condition": name, "view_index": view, "permutation": list(permutation),
                             **score_record(result, correct)})
    return rows


def _save_checkpoint(args: argparse.Namespace, runtime: Mapping[str, Any], optimizer: torch.optim.Optimizer,
                     step: int, output: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    path = args.output_dir / "checkpoints" / f"step-{step:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Raw RGB and optimizer are retained in addition to a human-viewable PNG.
    payload = {"step": step, "x_T_fp32": runtime["oracle"].x_T_fp32.detach().cpu(),
               "z": output.z_t.detach().cpu(), "rgb": output.image.detach().cpu(),
               "optimizer": optimizer.state_dict(), "state": dict(state)}
    torch.save(payload, path.with_suffix(".pt.tmp"))
    path.with_suffix(".pt.tmp").replace(path)
    png = path.with_suffix(".png")
    runtime["legacy"]._save_image(png, output.image)
    return {"step": step, "path": str(path.resolve()), "file_sha256": file_sha256(path),
            "png_path": str(png.resolve()), "png_sha256": file_sha256(png),
            "rgb": tensor_record(output.image), "optimizer_sha256": canonical_object_sha256(optimizer.state_dict())}


def _probe(args: argparse.Namespace, runtime: Mapping[str, Any]) -> dict[str, Any]:
    oracle = runtime["oracle"]
    rows = []
    for repeat in range(2):
        oracle.zero_grad(set_to_none=True)
        output = oracle()
        reader_output, scores = _reader_forward(runtime, output.image, 0)
        reader_output.loss.backward()
        audit = assert_frozen_gradient_contract(oracle, runtime["reader"])
        row = {"repeat": repeat, "scores": scores, "gradient_audit": audit,
               "xT": tensor_record(oracle.x_T_fp32), "z": tensor_record(output.z_t),
               "rgb": tensor_record(output.image), "effective_sigmas": list(output.effective_sigmas),
               "denoiser_steps": len(output.trajectory) - 1}
        rows.append(row)
        del reader_output, output
    def comparable(row):
        return {key: value for key, value in row.items() if key != "repeat"}
    passed = comparable(rows[0]) == comparable(rows[1])
    result = {**args.spec, "schema": SCHEMA, "status": "passed" if passed else "failed", "mode": "probe",
              "technical_pass": passed, "passed": passed, "bitwise_repeatability": passed,
              "backward_calls": 2, "optimizer_steps": 0, "rows": rows, "scientific_success_claim": False}
    write_json(args.output_dir / "probe.json", result)
    if not passed:
        raise RuntimeError("Identical complete forward/backward probe did not reproduce bitwise")
    return result


def _optimize(args: argparse.Namespace, runtime: Mapping[str, Any]) -> dict[str, Any]:
    oracle = runtime["oracle"]
    optimizer = torch.optim.Adam([oracle.x_T_fp32], lr=LEARNING_RATE, weight_decay=0.0,
                                 betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False)
    paths = {"xT": PathAccumulator(), "z": PathAccumulator()}
    index, checkpoints, metrics = [], [], []
    started = time.monotonic()
    initial_evaluations = []
    for step in range(STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad() if step == STEPS else contextlib.nullcontext():
            output = oracle()
        record = {"step": step, "xT": save_array(args.output_dir / "xT" / f"step-{step:03d}.npy", oracle.x_T_fp32),
                  "z": save_array(args.output_dir / "z" / f"step-{step:03d}.npy", output.z_t),
                  "geometry": {"xT": paths["xT"].add(oracle.x_T_fp32), "z": paths["z"].add(output.z_t)}}
        index.append(record)
        if step in CHECKPOINT_STEPS:
            checkpoints.append(_save_checkpoint(args, runtime, optimizer, step, output, record))
            write_json(args.output_dir / "checkpoint_index.json", checkpoints)
        if step == 0:
            initial_evaluations = _evaluate_image(runtime, output.image.detach())
            write_json(args.output_dir / "initial_evaluation.json", initial_evaluations)
        if step == STEPS:
            evaluations = _evaluate_image(runtime, output.image)
            break
        reader_output, scores = _reader_forward(runtime, output.image, step)
        reader_output.loss.backward()
        audit = assert_frozen_gradient_contract(oracle, runtime["reader"])
        before = oracle.x_T_fp32.detach().clone()
        optimizer.step()
        if not torch.isfinite(oracle.x_T_fp32).all():
            raise RuntimeError("Adam generated nonfinite x_T")
        row = {"schema": SCHEMA, "step": step, "optimizer_step": step + 1, **scores,
               "gradient": audit["gradient"], "gradient_nonzero_fraction": audit["gradient_nonzero_fraction"],
               "frozen_gradient_contract_passed": True, "xT_before_sha256": record["xT"]["sha256"],
               "xT_after_sha256": canonical_tensor_sha256(oracle.x_T_fp32),
               "xT_update_l2": float((oracle.x_T_fp32.detach() - before).double().norm()),
               "z_sha256": record["z"]["sha256"], "geometry": record["geometry"],
               "effective_sigmas": list(output.effective_sigmas), "denoiser_steps": len(output.trajectory) - 1,
               "elapsed_seconds": time.monotonic() - started}
        metrics.append(row)
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        write_json(args.output_dir / "trajectory_index.json", index)
        if step % 8 == 0:
            print(json.dumps({"run_id": args.spec["run_id"], "optimizer_step": step + 1,
                              "ce": scores["ce"], "gradient_norm": audit["gradient"]["norm"]}), flush=True)
        del reader_output, output, before
    write_json(args.output_dir / "trajectory_index.json", index)
    write_json(args.output_dir / "evaluation.json", evaluations)
    target = runtime["target"]
    legacy = runtime["legacy"]
    old_gate_rows = [{**row, "checkpoint": checkpoint, "suite": "frozen_oracle_geometry",
                      "pair_unit": target.segment_id}
                     for checkpoint, rows in (("m0", initial_evaluations), (legacy.R11_NEW_PRIMARY_ENDPOINT, evaluations))
                     for row in rows]
    old_statistics = legacy.phase1a_target_statistics(old_gate_rows, suite="frozen_oracle_geometry",
                                                      target_segment_id=target.segment_id)
    old_gate = legacy.phase1a_target_gate(old_statistics, technical_gate=True)
    return {**args.spec, "schema": SCHEMA, "status": "completed", "mode": "optimize",
            "task_id": target.segment_id, "technical_pass": True,
            "functional_chain": "frozen_dreamlite_vae_reader",
            "initial_xT_sha256": index[0]["xT"]["sha256"],
            "optimized_xT_sha256": index[-1]["xT"]["sha256"],
            "endpoint_z_sha256": index[-1]["z"]["sha256"],
            "loss_trajectory_sha256": canonical_object_sha256([row["ce"] for row in metrics]),
            "gradient_trajectory_sha256": canonical_object_sha256([row["gradient"]["sha256"] for row in metrics]),
            "source_episode_id": target.events[0].source_episode_id,
            "source_event_id": str(target.events[0].noise_turn_id), "optimizer_steps": STEPS,
            "initial_xT_path": index[0]["xT"]["path"], "initial_endpoint_z_path": index[0]["z"]["path"],
            "optimized_xT_path": index[-1]["xT"]["path"], "endpoint_z_path": index[-1]["z"]["path"],
            "source_latent_path": str((args.output_dir / "condition/source_latent.npy").resolve()),
            "event_embedding_path": str((args.output_dir / "condition/event_pooled_embedding.npy").resolve()),
            "event_token_embedding_path": str((args.output_dir / "condition/event_embedding.npy").resolve()),
            "event_pooled_embedding_path": str((args.output_dir / "condition/event_pooled_embedding.npy").resolve()),
            "event_attention_mask_path": str((args.output_dir / "condition/event_attention_mask.npy").resolve()),
            "optimized_xT_file_sha256": index[-1]["xT"]["file_sha256"],
            "endpoint_z_file_sha256": index[-1]["z"]["file_sha256"],
            "optimized_xT_tensor_sha256": index[-1]["xT"]["sha256"],
            "endpoint_z_tensor_sha256": index[-1]["z"]["sha256"],
            "prior_penalty": float(oracle.x_T_fp32.detach().double().square().mean()),
            "prior_penalty_definition": "mean(xT**2), squared RMS proxy, not Gaussian NLL",
            "perturbation": None, "evaluation_rows": evaluations, **strict_endpoint_gate(evaluations),
            "initial_evaluation_rows": initial_evaluations,
            "legacy_reachability_gate": old_gate, "legacy_reachability_statistics": old_statistics,
            "legacy_gate_comparability": "same diagnostic thresholds, new FP32/distribution protocol; not historical rerun",
            "geometry": index[-1]["geometry"], "trajectory_index_path": str((args.output_dir / "trajectory_index.json").resolve()),
            "checkpoint_index_path": str((args.output_dir / "checkpoint_index.json").resolve()),
            "manifest_path": str((args.output_dir / "manifest.json").resolve()),
            "elapsed_seconds": time.monotonic() - started}


def _evaluate(args: argparse.Namespace, runtime: Mapping[str, Any]) -> dict[str, Any]:
    specification = read_json_argument(args.evaluation_spec)
    cases = specification.get("cases", [])
    if not cases:
        raise ValueError("evaluate requires at least one explicit case")
    results = []
    oracle = runtime["oracle"]
    with torch.no_grad():
        for case in cases:
            if case.get("target_index", args.target_index) != args.target_index:
                raise ValueError("All cases in a worker must bind the same target_index")
            if case.get("task_id", runtime["target"].segment_id) != runtime["target"].segment_id:
                raise ValueError("Evaluation task_id disagrees with locked target")
            path = Path(case["latent_path"])
            sha = file_sha256(path)
            if case.get("latent_sha256") != sha:
                raise ValueError(f"Evaluation latent SHA binding mismatch: {path}")
            value = torch.from_numpy(np.load(path, allow_pickle=False)).to(oracle.x_T_fp32)
            if value.shape != oracle.x_T_fp32.shape or not torch.isfinite(value).all():
                raise ValueError("Evaluation latent shape/finite contract failed")
            if case["space"] == "xT":
                oracle.x_T_fp32.copy_(value)
                forward = oracle()
                image, z = forward.image, forward.z_t
                full_path = True
            elif case["space"] == "z":
                z = value
                image = runtime["legacy"].decode_model_latents_unit_interval(oracle.vae, z, clamp=True)
                full_path = False
            else:
                raise ValueError("Evaluation space must be xT or z")
            rows = _evaluate_image(runtime, image)
            results.append({"case": dict(case), "latent_file_sha256": sha, "input_tensor": tensor_record(value),
                            "endpoint_z": tensor_record(z), "full_dreamlite_path_executed": full_path,
                            "path": "DreamLite->VAE->Reader" if full_path else "VAE->Reader (z-space diagnostic)",
                            "evaluation_rows": rows, **strict_endpoint_gate(rows)})
    return {"schema": SCHEMA, "status": "completed", "mode": "evaluate", "run_spec": args.spec,
            "evaluation_spec_sha256": canonical_object_sha256(specification), "cases": results,
            "optimizer_steps": 0, "manifest_path": str((args.output_dir / "manifest.json").resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Atomic ownership: never resume over a partial trajectory or silently mix attempts.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "status.json", {"status": "loading", "run_spec": args.spec, "pid": os.getpid()})
    try:
        runtime = _load_runtime(args)
        manifest = _manifest(args, runtime)
        write_json(args.output_dir / "status.json", {"status": "running", "run_spec": args.spec, "pid": os.getpid()})
        result = {"probe": _probe, "optimize": _optimize, "evaluate": _evaluate}[args.mode](args, runtime)
        # Reverify model files and all Python sources at completion; a changed environment invalidates the run.
        verified = {name: runtime["legacy"].verify_snapshot_binding(binding)
                    for name, binding in runtime["snapshots"].items()}
        if verified != runtime["snapshots"] or _source_bindings() != manifest["source_files_sha256"]:
            raise RuntimeError("Model or source files changed during the run")
        result["model_snapshot_end_verified"] = True
        result["technical_pass"] = True
        result["devices"] = runtime["devices"]
        result["manifest_sha256"] = file_sha256(args.output_dir / "manifest.json")
        write_json(args.output_dir / "summary.json", result)
        write_json(args.output_dir / "status.json", {"status": "completed", "run_spec": args.spec})
        print(json.dumps({"status": result["status"], "mode": args.mode, "run_id": args.spec["run_id"],
                          "technical_pass": True, "qa_pass": result.get("qa_pass"),
                          "summary_path": str((args.output_dir / "summary.json").resolve())}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"status": "failed", "run_spec": args.spec,
                   "exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        write_json(args.output_dir / "status.json", {"status": "failed", "run_spec": args.spec, "message": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
