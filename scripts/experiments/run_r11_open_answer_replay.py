"""Read-only, answer-blind replay of the eight original R11 VAE-latent endpoints.

The 32 original MCQ views are an environment anchor, not new training. The 48
open-answer generations receive neither choices nor a gold answer. A fresh output
directory, complete records, exact image reproduction and unchanged model snapshots
are required before this exploratory replay can be marked completed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.inspire.model_snapshot_manifest import verify_snapshot_binding  # noqa: E402
from scripts.train import dreamlite_r4_free_pixel as r4  # noqa: E402
from scripts.train.latent_r11_vae_oracle import encode_model_latent  # noqa: E402
from vision_memory.data import REVERSE_CYCLIC4  # noqa: E402
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval  # noqa: E402
from vision_memory.reader.open_answer import generate_short_answer, score_short_answer  # noqa: E402
from vision_memory.reader.qwen3vl import (  # noqa: E402
    R3_QWEN_READER_RESIZE_CONTRACT,
    qwen3vl_listwise_choice_ce,
)
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import format_mcq_query  # noqa: E402


SCHEMA = "vision_memory.r11-open-answer-replay.v1"
INITIAL_LATENT_SHA256 = "719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa"
CONDITIONS = ("matched", "blank", "donor")
PROMPTS = ("original_open", "paraphrase_open")
GENERATION = {"do_sample": False, "max_new_tokens": 32, "eos_policy": "original_model"}
SNAPSHOT_ENV = {
    "dreamlite_mobile": "VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256",
    "qwen_reader": "VLM_READER_SNAPSHOT_MANIFEST_SHA256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "vision_memory.r11-open-answer-replay-config.v1":
        raise ValueError("Unexpected replay config schema.")
    if config.get("conditions") != list(CONDITIONS) or config.get("prompts") != list(PROMPTS):
        raise ValueError("Replay requires all three conditions and both fixed prompts.")
    if config.get("generation") != GENERATION:
        raise ValueError("Generation must be greedy, 32 new tokens, with original EOS.")
    targets = config.get("targets", [])
    if len(targets) != 8 or [item.get("target_index") for item in targets] != list(range(8)):
        raise ValueError("Replay requires exactly the eight original ordered R11 targets.")
    if len({item["segment_id"] for item in targets}) != 8:
        raise ValueError("Duplicate target segment IDs.")
    for target in targets:
        index = target["target_index"]
        original = target["original"]
        choices = original["choices"]
        answer_index = original["answer_index"]
        if len(choices) != 4 or len(set(choices)) != 4 or not isinstance(answer_index, int):
            raise ValueError("The original MCQ needs four distinct choices and an integer answer index.")
        if isinstance(answer_index, bool) or not 0 <= answer_index < 4:
            raise ValueError("Invalid original MCQ answer index.")
        gold = target["scorer_metadata"]["gold"]
        if not isinstance(gold, str) or not gold.strip() or choices[answer_index] != gold:
            raise ValueError("Scorer gold must agree with the original target choice.")
        if target["scorer_metadata"].get("aliases") != []:
            raise ValueError("No post-hoc aliases are permitted in this strict replay.")
        if set(target["inputs"]) != set(PROMPTS):
            raise ValueError("Every target must have precisely the two open questions.")
        for query in target["inputs"].values():
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Open question must be a nonempty string.")
            if re.search(r"(?<!\w)" + re.escape(gold) + r"(?!\w)", query, re.IGNORECASE):
                raise ValueError(f"Gold answer leaked into open question for target {index}.")
            if "choose exactly one option" in query.casefold() or re.search(r"(?:^|\n)\s*[A-D][.)]", query):
                raise ValueError("Open questions must not include choice instructions or option lists.")
        donor = targets[(index + 1) % 8]
        if (target["donor_target_index"], target["donor_segment_id"]) != (
            donor["target_index"], donor["segment_id"]
        ):
            raise ValueError("Donor mapping must use the fixed next-target rotation.")
        if donor["scorer_metadata"]["gold"] == gold:
            raise ValueError("A donor must carry a different target answer.")
        artifact = target["artifact"]
        if artifact["endpoint_member"] != f"target-{index:02d}/run/endpoint_raw.pt":
            raise ValueError("Unexpected endpoint member path.")
        for key in ("endpoint_file_sha256", "endpoint_tensor_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", artifact[key]):
                raise ValueError(f"Invalid artifact hash: {key}.")
    return config


def open_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enumerate an answer-blind generation plan; gold stays in the scorer config."""
    return [
        {
            "target_index": target["target_index"],
            "segment_id": target["segment_id"],
            "condition": condition,
            "prompt_id": prompt_id,
            "query": target["inputs"][prompt_id],
            "donor_target_index": target["donor_target_index"] if condition == "donor" else None,
            "donor_segment_id": target["donor_segment_id"] if condition == "donor" else None,
        }
        for target in config["targets"]
        for condition in CONDITIONS
        for prompt_id in PROMPTS
    ]


def load_endpoint(old_run_root: Path, target: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = old_run_root.resolve(strict=True)
    path = (root / target["artifact"]["endpoint_member"]).resolve(strict=True)
    if root not in path.parents:
        raise ValueError("Endpoint path escapes the old run root.")
    file_hash = sha256_file(path)
    if file_hash != target["artifact"]["endpoint_file_sha256"]:
        raise ValueError(f"Endpoint file SHA256 mismatch for {path}.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != "vision_memory.r11-vae-latent-endpoint.v1":
        raise ValueError("Unexpected endpoint payload schema.")
    latent, image = payload.get("latent_fp32"), payload.get("image")
    if not isinstance(latent, Tensor) or latent.dtype != torch.float32 or tuple(latent.shape) != (1, 4, 128, 128):
        raise ValueError("Endpoint must contain an FP32 [1,4,128,128] latent.")
    if not isinstance(image, Tensor) or not image.is_floating_point() or tuple(image.shape) != (1, 3, 1024, 1024):
        raise ValueError("Endpoint must contain its saved floating [1,3,1024,1024] image.")
    if not torch.isfinite(latent).all() or not torch.isfinite(image).all():
        raise ValueError("Endpoint tensors must be finite.")
    tensor_hash = canonical_tensor_sha256(latent)
    if tensor_hash != target["artifact"]["endpoint_tensor_sha256"]:
        raise ValueError(f"Endpoint tensor SHA256 mismatch for {path}.")
    manifest_path = path.parent / "manifest.json"
    if sha256_file(manifest_path) != payload.get("manifest_sha256"):
        raise ValueError("Endpoint does not match its archived run manifest.")
    audit = {
        "target_index": target["target_index"], "path": str(path),
        "file_sha256": file_hash, "latent_sha256": tensor_hash,
        "saved_image_sha256": canonical_tensor_sha256(image),
        "saved_image_dtype": str(image.dtype), "old_manifest_sha256": payload["manifest_sha256"],
    }
    return payload, audit


def old_mcq_rows(old_run_root: Path, target_index: int) -> dict[int, dict[str, Any]]:
    path = old_run_root / f"target-{target_index:02d}" / "run" / "target_evaluation_rows.jsonl"
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row.get("checkpoint") == "raw_latent_step256" and row.get("condition") == "normal"]
    return {int(row["view_index"]): row for row in selected}


def aggregate_results(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], mcq_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {(case["target_index"], case["condition"], case["prompt_id"]) for case in open_cases(config)}
    keys = [(row["target_index"], row["condition"], row["prompt_id"]) for row in rows]
    if len(rows) != 48 or len(set(keys)) != 48 or set(keys) != expected:
        raise ValueError("Cannot aggregate incomplete, duplicated or unexpected open-answer records; need exactly 48.")
    expected_mcq = {(target, view) for target in range(8) for view in range(4)}
    mcq_keys = [(row["target_index"], row["view_index"]) for row in mcq_rows]
    if len(mcq_rows) != 32 or len(set(mcq_keys)) != 32 or set(mcq_keys) != expected_mcq:
        raise ValueError("Cannot aggregate incomplete, duplicated or unexpected MCQ anchor records; need exactly 32.")
    targets = {target["target_index"]: target for target in config["targets"]}
    for row in rows:
        target = targets[row["target_index"]]
        expected_score = score_short_answer(row["raw"], target["scorer_metadata"]["gold"])
        if row["scorer"] != expected_score or row["query"] != target["inputs"][row["prompt_id"]]:
            raise ValueError("Open-answer scorer or question does not match the locked inputs.")
        if not isinstance(row["truncated"], bool):
            raise ValueError("Generation truncation metadata must be explicit.")
    for row in mcq_rows:
        if row["permutation"] != list(REVERSE_CYCLIC4[row["view_index"]]):
            raise ValueError("MCQ anchor must use the original reverse-cyclic views.")
        if not math.isfinite(row["ce"]) or not isinstance(row["correct"], bool):
            raise ValueError("MCQ anchor metrics must be finite and complete.")

    def statistics(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        correct = sum(row["scorer"]["strict_correct"] for row in selected)
        return {
            "count": len(selected), "strict_correct": correct, "strict_accuracy": correct / len(selected),
            "extra_text_count": sum(row["scorer"]["has_extra_text"] for row in selected),
            "truncated_count": sum(row["truncated"] for row in selected),
            "format_status_counts": dict(Counter(row["scorer"]["format_status"] for row in selected)),
        }

    by_key = dict(zip(keys, rows))
    paired: dict[str, Any] = {}
    for prompt_id in PROMPTS:
        paired[prompt_id] = {}
        for control in ("blank", "donor"):
            classes: dict[str, list[int]] = {
                "both_correct": [], "matched_only_correct": [], "control_only_correct": [], "neither_correct": []
            }
            for index in range(8):
                matched = by_key[(index, "matched", prompt_id)]["scorer"]["strict_correct"]
                other = by_key[(index, control, prompt_id)]["scorer"]["strict_correct"]
                key = "both_correct" if matched and other else "matched_only_correct" if matched else (
                    "control_only_correct" if other else "neither_correct"
                )
                classes[key].append(index)
            paired[prompt_id][control] = {
                key: {"count": len(indices), "target_indices": indices} for key, indices in classes.items()
            }
    correct_mcq = sum(row["correct"] for row in mcq_rows)
    return {
        "schema": SCHEMA + ".summary", "complete_records": True, "formal_success": False,
        "scope": "exploratory endpoint replay only; no optimization and no new latent endpoints",
        "open_answer": statistics(rows),
        "by_condition": {condition: statistics([r for r in rows if r["condition"] == condition]) for condition in CONDITIONS},
        "by_prompt": {prompt: statistics([r for r in rows if r["prompt_id"] == prompt]) for prompt in PROMPTS},
        "by_condition_and_prompt": {
            condition: {
                prompt: statistics([r for r in rows if r["condition"] == condition and r["prompt_id"] == prompt])
                for prompt in PROMPTS
            } for condition in CONDITIONS
        },
        "paired_matched_vs_controls": paired,
        "mcq_anchor": {"count": 32, "correct": correct_mcq, "all_correct": correct_mcq == 32,
                       "mean_ce": sum(row["ce"] for row in mcq_rows) / 32},
        "semantic_review": {"status": "pending_manual_raw_text_review", "automatic_semantic_credit": False},
    }


def command_output(arguments: list[str]) -> str:
    completed = subprocess.run(arguments, cwd=ROOT, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else f"unavailable: {completed.stderr.strip()}"


def runtime_versions() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "peft", "accelerate", "safetensors", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": packages,
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"]),
    }


def compare_old_runtime(old_run_root: Path, current: Mapping[str, Any]) -> dict[str, Any]:
    observed = {"python": platform.python_version(), "cuda_runtime": current["torch_cuda"], **current["packages"]}
    targets = []
    for index in range(8):
        path = old_run_root / f"target-{index:02d}/run/runtime.json"
        expected = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {key: {"old": value, "current": observed.get(key)}
                      for key, value in expected.items() if observed.get(key) != value}
        targets.append({"target_index": index, "path": str(path), "sha256": sha256_file(path),
                        "old_runtime": expected, "mismatches": mismatches, "matches": not mismatches})
    return {"targets": targets, "all_legacy_fields_match": all(item["matches"] for item in targets),
            "driver_policy": "Current driver is separately recorded in runtime.nvidia_smi; hardware parity uses exact image and MCQ gates."}


def snapshot_bindings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: r4._verified_snapshot_payload(
            model_dir=args.dreamlite if key == "dreamlite_mobile" else args.reader,
            model_key=key, env_name=env_name, required=True,
        ) for key, env_name in SNAPSHOT_ENV.items()
    }


def load_vae(model_dir: Path, device: torch.device) -> torch.nn.Module:
    import diffusers

    index = json.loads((model_dir / "model_index.json").read_text(encoding="utf-8"))
    library, class_name = index["vae"]
    if library != "diffusers" or not isinstance(class_name, str) or not class_name.startswith("Autoencoder"):
        raise ValueError("Only a Diffusers Autoencoder VAE class may be loaded from the locked model index.")
    vae_class = getattr(diffusers, class_name)
    vae = vae_class.from_pretrained(model_dir, subfolder="vae", local_files_only=True, torch_dtype=torch.bfloat16)
    return vae.to(device).requires_grad_(False).eval()


def frozen_audit(vae: torch.nn.Module, reader: torch.nn.Module) -> dict[str, Any]:
    report = {}
    for name, model in (("vae", vae), ("reader", reader)):
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        gradients = sum(parameter.grad is not None for parameter in model.parameters())
        if trainable or gradients or model.training:
            raise RuntimeError(f"{name} must remain frozen, in eval mode and without parameter gradients.")
        report[name] = {"trainable_parameters": trainable, "parameter_gradient_tensors": gradients, "training": False}
    return report


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(args.config)
    vae_device, reader_device = torch.device(args.vae_device), torch.device(args.reader_device)
    if vae_device.type != "cuda" or reader_device.type != "cuda" or vae_device == reader_device:
        raise ValueError("Replay requires two distinct CUDA devices.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the original BF16 R11 replay.")
    determinism = configure_strict_cuda_determinism(0)
    source_paths = [Path(__file__), ROOT / "src/vision_memory/reader/open_answer.py",
                    ROOT / "src/vision_memory/reader/qwen3vl.py", ROOT / "models.lock.json"]
    manifest: dict[str, Any] = {
        "schema": SCHEMA + ".manifest", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": command_output(["git", "status", "--porcelain"]),
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths},
        "config_path": str(args.config.resolve()), "config_sha256": sha256_file(args.config), "config": config,
        "old_run_root": str(args.old_run_root.resolve()), "output_dir": str(args.output_dir.resolve()),
        "runtime": runtime_versions(), "determinism": determinism,
        "vae_device": str(vae_device), "reader_device": str(reader_device), "compute_dtype": "bfloat16",
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "expected_generation_count": 48, "expected_mcq_anchor_count": 32,
        "generation_overrides": {**GENERATION, "num_beams": 1, "num_return_sequences": 1, "use_cache": True},
        "blank_definition": "decode(posterior mean encode of uniform RGB 127/255), original R11 step-zero image",
        "initial_latent_sha256_expected": INITIAL_LATENT_SHA256,
        "model_snapshot_payloads_start": snapshot_bindings(args), "formal_success": False,
    }
    manifest["old_runtime_comparison"] = compare_old_runtime(args.old_run_root, manifest["runtime"])
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "config.json", config)
    endpoints, artifact_audits = {}, []
    for target in config["targets"]:
        endpoints[target["target_index"]], audit = load_endpoint(args.old_run_root, target)
        artifact_audits.append(audit)
        write_json(args.output_dir / "endpoint_integrity.json", {"endpoints": artifact_audits})
    vae = load_vae(args.dreamlite, vae_device)
    processor, reader = r4._load_reader(args, reader_device, torch.bfloat16)
    reader.eval()
    manifest["vae_class"] = type(vae).__name__
    manifest["reader_class"] = type(reader).__name__
    manifest["reader_generation_config"] = reader.generation_config.to_dict()
    manifest["frozen_parameters_start"] = frozen_audit(vae, reader)
    manifest["processor_class"] = type(processor).__name__
    manifest["image_processor_class"] = type(processor.image_processor).__name__
    write_json(args.output_dir / "manifest.json", manifest)

    initial_rgb = r4._initial_rgb_tensor(resolution=1024, device=vae_device, dtype=torch.bfloat16)
    initial_latent = encode_model_latent(vae, initial_rgb).float()
    initial_hash = canonical_tensor_sha256(initial_latent)
    write_json(args.output_dir / "initial_latent_verification.json", {
        "expected_sha256": INITIAL_LATENT_SHA256, "observed_sha256": initial_hash,
        "passed": initial_hash == INITIAL_LATENT_SHA256,
    })
    if initial_hash != INITIAL_LATENT_SHA256:
        raise RuntimeError("Initial latent does not reproduce the original R11 canonical SHA256.")
    images: dict[Any, Tensor] = {}
    images["blank"] = decode_model_latents_unit_interval(vae, initial_latent.to(torch.bfloat16), clamp=True).cpu()
    torch.save({"latent_fp32": initial_latent.cpu(), "image": images["blank"]}, args.output_dir / "blank_raw.pt")
    image_audits = []
    for target in config["targets"]:
        index = target["target_index"]
        payload = endpoints[index]
        decoded = decode_model_latents_unit_interval(
            vae, payload["latent_fp32"].to(device=vae_device, dtype=torch.bfloat16), clamp=True
        ).cpu()
        saved = payload["image"]
        delta = decoded.double() - saved.double()
        identical = decoded.dtype == saved.dtype and torch.equal(decoded, saved)
        image_audits.append({
            "target_index": index, "decoded_sha256": canonical_tensor_sha256(decoded),
            "saved_sha256": canonical_tensor_sha256(saved), "exactly_equal": identical,
            "rmse": float(delta.square().mean().sqrt()), "max_absolute_difference": float(delta.abs().max()),
        })
        write_json(args.output_dir / "image_reproduction.json", {"targets": image_audits})
        if not identical:
            torch.save({"decoded_image": decoded, "saved_image": saved}, args.output_dir / f"image_mismatch_{index:02d}.pt")
            raise RuntimeError(f"Target {index} VAE image does not exactly reproduce the saved endpoint.")
        images[index] = decoded
    image_hashes = {index: canonical_tensor_sha256(value) for index, value in images.items()}

    mcq_records = []
    for target in config["targets"]:
        index = target["target_index"]
        old_views = old_mcq_rows(args.old_run_root, index)
        for view_index, permutation in enumerate(REVERSE_CYCLIC4):
            begin = time.perf_counter()
            original = target["original"]
            choices = tuple(original["choices"][i] for i in permutation)
            target_index = permutation.index(original["answer_index"])
            query = format_mcq_query(original["query"], choices)
            output = qwen3vl_listwise_choice_ce(
                model=reader, processor=processor, image=images[index][0].to(reader_device), query=query,
                choices=choices, target_index=target_index, device=reader_device, require_image_grad=False,
                reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT, deterministic_ce=True,
            )
            logits = output.choice_logits.detach().float().cpu()
            if not torch.isfinite(logits).all() or not torch.isfinite(output.loss):
                raise RuntimeError("Nonfinite MCQ environment anchor scores.")
            predicted = int(logits.argmax())
            alternative_logits = torch.cat((logits[:target_index], logits[target_index + 1:]))
            previous = old_views.get(view_index)
            row = {
                "schema": SCHEMA + ".mcq-anchor", "target_index": index, "segment_id": target["segment_id"],
                "view_index": view_index, "permutation": list(permutation), "query": query, "choices": choices,
                "answer_index": target_index, "predicted_index": predicted, "correct": predicted == target_index,
                "ce": float(output.loss), "choice_mean_nll": output.choice_mean_nll.detach().float().cpu().tolist(),
                "margin": float(logits[target_index] - alternative_logits.max()),
                "choice_logits": logits.tolist(), "old_record": previous,
                "ce_difference_from_old": None if previous is None else float(output.loss) - float(previous["ce"]),
                "image_sha256": image_hashes[index],
                "latent_sha256": target["artifact"]["endpoint_tensor_sha256"],
                "elapsed_seconds": time.perf_counter() - begin,
            }
            append_jsonl(args.output_dir / "mcq_anchor.jsonl", row)
            mcq_records.append(row)
            print(json.dumps({"stage": "mcq_anchor", "completed": len(mcq_records), "correct": row["correct"]}), flush=True)
    if len(mcq_records) != 32 or not all(row["correct"] for row in mcq_records):
        raise RuntimeError("Original R11 MCQ environment anchor failed: all 32 views must remain correct.")

    records = []
    for case in open_cases(config):
        begin = time.perf_counter()
        target = config["targets"][case["target_index"]]
        image_index = "blank" if case["condition"] == "blank" else (
            case["donor_target_index"] if case["condition"] == "donor" else case["target_index"]
        )
        # This call intentionally has no gold, choices, expected answer or answer-prefix argument.
        generated = generate_short_answer(
            model=reader, processor=processor, image=images[image_index].to(reader_device), query=case["query"],
            device=reader_device, max_new_tokens=32, do_sample=False,
            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
        )
        record = {
            "schema": SCHEMA + ".generation", **case, **generated,
            "scorer": score_short_answer(generated["raw"], target["scorer_metadata"]["gold"]),
            "scorer_metadata": target["scorer_metadata"], "semantic_review": "pending_manual_raw_text_review",
            "image_sha256": image_hashes[image_index],
            "latent_sha256": initial_hash if image_index == "blank" else canonical_tensor_sha256(endpoints[image_index]["latent_fp32"]),
            "elapsed_seconds": time.perf_counter() - begin,
        }
        append_jsonl(args.output_dir / "generations.jsonl", record)
        records.append(record)
        print(json.dumps({"stage": "open_generation", "completed": len(records), "target_index": case["target_index"],
                          "condition": case["condition"], "prompt_id": case["prompt_id"], "raw": generated["raw"]}), flush=True)

    frozen_end = frozen_audit(vae, reader)
    end_bindings = {key: verify_snapshot_binding(binding) for key, binding in manifest["model_snapshot_payloads_start"].items()}
    snapshots_unchanged = end_bindings == manifest["model_snapshot_payloads_start"]
    write_json(args.output_dir / "model_snapshot_verification_end.json", {
        "bindings": end_bindings, "passed": snapshots_unchanged, "frozen_parameters": frozen_end,
    })
    if not snapshots_unchanged:
        raise RuntimeError("Model snapshots changed during the replay.")
    summary = aggregate_results(config, records, mcq_records)
    summary.update({
        "technical_passed": True, "all_images_exactly_reproduced": True, "snapshots_unchanged": True,
        "initial_latent_sha256": initial_hash, "elapsed_seconds": time.perf_counter() - started,
    })
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--old-run-root", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    # Never modify an existing output directory, including terminal/error files.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        summary = run(args)
    except BaseException as error:
        write_json(args.output_dir / "terminal.json", {
            "schema": SCHEMA + ".terminal", "status": "failed", "technical_passed": False, "formal_success": False,
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        })
        traceback.print_exc()
        return 1
    write_json(args.output_dir / "terminal.json", {
        "schema": SCHEMA + ".terminal", "status": "completed", "technical_passed": True,
        "formal_success": False, "generation_count": summary["open_answer"]["count"],
        "mcq_anchor_count": summary["mcq_anchor"]["count"], "elapsed_seconds": time.perf_counter() - started,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
