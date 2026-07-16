from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import read_jsonl  # noqa: E402
from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.reader import qwen3vl_query_free_visual_features  # noqa: E402
from vision_memory.repro import (  # noqa: E402
    configure_strict_cuda_determinism,
    cuda_peak_memory_report,
    load_initial_image,
    probe_provenance,
    reset_cuda_peak_memory,
)
from vision_memory.teacher import (  # noqa: E402
    FrozenTeacherLossCalibration,
    composite_teacher_distillation_loss,
    load_teacher_cache_manifest,
    load_teacher_transition_sidecar,
    make_disk_teacher_provider,
    save_teacher_calibration,
)
from vision_memory.training import DreamLiteEpisodeModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze R3 teacher-loss median scales before training")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    # This is the updater/global diffusion seed, not the dataset-generation seed.
    # R3 micro training is pre-registered at global seed 0, so calibration must
    # observe the exact same initial student/noise protocol.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-seed", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("Teacher-loss calibration requires two visible CUDA GPUs.")
    if args.output.exists() or args.report.exists():
        raise SystemExit("Calibration refuses to overwrite an existing output or report.")
    determinism = configure_strict_cuda_determinism(args.seed)
    provenance = probe_provenance(
        root=ROOT,
        arguments=args,
        models={"dreamlite": args.dreamlite, "reader": args.reader},
    )
    git = provenance.get("git", {})
    if git.get("clean") is not True or not git.get("commit"):
        raise SystemExit("Teacher-loss calibration requires a clean, committed checkout.")
    manifest_path = args.cache_dir / "manifest.json"
    sidecar_path = args.cache_dir / "transitions.jsonl"
    manifest = load_teacher_cache_manifest(manifest_path)
    transitions = load_teacher_transition_sidecar(sidecar_path, manifest=manifest)
    provider = make_disk_teacher_provider(manifest_path)
    for record in manifest.records:
        provider.get(record.state_id, split="train")
    transition_lookup = {(record.episode_id, record.turn_id): record for record in transitions}
    if len(transition_lookup) != len(transitions):
        raise RuntimeError("Teacher transition keys are not unique.")

    episodes = read_jsonl(args.train)
    expected_event_keys = {
        (episode.episode_id, index)
        for episode in episodes
        for index, turn in enumerate(episode.turns)
        if turn.type.value in {"event", "mixed"}
    }
    if expected_event_keys != set(transition_lookup):
        missing = sorted(expected_event_keys - set(transition_lookup))
        extra = sorted(set(transition_lookup) - expected_event_keys)
        raise ValueError(f"Train episodes and teacher sidecar differ: missing={missing[:4]}, extra={extra[:4]}.")

    dreamlite_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    if dreamlite_device == reader_device:
        raise SystemExit("DreamLite and Reader calibration devices must differ.")
    dtype = torch.bfloat16
    from diffusers import DreamLiteMobilePipeline
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    set_all_seeds(args.seed)
    pipe = DreamLiteMobilePipeline.from_pretrained(
        args.dreamlite,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(dreamlite_device)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    pipe.unet.requires_grad_(False)
    set_all_seeds(args.adapter_seed)
    pipe.unet = get_peft_model(
        pipe.unet,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            lora_dropout=0.0,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
    )
    pipe.unet.eval()
    source_pil, source_metadata = load_initial_image("blank", None, resolution=args.resolution)
    source_image = pipe.image_processor.preprocess(source_pil, height=args.resolution, width=args.resolution)
    with torch.no_grad():
        initial_state = pipe.prepare_image_latents(source_image, dtype=dtype, device=dreamlite_device)
    model = DreamLiteEpisodeModel(
        pipeline=pipe,
        initial_state=initial_state,
        global_seed=args.seed,
        checkpoint_unet=False,
    )

    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=256 * 256,
        max_pixels=256 * 256,
    )
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("Calibration requires the tensor-native fast Qwen image processor.")
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(reader_device)
    freeze_module(reader)
    reader.eval()
    reader.config.use_cache = False
    reset_cuda_peak_memory((dreamlite_device, reader_device))

    unit = FrozenTeacherLossCalibration(latent_scale=1.0, image_scale=1.0, feature_scale=1.0)
    raw_latent: list[float] = []
    raw_image: list[float] = []
    raw_feature: list[float] = []
    with torch.no_grad():
        for episode in episodes:
            state = model.reset_state()
            for turn_index, turn in enumerate(episode.turns):
                if turn.type.value not in {"event", "mixed"}:
                    continue
                assert turn.event_text is not None
                state = model.updater(state, turn.event_text, episode.episode_id, turn_index)
                transition = transition_lookup[(episode.episode_id, turn_index)]
                teacher = provider.get(transition.after_state_id, split="train")
                image = model.updater.decode_for_reader(state).to(reader_device)
                feature = qwen3vl_query_free_visual_features(
                    model=reader,
                    processor=processor,
                    image=image,
                    device=reader_device,
                    require_image_grad=False,
                ).features
                output = composite_teacher_distillation_loss(
                    student_latent=state.to(reader_device),
                    student_image=image,
                    student_feature=feature,
                    teacher=teacher,
                    calibration=unit,
                )
                raw_latent.append(float(output.latent_raw))
                raw_image.append(float(output.image_raw))
                raw_feature.append(float(output.feature_raw))

    scales = tuple(statistics.median(values) for values in (raw_latent, raw_image, raw_feature))
    calibration = FrozenTeacherLossCalibration(
        latent_scale=scales[0],
        image_scale=scales[1],
        feature_scale=scales[2],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibration_file_sha256 = save_teacher_calibration(args.output, calibration)
    report = {
        "schema": "vision_memory.r3-teacher-calibration-report.v1",
        "train_sha256": sha256_file(args.train),
        "manifest_sha256": sha256_file(manifest_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "calibration_file_sha256": calibration_file_sha256,
        "calibration_contract_sha256": calibration.contract_sha256,
        "transition_count": len(raw_latent),
        "seed": args.seed,
        "adapter_seed": args.adapter_seed,
        "lora_rank": args.lora_rank,
        "initial_state": source_metadata,
        "sample_selection": {
            "split": "train",
            "unit": "one-unweighted-sample-per-updater-transition",
            "query_turns_excluded": True,
            "duplicate-semantic-after-states_retained": True,
        },
        "scales": calibration.to_dict(),
        "raw_component_ranges": {
            "latent": [min(raw_latent), max(raw_latent)],
            "image": [min(raw_image), max(raw_image)],
            "feature": [min(raw_feature), max(raw_feature)],
        },
        "strict_determinism": determinism,
        "cuda_peak_memory": cuda_peak_memory_report((dreamlite_device, reader_device)),
        "provenance": provenance,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
