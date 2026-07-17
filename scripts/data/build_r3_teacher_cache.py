from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import __version__ as PILLOW_VERSION


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.dreamlite.recurrent import DreamLiteRecurrentUpdater  # noqa: E402
from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_RESIZE_CONTRACT,
    qwen3vl_query_free_visual_features,
)
from vision_memory.teacher import (  # noqa: E402
    FixedFontContract,
    FullStateCardRenderer,
    SemanticState,
    TeacherArtifactRecord,
    TeacherBuildContract,
    TeacherCacheManifest,
    TeacherTransitionRecord,
    build_teacher_state,
    file_sha256,
    save_teacher_manifest,
    save_teacher_sidecar,
    save_teacher_tensor,
)


LOCKED_FONT_SHA256 = "3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22"
RAW_SIDECAR_SCHEMA = "vlm.r3.teacher_transition.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the query-free, train-only R3 full-state teacher cache")
    parser.add_argument("--raw-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=ROOT / "assets" / "fonts" / "DejaVuSans.ttf")
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    return parser.parse_args()


def locked_revision(path: Path) -> str:
    marker = path / ".locked_revision"
    if not marker.is_file():
        raise ValueError(f"Model snapshot lacks .locked_revision: {path}")
    value = marker.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Model snapshot has an empty .locked_revision: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw_sidecar(path: Path) -> tuple[dict[str, Any], ...]:
    expected = {
        "schema_version",
        "split",
        "episode_id",
        "turn_id",
        "event_kind",
        "before_state",
        "after_state",
    }
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"Raw teacher sidecar contains a blank line at {line_number}.")
        value = json.loads(line)
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"Raw teacher sidecar line {line_number} differs from its locked schema.")
        if value["schema_version"] != RAW_SIDECAR_SCHEMA or value["split"] != "train":
            raise ValueError("R3 teacher sidecars are schema-locked and train-only.")
        if isinstance(value["turn_id"], bool) or not isinstance(value["turn_id"], int):
            raise ValueError("Teacher sidecar turn_id must be an integer.")
        records.append(dict(value))
    if not records:
        raise ValueError("Raw teacher sidecar is empty.")
    return tuple(records)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("Teacher cache construction requires two visible CUDA GPUs.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("Teacher cache construction refuses a non-empty output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if file_sha256(args.font) != LOCKED_FONT_SHA256:
        raise SystemExit("Embedded DejaVuSans.ttf SHA256 differs from the R3 lock.")

    raw_records = read_raw_sidecar(args.raw_sidecar)
    semantic_states: dict[str, SemanticState] = {}
    parsed_pairs: list[tuple[dict[str, Any], SemanticState, SemanticState]] = []
    for raw in raw_records:
        before = SemanticState.from_dict(raw["before_state"])
        after = SemanticState.from_dict(raw["after_state"])
        for state in (before, after):
            existing = semantic_states.get(state.state_id)
            if existing is not None and existing.canonical_bytes != state.canonical_bytes:
                raise RuntimeError("Semantic state_id collision detected.")
            semantic_states[state.state_id] = state
        parsed_pairs.append((raw, before, after))

    dreamlite_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    if dreamlite_device == reader_device:
        raise SystemExit("DreamLite VAE and Qwen Reader must use distinct devices.")
    dtype = torch.bfloat16
    from diffusers import DreamLiteMobilePipeline
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    pipe = DreamLiteMobilePipeline.from_pretrained(
        args.dreamlite,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(dreamlite_device)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    freeze_module(pipe.unet)
    pipe.vae.eval()
    updater = DreamLiteRecurrentUpdater(pipeline=pipe, global_seed=0, checkpoint_unet=False)

    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=256 * 256,
        max_pixels=256 * 256,
    )
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("Teacher construction requires the tensor-native fast Qwen image processor.")
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(reader_device)
    freeze_module(reader)
    reader.eval()

    font = FixedFontContract(
        font_id="DejaVuSans-2.37-embedded",
        path=args.font,
        sha256=LOCKED_FONT_SHA256,
        pillow_version=PILLOW_VERSION,
    )
    renderer = FullStateCardRenderer(font)
    contract = TeacherBuildContract(
        latent_callback_id="dreamlite-vae-posterior-mean.scaled.v1",
        decode_callback_id="dreamlite-vae-decode-unit-clamped.v1",
        feature_callback_id="qwen3vl-post-merger-query-free.v1",
        vae_revision=locked_revision(args.dreamlite),
        reader_revision=locked_revision(args.reader),
    )

    def encode_image(image: torch.Tensor) -> torch.Tensor:
        return updater.reencode_posterior_mean(image.to(device=dreamlite_device, dtype=dtype))

    def decode_latent(latent: torch.Tensor) -> torch.Tensor:
        return updater.decode_for_reader(latent.to(device=dreamlite_device, dtype=dtype))

    def encode_feature(image: torch.Tensor) -> torch.Tensor:
        return qwen3vl_query_free_visual_features(
            model=reader,
            processor=processor,
            image=image.to(device=reader_device, dtype=dtype),
            device=reader_device,
            require_image_grad=False,
            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
        ).features

    teachers = tuple(
        build_teacher_state(
            semantic_states[state_id],
            renderer=renderer,
            contract=contract,
            encode_image=encode_image,
            decode_latent=decode_latent,
            encode_visual_feature=encode_feature,
        )
        for state_id in sorted(semantic_states)
    )
    records = tuple(TeacherArtifactRecord.from_teacher_state(teacher) for teacher in teachers)
    manifest = TeacherCacheManifest(
        teacher_contract_sha256=contract.contract_sha256,
        renderer_contract_sha256=renderer.contract_sha256,
        records=records,
    )
    by_state = {teacher.state_id: teacher for teacher in teachers}
    transitions = tuple(
        TeacherTransitionRecord(
            episode_id=raw["episode_id"],
            turn_id=raw["turn_id"],
            before_state_id=before.state_id,
            after_state_id=after.state_id,
            event_kind=raw["event_kind"],
            teacher_key=by_state[after.state_id].teacher_key,
        )
        for raw, before, after in parsed_pairs
    )

    artifact_file_sha256: dict[str, str] = {}
    for record in manifest.records:
        teacher = by_state[record.state_id]
        for tensor, specification in (
            (teacher.image, record.image),
            (teacher.latent, record.latent),
            (teacher.feature, record.feature),
        ):
            artifact_path = args.output_dir / Path(specification.relative_path)
            artifact_file_sha256[specification.relative_path] = save_teacher_tensor(
                artifact_path,
                tensor,
                specification=specification,
            )
    sidecar_path = args.output_dir / "transitions.jsonl"
    sidecar_sha256 = save_teacher_sidecar(sidecar_path, transitions, manifest=manifest)
    manifest_path = args.output_dir / "manifest.json"
    manifest_sha256 = save_teacher_manifest(manifest_path, manifest)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    report = {
        "schema": "vision_memory.r3-teacher-cache-build-report.v1",
        "git_commit": commit,
        "raw_sidecar": {"path": str(args.raw_sidecar), "sha256": sha256_file(args.raw_sidecar)},
        "model_revisions": {
            "dreamlite": locked_revision(args.dreamlite),
            "qwen_reader": locked_revision(args.reader),
        },
        "font": {"path": "assets/fonts/DejaVuSans.ttf", "sha256": LOCKED_FONT_SHA256},
        "pillow_version": PILLOW_VERSION,
        "teacher_contract_sha256": contract.contract_sha256,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "renderer_contract_sha256": renderer.contract_sha256,
        "state_count": len(teachers),
        "transition_count": len(transitions),
        "manifest_sha256": manifest_sha256,
        "sidecar_sha256": sidecar_sha256,
        "artifact_file_sha256": dict(sorted(artifact_file_sha256.items())),
        "calibration_status": "not-built; run calibrate_r3_teacher_loss.py before training",
    }
    report_path = args.output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
