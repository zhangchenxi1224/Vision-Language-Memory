from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import read_jsonl  # noqa: E402
from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.eval.teacher_retrieval import (  # noqa: E402
    TEACHER_RETRIEVAL_SCHEMA,
    compare_retrieval_retention,
    final_teacher_state_ids,
    retrieve_teacher_state,
    score_teacher_retrieval,
    teacher_cache_lock_sha256,
    validate_teacher_checkpoint_lineage,
)
from vision_memory.repro import load_initial_image  # noqa: E402
from vision_memory.teacher import (  # noqa: E402
    MANIFEST_FILENAME,
    SIDECAR_FILENAME,
    file_sha256,
    load_teacher_cache,
)
from vision_memory.training import DreamLiteEpisodeModel, load_trainable_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay train histories and retrieve the final full-state teacher latent with the locked L_z metric"
        )
    )
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument(
        "--teacher-calibration",
        type=Path,
        required=True,
        help="Immutable suite calibration file; kept outside the read-only tensor cache.",
    )
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-image", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-episodes", type=int, default=8)
    parser.add_argument("--minimum-correct", type=int, default=7)
    parser.add_argument("--tie-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--distill-reference-report",
        type=Path,
        help="For a QA-end checkpoint, compare retention against its exact distill-parent report.",
    )
    parser.add_argument("--minimum-retention", type=float, default=0.9)
    parser.add_argument(
        "--expected-teacher-control",
        choices=("correct", "shuffled", "random-moment-matched"),
        default="correct",
        help="Bind this diagnostic to the checkpoint's preregistered teacher arm.",
    )
    parser.add_argument(
        "--fail-on-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when top-1 or optional retention gate fails.",
    )
    return parser.parse_args()


def _locked_revision(path: Path) -> str:
    marker = path.expanduser().resolve(strict=True) / ".locked_revision"
    if not marker.is_file():
        raise ValueError(f"Model snapshot is missing required revision marker: {marker}")
    revision = marker.read_text(encoding="utf-8").strip()
    if not revision:
        raise ValueError(f"Model snapshot has an empty revision marker: {marker}")
    return revision


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path.expanduser().resolve(strict=True), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported DreamLite checkpoint schema.")
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("DreamLite checkpoint is missing its manifest.")
    return dict(payload), dict(manifest)


def _required_argument(arguments: Mapping[str, Any], name: str, expected_type: type) -> Any:
    value = arguments.get(name)
    if isinstance(value, bool) and expected_type is int:
        raise ValueError(f"Checkpoint argument {name!r} has invalid boolean type.")
    if not isinstance(value, expected_type):
        raise ValueError(f"Checkpoint argument {name!r} is missing or has invalid type.")
    return value


def _canonical_mapping_sha256(value: Mapping[str, str]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid retrieval reference report: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Retrieval reference report root must be an object.")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, destination)


def _build_model(
    *,
    args: argparse.Namespace,
    checkpoint_manifest: Mapping[str, Any],
    device: torch.device,
) -> tuple[DreamLiteEpisodeModel, str, dict[str, Any]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Real DreamLite teacher-state retrieval requires a CUDA device.")
    arguments = checkpoint_manifest.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("Checkpoint manifest is missing its arguments mapping.")
    adapter_seed = int(_required_argument(arguments, "adapter_seed", int))
    lora_rank = int(_required_argument(arguments, "lora_rank", int))
    global_seed = int(_required_argument(arguments, "seed", int))
    resolution = int(_required_argument(arguments, "resolution", int))
    initial_state_mode = str(_required_argument(arguments, "initial_state_mode", str))
    learn_initial_state = bool(_required_argument(arguments, "learn_initial_state", bool))
    if resolution != 1024:
        raise ValueError("R3 teacher-state retrieval is locked to 1024x1024 state images.")

    dreamlite_revision = _locked_revision(args.dreamlite)
    if checkpoint_manifest.get("dreamlite_revision") != dreamlite_revision:
        raise ValueError("DreamLite snapshot revision differs from the checkpoint manifest.")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    torch.manual_seed(adapter_seed)
    torch.cuda.manual_seed_all(adapter_seed)

    from diffusers import DreamLiteMobilePipeline
    from peft import LoraConfig, get_peft_model

    pipe = DreamLiteMobilePipeline.from_pretrained(
        args.dreamlite,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    pipe.unet.requires_grad_(False)
    torch.manual_seed(adapter_seed)
    torch.cuda.manual_seed_all(adapter_seed)
    pipe.unet = get_peft_model(
        pipe.unet,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0.0,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
    )
    pipe.unet.eval()

    checkpoint_initial = checkpoint_manifest.get("initial_image")
    if not isinstance(checkpoint_initial, Mapping):
        raise ValueError("Checkpoint manifest is missing initial-image provenance.")
    source_path = args.source_image
    if initial_state_mode == "file":
        if source_path is None:
            recorded = checkpoint_initial.get("path")
            if isinstance(recorded, str) and recorded:
                source_path = Path(recorded)
        if source_path is None:
            raise ValueError("File initial state requires --source-image or a reachable checkpoint path.")
    elif source_path is not None:
        raise ValueError("--source-image is accepted only for a file initial state.")
    source_pil, initial_metadata = load_initial_image(
        initial_state_mode,
        source_path,
        resolution=resolution,
    )
    provenance_fields = (
        "initial_state_mode",
        "origin",
        "fixture_id",
        "file_sha256",
        "rgb_sha256",
        "mode",
        "size",
    )
    drift = {
        field: (checkpoint_initial.get(field), initial_metadata.get(field))
        for field in provenance_fields
        if checkpoint_initial.get(field) != initial_metadata.get(field)
    }
    if drift:
        raise ValueError(f"Initial-image provenance differs from checkpoint: {drift}")
    source_tensor = pipe.image_processor.preprocess(source_pil, height=resolution, width=resolution)
    with torch.no_grad():
        initial_state = pipe.prepare_image_latents(source_tensor, dtype=dtype, device=device)
    model = DreamLiteEpisodeModel(
        pipeline=pipe,
        initial_state=initial_state,
        global_seed=global_seed,
        checkpoint_unet=False,
        learn_initial_state=learn_initial_state,
    )
    loaded = load_trainable_weights(args.checkpoint, trainable_module=model)
    if loaded.get("manifest") != dict(checkpoint_manifest):
        raise RuntimeError("Checkpoint manifest changed between inspection and weight loading.")
    model.eval()
    return model, dreamlite_revision, initial_metadata


def main() -> int:
    args = parse_args()
    if args.expected_episodes <= 0:
        raise ValueError("--expected-episodes must be positive.")
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    episodes_path = args.episodes.expanduser().resolve(strict=True)
    cache_root = args.teacher_cache.expanduser().resolve(strict=True)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    episodes_sha256 = file_sha256(episodes_path)
    _checkpoint_payload, checkpoint_manifest = _load_checkpoint(checkpoint_path)
    if checkpoint_manifest.get("train_sha256") != episodes_sha256:
        raise ValueError("Retrieval episodes are not the exact training JSONL recorded by the checkpoint.")

    manifest_file_sha256 = file_sha256(cache_root / MANIFEST_FILENAME)
    sidecar_file_sha256 = file_sha256(cache_root / SIDECAR_FILENAME)
    calibration_path = args.teacher_calibration.expanduser().resolve(strict=True)
    calibration_file_sha256 = file_sha256(calibration_path)
    lineage = validate_teacher_checkpoint_lineage(
        checkpoint_manifest,
        manifest_file_sha256=manifest_file_sha256,
        sidecar_file_sha256=sidecar_file_sha256,
        calibration_file_sha256=calibration_file_sha256,
        expected_teacher_control=args.expected_teacher_control,
    )
    cache = load_teacher_cache(
        cache_root,
        calibration_path=calibration_path,
        expected_manifest_file_sha256=manifest_file_sha256,
        expected_sidecar_file_sha256=sidecar_file_sha256,
        expected_calibration_file_sha256=calibration_file_sha256,
        verify_all=False,
    )
    cache_lock = teacher_cache_lock_sha256(
        manifest_file_sha256=manifest_file_sha256,
        sidecar_file_sha256=sidecar_file_sha256,
        calibration_file_sha256=calibration_file_sha256,
        manifest_payload_sha256=cache.manifest.canonical_sha256,
    )
    episodes = read_jsonl(episodes_path)
    expected_state_ids = final_teacher_state_ids(episodes, cache.sidecar)
    episode_state_contract_sha256 = _canonical_mapping_sha256(expected_state_ids)
    teacher_latents = {
        record.state_id: cache.get(record.state_id, split="train").latent for record in cache.manifest.records
    }

    device = torch.device(args.device)
    model, dreamlite_revision, initial_metadata = _build_model(
        args=args,
        checkpoint_manifest=checkpoint_manifest,
        device=device,
    )
    matches = []
    with torch.inference_mode():
        for episode in episodes:
            state = model.reset_state()
            for turn_index, turn in enumerate(episode.turns):
                if turn.calls_updater:
                    if turn.event_text is None:
                        raise RuntimeError("Updater route is missing event_text.")
                    state = model.updater(state, turn.event_text, episode.episode_id, turn_index)
            matches.append(
                retrieve_teacher_state(
                    episode_id=episode.episode_id,
                    student_latent=state,
                    expected_state_id=expected_state_ids[episode.episode_id],
                    teacher_latents=teacher_latents,
                    tie_tolerance=args.tie_tolerance,
                )
            )
    summary = score_teacher_retrieval(
        matches,
        expected_episodes=args.expected_episodes,
        minimum_correct=args.minimum_correct,
    )
    report: dict[str, Any] = {
        "schema": TEACHER_RETRIEVAL_SCHEMA,
        "objective_stage": lineage["objective_stage"],
        "training_regime": lineage["training_regime"],
        "teacher_control": lineage["teacher_control"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "episodes": str(episodes_path),
        "episodes_sha256": episodes_sha256,
        "teacher_cache": str(cache_root),
        "teacher_calibration": str(calibration_path),
        "teacher_cache_lock_sha256": cache_lock,
        "teacher_cache_files": {
            "manifest_sha256": manifest_file_sha256,
            "sidecar_sha256": sidecar_file_sha256,
            "calibration_sha256": calibration_file_sha256,
            "manifest_payload_sha256": cache.manifest.canonical_sha256,
        },
        "episode_state_contract_sha256": episode_state_contract_sha256,
        "candidate_state_count": len(teacher_latents),
        "dreamlite_revision": dreamlite_revision,
        "reader_revision": checkpoint_manifest.get("reader_revision"),
        "initial_image": initial_metadata,
        "training_lineage": lineage,
        "distance_contract": {
            "metric": "smooth-l1-after-independent-per-channel-standardization",
            "normalization_epsilon": 1e-6,
            "candidate_set": "complete-locked-teacher-cache",
            "ties": "top-distance-tie-is-incorrect",
            "tie_tolerance": args.tie_tolerance,
        },
        "results": [match.to_dict() for match in matches],
        "summary": summary,
    }
    retention = None
    if args.distill_reference_report is not None:
        retention = compare_retrieval_retention(
            reference_report=_load_json_object(args.distill_reference_report),
            current_report=report,
            minimum_retention=args.minimum_retention,
        )
        report["retention"] = retention
    elif lineage["objective_stage"] == "qa" and lineage["teacher_control"] == "correct":
        raise ValueError("Correct-teacher QA retrieval requires --distill-reference-report.")
    _write_json_atomic(args.output, report)
    passed = bool(summary["gate_passed"]) and (retention is None or bool(retention["gate_passed"]))
    print(json.dumps({"output": str(args.output.resolve()), "passed": passed, **summary}, sort_keys=True))
    return 0 if passed or not args.fail_on_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
