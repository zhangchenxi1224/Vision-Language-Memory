from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import assert_no_frozen_parameter_grads, freeze_module  # noqa: E402
from vision_memory.data import CYCLIC4, REVERSE_CYCLIC4, permutation_family_sha256  # noqa: E402
from vision_memory.data import read_jsonl as read_episode_jsonl  # noqa: E402
from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_RESIZE_CONTRACT,
    qwen3vl_listwise_choice_ce,
    qwen3vl_query_free_visual_features,
    qwen3vl_target_only_ce,
)
from vision_memory.repro import (  # noqa: E402
    canonical_tensor_sha256,
    configure_strict_cuda_determinism,
    load_initial_image,
)
from vision_memory.teacher import (  # noqa: E402
    TeacherState,
    TeacherTransitionRecord,
    composite_teacher_distillation_loss,
    load_teacher_cache_manifest,
    load_teacher_calibration,
    load_teacher_transition_sidecar,
    make_disk_teacher_provider,
)
from vision_memory.training import (  # noqa: E402
    DreamLiteEpisodeModel,
    load_trainable_weights,
    load_training_checkpoint,
    read_prefeval_adapted_jsonl,
    read_prefeval_supervised_jsonl,
    run_episode,
    save_training_checkpoint,
    select_curriculum_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Episode-level DreamLite latent-memory training")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument(
        "--dataset-format",
        choices=("synthetic", "prefeval-export", "prefeval-supervised"),
        default="synthetic",
        help="PrefEval export means the separated model_input/label JSONL from prepare_prefeval.py.",
    )
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument(
        "--reader-loss-mode",
        choices=("listwise-choice", "target-only"),
        default="listwise-choice",
        help="Formal R3 training uses listwise choice CE; target-only is retained for legacy diagnostics.",
    )
    parser.add_argument(
        "--choice-view-schedule",
        choices=("cyclic4", "canonical"),
        default="cyclic4",
        help="Deterministically rotate four-choice prompts during training and average all views for dev.",
    )
    parser.add_argument(
        "--training-regime",
        choices=("qa_only", "teacher_assisted"),
        default="qa_only",
        help="Fail-closed supervision lineage recorded in manifests and checkpoints.",
    )
    parser.add_argument(
        "--objective-stage",
        choices=("qa", "distill"),
        default="qa",
        help="R3 stages are disjoint: teacher distillation first, then a teacher-free QA stage.",
    )
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        help="Train-only cache manifest; required only for teacher_assisted distill.",
    )
    parser.add_argument("--teacher-sidecar", type=Path, help="Locked train-only transition sidecar for distillation")
    parser.add_argument("--teacher-calibration", type=Path, help="Frozen three-component calibration JSON")
    parser.add_argument(
        "--teacher-control",
        choices=("correct", "shuffled", "random-moment-matched"),
        default="correct",
        help="Pre-registered attribution control; inherited permanently by the teacher lineage.",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Weights-only parent checkpoint. Required for teacher-assisted QA; optimizer/RNG start fresh.",
    )
    parser.add_argument("--presentations-per-state", type=int)
    parser.add_argument("--distill-presentations", type=int, default=0)
    parser.add_argument("--qa-presentations", type=int, default=0)
    parser.add_argument(
        "--initial-state-mode",
        choices=("blank", "fixture", "file"),
        default="blank",
        help="Formal default is a uniform neutral-gray blank; fixture is probe-only and file requires --source-image.",
    )
    parser.add_argument("--source-image", type=Path, help="Accepted only with --initial-state-mode file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0, help="Episode order and per-event noise seed")
    parser.add_argument("--adapter-seed", type=int, default=0, help="LoRA initialization seed")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument(
        "--eval-start-step",
        type=int,
        default=1,
        help="First optimizer step eligible for dev evaluation; R3 micro gates start at 64 presentations/state.",
    )
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Record scheduled dev losses without allowing them to shorten a locked scientific budget.",
    )
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        help="Fixed technical-probe budget; formal scientific runs normally leave this unset.",
    )
    parser.add_argument(
        "--audit-gradient-sha",
        action="store_true",
        help="Record bitwise raw/clipped gradient payloads for DL-S; incurs synchronization overhead.",
    )
    parser.add_argument(
        "--audit-state-gradients",
        action="store_true",
        help="Fail closed unless every expected micro-run state/image tensor has a finite non-zero gradient.",
    )
    parser.add_argument(
        "--require-mixed-delayed-probe",
        action="store_true",
        help=(
            "Fail closed unless each mixed read has a same-target, same-choice-multiset pure-query "
            "probe before the next updater. Formal R3 runs require this flag."
        ),
    )
    parser.add_argument(
        "--strict-determinism",
        action="store_true",
        help=(
            "Enable the fail-closed math-only CUDA determinism contract. The required "
            "process environment must be exported before Python starts."
        ),
    )
    parser.add_argument("--recurrence-mode", choices=["direct_latent", "decode_reencode"], default="direct_latent")
    parser.add_argument("--detach-between-events", action="store_true")
    parser.add_argument(
        "--noop-policy",
        choices=("update", "skip"),
        default="update",
        help="Whether labeled noop/distractor events call the updater; recorded in the manifest and route trace.",
    )
    parser.add_argument(
        "--curriculum",
        choices=("full", "set-only"),
        default="full",
        help="set-only selects whole training episodes containing only set/noop updater labels.",
    )
    parser.add_argument("--learn-initial-state", action="store_true")
    parser.add_argument("--checkpoint-unet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


def locked_revision(path: Path) -> str:
    marker = path / ".locked_revision"
    if not marker.is_file():
        raise RuntimeError(f"Model snapshot is missing required revision marker: {marker}")
    revision = marker.read_text(encoding="utf-8").strip()
    if not revision:
        raise RuntimeError(f"Model snapshot has an empty revision marker: {marker}")
    return revision


def compute_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        raise ValueError("Real DreamLite episode training requires CUDA devices.")
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _checkpoint_lineage(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError(f"Parent checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("manifest"), dict):
        raise ValueError("Parent checkpoint does not contain a supported training manifest.")
    manifest = payload["manifest"]
    if manifest.get("reader_resize_contract") != R3_QWEN_READER_RESIZE_CONTRACT:
        raise ValueError("Parent checkpoint has a missing or incompatible Reader resize contract.")
    lineage = manifest.get("training_lineage")
    if not isinstance(lineage, dict):
        raise ValueError("Parent checkpoint is missing training_lineage.")
    return lineage, sha256_file(path)


def _required_file_sha(path: Path | None, *, option: str) -> str:
    if path is None:
        raise ValueError(f"{option} is required for this training stage.")
    if not path.is_file():
        raise ValueError(f"{option} does not exist: {path}")
    return sha256_file(path)


def teacher_control_contract(manifest_path: Path, control: str) -> tuple[str, dict[str, str]]:
    manifest = load_teacher_cache_manifest(manifest_path)
    state_ids = sorted(manifest.by_state_id)
    mapping: dict[str, str]
    if control == "correct":
        mapping = {state_id: state_id for state_id in state_ids}
    elif control == "shuffled":
        if len(state_ids) < 2:
            raise ValueError("Shuffled-teacher control requires at least two cached states.")
        mapping = {state_id: state_ids[(index + 1) % len(state_ids)] for index, state_id in enumerate(state_ids)}
        if any(source == target for source, target in mapping.items()):
            raise RuntimeError("Fixed shuffled-teacher mapping is not a derangement.")
    elif control == "random-moment-matched":
        mapping = {state_id: state_id for state_id in state_ids}
    else:
        raise ValueError(f"Unknown teacher control: {control!r}.")
    payload = {
        "schema": "vision_memory.teacher-control.v1",
        "control": control,
        "algorithm": (
            "identity"
            if control == "correct"
            else "sorted-state-id-rotate-one-derangement"
            if control == "shuffled"
            else "deterministic-independent-within-channel-permutation"
        ),
        "mapping": mapping,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, mapping


def training_lineage(args: argparse.Namespace) -> dict[str, Any]:
    training_regime = str(args.training_regime)
    objective_stage = str(getattr(args, "objective_stage", "qa"))
    reader_loss_mode = str(args.reader_loss_mode)
    teacher_manifest = getattr(args, "teacher_manifest", None)
    teacher_sidecar = getattr(args, "teacher_sidecar", None)
    teacher_calibration = getattr(args, "teacher_calibration", None)
    initialize_from = getattr(args, "initialize_from", None)
    requested_teacher_control = str(getattr(args, "teacher_control", "correct"))
    parent_lineage: dict[str, Any] | None = None
    parent_checkpoint_sha256: str | None = None
    if initialize_from is not None:
        parent_lineage, parent_checkpoint_sha256 = _checkpoint_lineage(initialize_from)

    if training_regime == "qa_only":
        if objective_stage != "qa":
            raise ValueError("qa_only training supports only --objective-stage qa.")
        if requested_teacher_control != "correct":
            raise ValueError("qa_only training forbids teacher controls.")
        if any(value is not None for value in (teacher_manifest, teacher_sidecar, teacher_calibration)):
            raise ValueError("qa_only training forbids teacher cache, sidecar, calibration, and derived losses.")
        if parent_lineage is not None and parent_lineage.get("training_regime") != "qa_only":
            raise ValueError("A teacher-lineage checkpoint can never initialize a qa_only result.")
        teacher_manifest_sha256 = None
        teacher_sidecar_sha256 = None
        teacher_calibration_sha256 = None
        parent_checkpoint_regime = None if parent_lineage is None else "qa_only"
        inherited_distill_presentations = 0
        teacher_control = "none"
        teacher_control_sha256 = None
    elif training_regime == "teacher_assisted":
        if objective_stage == "distill":
            if initialize_from is not None:
                raise ValueError("teacher-assisted distillation must use a fresh LoRA, not --initialize-from.")
            teacher_manifest_sha256 = _required_file_sha(teacher_manifest, option="--teacher-manifest")
            teacher_sidecar_sha256 = _required_file_sha(teacher_sidecar, option="--teacher-sidecar")
            teacher_calibration_sha256 = _required_file_sha(
                teacher_calibration,
                option="--teacher-calibration",
            )
            parent_checkpoint_regime = None
            inherited_distill_presentations = 0
            teacher_control = requested_teacher_control
            assert teacher_manifest is not None
            teacher_control_sha256, _ = teacher_control_contract(teacher_manifest, teacher_control)
        elif objective_stage == "qa":
            if any(value is not None for value in (teacher_manifest, teacher_sidecar, teacher_calibration)):
                raise ValueError(
                    "teacher-assisted QA must fully unload teacher inputs and accept only --initialize-from."
                )
            if parent_lineage is None:
                raise ValueError("teacher-assisted QA requires --initialize-from a distill-only checkpoint.")
            if (
                parent_lineage.get("training_regime") != "teacher_assisted"
                or parent_lineage.get("objective_stage") != "distill"
            ):
                raise ValueError("teacher-assisted QA parent must be a teacher_assisted distill checkpoint.")
            teacher_manifest_sha256 = parent_lineage.get("teacher_manifest_sha256")
            teacher_sidecar_sha256 = parent_lineage.get("teacher_sidecar_sha256")
            teacher_calibration_sha256 = parent_lineage.get("teacher_calibration_sha256")
            if not all(
                isinstance(value, str) and len(value) == 64
                for value in (
                    teacher_manifest_sha256,
                    teacher_sidecar_sha256,
                    teacher_calibration_sha256,
                )
            ):
                raise ValueError("Teacher-assisted parent is missing locked teacher artifact hashes.")
            parent_checkpoint_regime = "teacher_assisted"
            inherited_distill_presentations = int(parent_lineage.get("distill_presentations", 0))
            teacher_control = str(parent_lineage.get("teacher_control", ""))
            teacher_control_sha256 = parent_lineage.get("teacher_control_sha256")
            if requested_teacher_control != teacher_control:
                raise ValueError("Teacher-assisted QA --teacher-control must match its parent lineage.")
            if not isinstance(teacher_control_sha256, str) or len(teacher_control_sha256) != 64:
                raise ValueError("Teacher-assisted parent is missing its teacher-control contract SHA256.")
        else:
            raise ValueError("teacher_assisted objective stage must be 'distill' or 'qa'.")
    else:
        raise ValueError("training_regime must be 'qa_only' or 'teacher_assisted'.")

    presentations = getattr(args, "presentations_per_state", None)
    presentations = int(getattr(args, "epochs", 1) if presentations is None else presentations)
    if presentations <= 0:
        raise ValueError("presentations_per_state must be positive.")
    distill_presentations = int(getattr(args, "distill_presentations", 0))
    qa_presentations = int(getattr(args, "qa_presentations", 0))
    if distill_presentations < 0 or qa_presentations < 0:
        raise ValueError("distill/qa presentations cannot be negative.")
    if training_regime == "qa_only":
        if distill_presentations != 0:
            raise ValueError("qa_only lineage cannot record distillation presentations.")
        qa_presentations = qa_presentations or presentations
        total_presentations = qa_presentations
    elif objective_stage == "distill":
        if qa_presentations != 0:
            raise ValueError("A distill-only stage cannot record QA presentations.")
        distill_presentations = distill_presentations or presentations
        total_presentations = distill_presentations
    else:
        if distill_presentations not in {0, inherited_distill_presentations}:
            raise ValueError("QA-stage distill presentations must match the parent lineage.")
        distill_presentations = inherited_distill_presentations
        qa_presentations = qa_presentations or presentations
        total_presentations = distill_presentations + qa_presentations

    return {
        "schema_version": 2,
        "training_regime": training_regime,
        "parent_checkpoint_regime": parent_checkpoint_regime,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "objective_stage": objective_stage,
        "reader_loss_mode": reader_loss_mode,
        "qa_supervision": "listwise-choice" if reader_loss_mode == "listwise-choice" else "legacy-R1-target-only",
        "choice_view_schedule": str(args.choice_view_schedule),
        "choice_permutation_family_sha256": permutation_family_sha256(CYCLIC4),
        "eval_choice_permutation_family_sha256": permutation_family_sha256(REVERSE_CYCLIC4),
        "teacher_manifest_sha256": teacher_manifest_sha256,
        "teacher_sidecar_sha256": teacher_sidecar_sha256,
        "teacher_calibration_sha256": teacher_calibration_sha256,
        "teacher_control": teacher_control,
        "teacher_control_sha256": teacher_control_sha256,
        "presentations_per_state": total_presentations,
        "distill_presentations": distill_presentations,
        "qa_presentations": qa_presentations,
        "teacher_supervision_loaded": training_regime == "teacher_assisted" and objective_stage == "distill",
        "teacher_checkpoint_is_qa_only_eligible": training_regime == "qa_only",
    }


def state_gradient_audit_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Return the immutable R3 gradient-evidence contract stored in each manifest."""

    enabled = bool(getattr(args, "audit_state_gradients", False))
    objective_stage = str(getattr(args, "objective_stage", "qa"))
    if objective_stage == "qa":
        required_always = ("final_state", "query_image")
    elif objective_stage == "distill":
        required_always = ("final_state", "state_image", "student_visual_feature")
    else:
        raise ValueError("State-gradient audit supports only qa or distill objective stages.")
    return {
        "schema": "vision_memory.r3-state-gradient-audit.v1",
        "enabled": enabled,
        "objective_stage": objective_stage,
        "required_always": list(required_always),
        "required_when_two_or_more_updates": ["first_intermediate_state"],
        "acceptance": "every retained tensor gradient is present, finite, and has L2 norm > 0",
        "scope": "every training episode backward in this process segment",
    }


def make_manifest(args: argparse.Namespace) -> dict[str, Any]:
    commit = git_value("rev-parse", "HEAD")
    status = git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise RuntimeError("Formal training refuses a dirty worktree. Commit or pass --allow-dirty for debugging only.")
    compatibility_args = serializable_args(args)
    for excluded in ("resume", "initialize_from", "output_dir", "allow_dirty"):
        compatibility_args.pop(excluded, None)
    lineage = training_lineage(args)
    model_snapshot_manifests = {
        "dreamlite_mobile": os.environ.get("VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256"),
        "qwen_reader": os.environ.get("VLM_READER_SNAPSHOT_MANIFEST_SHA256"),
    }
    if args.strict_determinism and any(
        not isinstance(value, str) or len(value) != 64 or value != value.lower()
        for value in model_snapshot_manifests.values()
    ):
        raise RuntimeError("Formal strict R3 training requires both model snapshot manifest SHA256 bindings.")
    return {
        "schema_version": 2,
        "git_commit": commit,
        "git_dirty": bool(status),
        "dreamlite_revision": locked_revision(args.dreamlite),
        "reader_revision": locked_revision(args.reader),
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "model_snapshot_manifests": model_snapshot_manifests,
        "train_sha256": sha256_file(args.train),
        "dev_sha256": sha256_file(args.dev),
        "initial_image": load_initial_image(
            args.initial_state_mode,
            args.source_image,
            resolution=args.resolution,
        )[1],
        "state_gradient_audit_contract": state_gradient_audit_contract(args),
        "training_lineage": lineage,
        "arguments": compatibility_args,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "cuda_runtime": torch.version.cuda,
            "diffusers": importlib.metadata.version("diffusers"),
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
        },
    }


def assert_frozen_contract(pipe: Any, reader: torch.nn.Module) -> None:
    assert_no_frozen_parameter_grads(pipe.vae, "VAE")
    assert_no_frozen_parameter_grads(pipe.text_encoder, "DreamLite conditioner")
    assert_no_frozen_parameter_grads(pipe.unet, "DreamLite base U-Net")
    assert_no_frozen_parameter_grads(reader, "Qwen Reader")


class StateSupervisionProvider(Protocol):
    """Teacher callback contract; concrete providers live outside this unified trainer."""

    def __call__(self, state: torch.Tensor, episode_id: str, turn_id: str | int) -> Any: ...


def _per_channel_random_permutation(tensor: torch.Tensor, *, seed_key: str, feature: bool) -> torch.Tensor:
    source = tensor.detach().cpu().contiguous()
    if source.shape[0] != 1 or source.ndim < 3:
        raise ValueError("Moment-matched teacher control expects a batch-one tensor with at least three axes.")
    if feature:
        if source.ndim != 3:
            raise ValueError("Teacher visual features must have shape [1,tokens,hidden].")
        matrix = source[0].transpose(0, 1).contiguous()
    else:
        matrix = source[0].reshape(source.shape[1], -1)
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    width = matrix.shape[1]
    base_permutation = torch.randperm(width, generator=generator)
    offsets = torch.randint(width, (matrix.shape[0], 1), generator=generator)
    shifted = (torch.arange(width).unsqueeze(0) + offsets) % width
    random_order = base_permutation[shifted]
    permuted = matrix.gather(1, random_order)
    if feature:
        result = permuted.transpose(0, 1).unsqueeze(0)
    else:
        result = permuted.reshape(source.shape)
    # A within-channel permutation preserves every per-channel moment exactly.
    return result.contiguous()


def random_moment_matched_teacher(teacher: TeacherState) -> TeacherState:
    teacher_key = hashlib.sha256(
        f"vision_memory.random-moment-matched.v1\0{teacher.teacher_key}".encode("utf-8")
    ).hexdigest()
    return TeacherState(
        state_id=teacher.state_id,
        teacher_key=teacher_key,
        semantic_state_sha256=teacher.semantic_state_sha256,
        teacher_contract_sha256=teacher.teacher_contract_sha256,
        renderer_contract_sha256=teacher.renderer_contract_sha256,
        image=_per_channel_random_permutation(
            teacher.image,
            seed_key=f"{teacher.state_id}:image",
            feature=False,
        ),
        latent=_per_channel_random_permutation(
            teacher.latent,
            seed_key=f"{teacher.state_id}:latent",
            feature=False,
        ),
        feature=_per_channel_random_permutation(
            teacher.feature,
            seed_key=f"{teacher.state_id}:feature",
            feature=True,
        ),
    )


def build_state_supervision_provider(
    *,
    args: argparse.Namespace,
    model: DreamLiteEpisodeModel,
    pipe: Any,
    reader: torch.nn.Module,
    processor: Any,
    expected_lineage: dict[str, Any],
    gradient_audit_tensors: list[tuple[str, torch.Tensor]] | None = None,
) -> StateSupervisionProvider | None:
    """Build the locked train-only full-state teacher boundary.

    The callback accepts only the post-update student latent plus oracle episode/turn IDs.
    Query text, choices, target indices, and model-visible episode metadata never cross this
    boundary. QA stages return before loading any teacher artifact.
    """

    del pipe
    if args.training_regime != "teacher_assisted" or args.objective_stage != "distill":
        return None
    assert args.teacher_manifest is not None
    assert args.teacher_sidecar is not None
    assert args.teacher_calibration is not None
    manifest_sha256 = sha256_file(args.teacher_manifest)
    sidecar_sha256 = sha256_file(args.teacher_sidecar)
    calibration_sha256 = sha256_file(args.teacher_calibration)
    observed_hashes = {
        "teacher_manifest_sha256": manifest_sha256,
        "teacher_sidecar_sha256": sidecar_sha256,
        "teacher_calibration_sha256": calibration_sha256,
    }
    for field, observed in observed_hashes.items():
        expected = expected_lineage.get(field)
        if expected != observed:
            raise RuntimeError(
                f"Teacher artifact changed after manifest lock: {field} expected {expected}, observed {observed}."
            )
    manifest = load_teacher_cache_manifest(
        args.teacher_manifest,
        expected_file_sha256=manifest_sha256,
    )
    transitions = load_teacher_transition_sidecar(
        args.teacher_sidecar,
        manifest=manifest,
        expected_file_sha256=sidecar_sha256,
    )
    calibration = load_teacher_calibration(
        args.teacher_calibration,
        expected_file_sha256=calibration_sha256,
    )
    disk_provider = make_disk_teacher_provider(
        args.teacher_manifest,
        expected_manifest_file_sha256=manifest_sha256,
    )
    # Fail before the first optimizer step if any tensor is missing, corrupt, or drifted.
    for record in manifest.records:
        disk_provider.get(record.state_id, split="train")

    transition_lookup: dict[tuple[str, str], TeacherTransitionRecord] = {}
    for transition in transitions:
        key = (transition.episode_id, str(transition.turn_id))
        if key in transition_lookup:
            raise ValueError(f"Duplicate teacher transition key: {key!r}.")
        transition_lookup[key] = transition
    reader_device = next(reader.parameters()).device
    _control_sha256, control_mapping = teacher_control_contract(args.teacher_manifest, args.teacher_control)
    if sha256_file(args.teacher_manifest) != manifest_sha256:
        raise RuntimeError("Teacher manifest changed while constructing the control mapping.")

    @functools.lru_cache(maxsize=32)
    def cached_teacher(state_id: str):
        return disk_provider.get(state_id, split="train")

    @functools.lru_cache(maxsize=32)
    def controlled_teacher(state_id: str):
        selected = cached_teacher(control_mapping[state_id])
        if args.teacher_control == "random-moment-matched":
            return random_moment_matched_teacher(selected)
        return selected

    def supervise(state: torch.Tensor, episode_id: str, turn_id: str | int):
        transition = transition_lookup.get((episode_id, str(turn_id)))
        if transition is None:
            raise KeyError(f"No locked teacher transition for episode={episode_id!r}, turn={turn_id!r}.")
        true_teacher = cached_teacher(transition.after_state_id)
        if true_teacher.teacher_key != transition.teacher_key:
            raise RuntimeError("Teacher sidecar/cache key drifted after provider construction.")
        teacher = controlled_teacher(transition.after_state_id)
        student_image = model.updater.decode_for_reader(state).to(reader_device)
        if gradient_audit_tensors is not None and torch.is_grad_enabled():
            retain_gradient_audit_tensor(
                gradient_audit_tensors,
                category="state_image",
                tensor=student_image,
            )
        student_feature = qwen3vl_query_free_visual_features(
            model=reader,
            processor=processor,
            image=student_image,
            device=reader_device,
            require_image_grad=torch.is_grad_enabled(),
            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
        ).features
        if gradient_audit_tensors is not None and torch.is_grad_enabled():
            retain_gradient_audit_tensor(
                gradient_audit_tensors,
                category="student_visual_feature",
                tensor=student_feature,
            )
        return composite_teacher_distillation_loss(
            student_latent=state.to(reader_device),
            student_image=student_image,
            student_feature=student_feature,
            teacher=teacher,
            calibration=calibration,
        )

    return supervise


def evaluate_distillation(
    *,
    model: DreamLiteEpisodeModel,
    records: Iterable[dict[str, Any]],
    recurrence_mode: str,
    detach_between_events: bool,
    noop_policy: str,
    state_supervision_fn: StateSupervisionProvider,
    require_mixed_delayed_probe: bool = False,
) -> dict[str, float]:
    totals = {"distill_loss": 0.0, "latent_raw": 0.0, "image_raw": 0.0, "feature_raw": 0.0}
    count = 0
    with torch.no_grad():
        for episode in records:
            result = run_episode(
                episode=episode,
                initial_state=model.reset_state(),
                update_fn=model.updater,
                decode_fn=model.updater.decode_for_reader,
                reader_loss_mode="listwise-choice",
                recurrence_mode=recurrence_mode,
                reencode_fn=model.updater.reencode_posterior_mean,
                reencode_decode_fn=model.updater.decode_for_reencode,
                detach_between_events=detach_between_events,
                collect_states=False,
                noop_policy=noop_policy,
                training_regime="teacher_assisted",
                state_supervision_fn=state_supervision_fn,
                objective_stage="distill",
                require_mixed_delayed_probe=require_mixed_delayed_probe,
            )
            assert result.state_supervision_loss is not None
            assert result.latent_distill_loss is not None
            assert result.image_distill_loss is not None
            assert result.visual_feature_distill_loss is not None
            totals["distill_loss"] += float(result.state_supervision_loss)
            totals["latent_raw"] += float(result.latent_distill_loss)
            totals["image_raw"] += float(result.image_distill_loss)
            totals["feature_raw"] += float(result.visual_feature_distill_loss)
            count += 1
    if count == 0:
        raise ValueError("Distillation evaluation requires at least one episode.")
    return {name: value / count for name, value in totals.items()}


def target_reader_callable(
    *,
    reader: Any,
    processor: Any,
    reader_device: torch.device,
    require_grad: bool,
    deterministic_ce: bool = False,
    gradient_audit_tensors: list[tuple[str, torch.Tensor]] | None = None,
):
    def call(image: torch.Tensor, query: str, target: str):
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError("Reader currently supports one state image per episode.")
            image = image[0]
        image = image.to(reader_device)
        if gradient_audit_tensors is not None and require_grad:
            retain_gradient_audit_tensor(
                gradient_audit_tensors,
                category="query_image",
                tensor=image,
            )
        return qwen3vl_target_only_ce(
            model=reader,
            processor=processor,
            image=image,
            query=query,
            target=target,
            device=reader_device,
            require_image_grad=require_grad,
            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
            deterministic_ce=deterministic_ce,
        )

    return call


def choice_reader_callable(
    *,
    reader: Any,
    processor: Any,
    reader_device: torch.device,
    require_grad: bool,
    deterministic_ce: bool = False,
    gradient_audit_tensors: list[tuple[str, torch.Tensor]] | None = None,
):
    def call(image: torch.Tensor, query: str, choices: tuple[str, ...], target_index: int):
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError("Reader currently supports one state image per episode.")
            image = image[0]
        image = image.to(reader_device)
        if gradient_audit_tensors is not None and require_grad:
            retain_gradient_audit_tensor(
                gradient_audit_tensors,
                category="query_image",
                tensor=image,
            )
        return qwen3vl_listwise_choice_ce(
            model=reader,
            processor=processor,
            image=image,
            query=query,
            choices=choices,
            target_index=target_index,
            device=reader_device,
            require_image_grad=require_grad,
            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
            deterministic_ce=deterministic_ce,
        )

    return call


def rotate_choice_view(
    choices: tuple[str, ...],
    target_index: int,
    *,
    rotation: int,
) -> tuple[tuple[str, ...], int]:
    if len(choices) != 4:
        raise ValueError("cyclic4 choice scheduling requires exactly four choices.")
    if isinstance(target_index, bool) or not isinstance(target_index, int) or not 0 <= target_index < 4:
        raise ValueError("target_index must be an integer in [0, 3].")
    normalized = rotation % 4
    return choices[normalized:] + choices[:normalized], (target_index - normalized) % 4


def choice_view_for_rotation(schedule: str, rotation: int):
    if schedule == "canonical":
        return None
    if schedule != "cyclic4":
        raise ValueError("choice_view_schedule must be 'canonical' or 'cyclic4'.")

    def view(
        _episode_id: str,
        _turn_id: str | int,
        choices: tuple[str, ...],
        target_index: int,
    ) -> tuple[tuple[str, ...], int]:
        return rotate_choice_view(choices, target_index, rotation=rotation)

    return view


def choice_view_for_permutation(schedule: str, permutation: tuple[int, int, int, int]):
    if schedule == "canonical":
        return None
    if schedule != "cyclic4":
        raise ValueError("choice_view_schedule must be 'canonical' or 'cyclic4'.")
    if len(permutation) != 4 or set(permutation) != {0, 1, 2, 3}:
        raise ValueError("Choice permutation must contain indices 0, 1, 2, and 3 exactly once.")

    def view(
        _episode_id: str,
        _turn_id: str | int,
        choices: tuple[str, ...],
        target_index: int,
    ) -> tuple[tuple[str, ...], int]:
        if len(choices) != 4:
            raise ValueError("R3 choice permutations require exactly four choices.")
        permuted = tuple(choices[index] for index in permutation)
        return permuted, permutation.index(target_index)

    return view


def choice_rotation_for_training(
    schedule: str,
    *,
    epoch: int,
    position: int,
    episodes_per_epoch: int,
    schedule_key: str | None = None,
) -> int:
    """Return the deterministic choice view for one episode exposure.

    Formal R3 calls supply an answer-agnostic state key, so each state advances exactly one
    cyclic view per presentation/epoch regardless of shuffled batch position.  The legacy
    keyless branch remains deterministic for older diagnostics.
    """

    if schedule == "canonical":
        return 0
    if schedule != "cyclic4":
        raise ValueError("choice_view_schedule must be 'canonical' or 'cyclic4'.")
    if epoch < 0 or position < 0 or episodes_per_epoch <= 0 or position >= episodes_per_epoch:
        raise ValueError("Training choice rotation requires a valid epoch, cursor, and epoch size.")
    if schedule_key is None:
        return (epoch * episodes_per_epoch + position) % 4
    if not schedule_key.strip():
        raise ValueError("choice schedule_key must be non-empty when supplied.")
    phase = int.from_bytes(hashlib.sha256(schedule_key.encode("utf-8")).digest()[:2], "big") % 4
    return (epoch + phase) % 4


def episode_choice_schedule_key(episode: Any) -> str:
    """Extract an answer-agnostic key so every semantic state cycles exactly over epochs."""

    if isinstance(episode, dict):
        turns = episode.get("turns", ())
        episode_id = str(episode.get("episode_id", ""))
        entity_id = str(episode.get("entity_id", ""))
        template_id = str(episode.get("template_id", ""))
    else:
        turns = getattr(episode, "turns", ())
        episode_id = str(getattr(episode, "episode_id", ""))
        entity_id = str(getattr(episode, "entity_id", ""))
        template_id = str(getattr(episode, "template_id", ""))
    comparison_ids: set[str] = set()
    for turn in turns:
        if isinstance(turn, dict):
            query = turn.get("query")
            if isinstance(query, dict) and query.get("comparison_id"):
                comparison_ids.add(str(query["comparison_id"]))
        else:
            query = getattr(turn, "query", None)
            comparison_id = getattr(query, "comparison_id", None)
            if comparison_id:
                comparison_ids.add(str(comparison_id))
    if comparison_ids:
        return "comparison:" + "|".join(sorted(comparison_ids))
    if not entity_id or not template_id:
        raise ValueError(f"Episode {episode_id!r} lacks an answer-agnostic choice scheduling key.")
    return f"entity-template:{entity_id}|{template_id}"


def episode_order(records: list[Any], seed: int, epoch: int) -> list[int]:
    order = list(range(len(records)))
    random.Random((seed << 16) ^ epoch).shuffle(order)
    return order


def evaluate_dev(
    *,
    model: DreamLiteEpisodeModel,
    records: Iterable[dict[str, Any]],
    recurrence_mode: str,
    detach_between_events: bool,
    noop_policy: str,
    reader_loss_mode: str,
    reader_loss_fn,
    choice_reader_loss_fn,
    choice_view_schedule: str,
    require_mixed_delayed_probe: bool = False,
) -> float:
    losses: list[float] = []
    permutations = REVERSE_CYCLIC4 if choice_view_schedule == "cyclic4" else ((0, 1, 2, 3),)
    with torch.no_grad():
        for episode in records:
            for permutation in permutations:
                episode_kwargs: dict[str, Any] = {
                    "episode": episode,
                    "initial_state": model.reset_state(),
                    "update_fn": model.updater,
                    "decode_fn": model.updater.decode_for_reader,
                    "reencode_fn": model.updater.reencode_posterior_mean,
                    "reencode_decode_fn": model.updater.decode_for_reencode,
                    "reader_loss_mode": reader_loss_mode,
                    "recurrence_mode": recurrence_mode,
                    "detach_between_events": detach_between_events,
                    "collect_states": False,
                    "noop_policy": noop_policy,
                    "choice_view_fn": choice_view_for_permutation(choice_view_schedule, permutation),
                    "training_regime": "qa_only",
                    "require_mixed_delayed_probe": require_mixed_delayed_probe,
                }
                if reader_loss_mode == "listwise-choice":
                    episode_kwargs["choice_reader_loss_fn"] = choice_reader_loss_fn
                else:
                    episode_kwargs["reader_loss_fn"] = reader_loss_fn
                result = run_episode(**episode_kwargs)
                losses.append(float(result.qa_loss.item()))
    return sum(losses) / len(losses)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def audit_episode_gradients(
    tensors: Iterable[tuple[str, torch.Tensor]],
    accumulator: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Fail closed and record every retained state/image gradient after one backward."""

    observed: dict[str, list[float]] = {}
    for category, tensor in tensors:
        gradient = tensor.grad
        if gradient is None:
            raise RuntimeError(f"R3 micro gradient audit found no gradient for {category}.")
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"R3 micro gradient audit found NaN/Inf for {category}.")
        norm = float(torch.linalg.vector_norm(gradient.float()).item())
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f"R3 micro gradient audit found non-positive {category} norm: {norm}.")
        accumulator.setdefault(category, []).append(norm)
        observed.setdefault(category, []).append(norm)
    return observed


def retain_gradient_audit_tensor(
    tensors: list[tuple[str, torch.Tensor]],
    *,
    category: str,
    tensor: torch.Tensor,
) -> None:
    """Retain a non-leaf audit target and reject a disconnected tensor before backward."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"R3 micro gradient audit target {category!r} is not a tensor.")
    if not tensor.requires_grad:
        raise RuntimeError(f"R3 micro gradient audit target {category!r} does not require gradients.")
    if any(existing_category == category and existing is tensor for existing_category, existing in tensors):
        raise RuntimeError(f"R3 micro gradient audit registered {category!r} twice for the same tensor.")
    tensor.retain_grad()
    tensors.append((category, tensor))


def gradient_audit_summary(
    accumulator: dict[str, list[float]],
    expected_counts: Counter[str] | None = None,
    *,
    contract: dict[str, Any] | None = None,
    multi_update_episode_count: int | None = None,
    enabled: bool | None = None,
    objective_stage: str | None = None,
) -> dict[str, Any]:
    # ``enabled``/``objective_stage`` retain compatibility with existing probe callers;
    # formal training always supplies the manifest contract and explicit expected counts.
    if contract is None:
        if enabled is None or objective_stage is None:
            raise TypeError("gradient_audit_summary requires a contract or enabled/objective_stage.")
        contract = state_gradient_audit_contract(
            argparse.Namespace(audit_state_gradients=enabled, objective_stage=objective_stage)
        )
    elif enabled is not None or objective_stage is not None:
        raise TypeError("Do not combine a manifest contract with legacy enabled/objective_stage arguments.")
    if expected_counts is None:
        expected_counts = Counter({category: len(values) for category, values in accumulator.items()})
    if multi_update_episode_count is None:
        multi_update_episode_count = int(bool(expected_counts.get("first_intermediate_state", 0)))
    enabled = bool(contract["enabled"])
    required = set(contract["required_always"]) if enabled else set()
    if enabled and multi_update_episode_count > 0:
        required.update(contract["required_when_two_or_more_updates"])
    all_categories = sorted(set(accumulator) | set(expected_counts) | required)
    categories = {
        category: {
            "expected": int(expected_counts.get(category, 0)),
            "observed": len(values),
            "positive_finite": len(values),
            "min_norm": min(values) if values else None,
            "max_norm": max(values) if values else None,
            "norm_payload_sha256": (
                hashlib.sha256(
                    json.dumps([value.hex() for value in values], separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                if values
                else None
            ),
        }
        for category in all_categories
        for values in (accumulator.get(category, []),)
    }
    passed = not enabled or (
        bool(required)
        and all(
            categories[category]["expected"] > 0
            and categories[category]["observed"] == categories[category]["expected"]
            and categories[category]["positive_finite"] == categories[category]["expected"]
            for category in required
        )
        and all(item["observed"] == item["expected"] == item["positive_finite"] for item in categories.values())
    )
    return {
        **contract,
        "required_categories": sorted(required),
        "multi_update_episode_count": multi_update_episode_count,
        "categories": categories,
        "passed": passed,
    }


def gradient_payload_sha256(named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> str:
    payload: dict[str, str | None] = {}
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        payload[name] = None if parameter.grad is None else canonical_tensor_sha256(parameter.grad)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def truncate_metrics_for_resume(path: Path, *, optimizer_step: int) -> float:
    if not path.exists():
        return 0.0
    kept: list[dict[str, Any]] = []
    prior_elapsed = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            step = value.get("optimizer_step")
            if not isinstance(step, int):
                raise ValueError(f"Invalid optimizer_step in {path}:{line_number}")
            if step <= optimizer_step:
                kept.append(value)
                if value.get("elapsed_seconds") is not None:
                    prior_elapsed = max(prior_elapsed, float(value["elapsed_seconds"]))
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in kept:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.write(json.dumps({"kind": "resume", "optimizer_step": optimizer_step}) + "\n")
    temporary.replace(path)
    return prior_elapsed


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("DreamLite episode training requires two visible CUDA GPUs.")
    determinism_report: dict[str, Any] | None = None
    if args.strict_determinism:
        determinism_report = configure_strict_cuda_determinism(args.seed)
    positive_values = {
        "epochs": args.epochs,
        "gradient_accumulation": args.gradient_accumulation,
        "gradient_clip": args.gradient_clip,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "eval_start_step": args.eval_start_step,
        "eval_limit": args.eval_limit,
        "early_stopping_patience": args.early_stopping_patience,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise SystemExit(f"Training arguments must be positive: {invalid}")
    if args.max_train_episodes is not None and args.max_train_episodes <= 0:
        raise SystemExit("--max-train-episodes must be positive when supplied.")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps <= 0:
        raise SystemExit("--max-optimizer-steps must be positive when supplied.")
    if args.resolution != 1024:
        raise SystemExit("Formal DreamLite R3 requires --resolution 1024 for the locked Reader resize contract.")
    if args.resume is not None and args.initialize_from is not None:
        raise SystemExit("--resume and --initialize-from are mutually exclusive.")
    for field in ("presentations_per_state", "distill_presentations", "qa_presentations"):
        value = getattr(args, field)
        if value is not None and value < 0:
            raise SystemExit(f"--{field.replace('_', '-')} cannot be negative.")
    if args.presentations_per_state is not None and args.presentations_per_state != args.epochs:
        raise SystemExit("One R3 epoch is one presentation/state; --presentations-per-state must equal --epochs.")
    current_stage_presentations = (
        args.distill_presentations if args.objective_stage == "distill" else args.qa_presentations
    )
    if current_stage_presentations not in {0, args.epochs}:
        raise SystemExit("The current stage presentation count must be zero (derived) or exactly --epochs.")
    if args.reader_loss_mode != "listwise-choice" and (
        args.training_regime != "qa_only" or args.objective_stage != "qa"
    ):
        raise SystemExit("target-only is legacy-R1 diagnostics only; R3 teacher lineage requires listwise-choice.")
    if args.initial_state_mode == "file" and args.source_image is None:
        raise SystemExit("--initial-state-mode file requires --source-image.")
    if args.initial_state_mode != "file" and args.source_image is not None:
        raise SystemExit("--source-image is accepted only with --initial-state-mode file.")
    if torch.device(args.dreamlite_device) == torch.device(args.reader_device):
        raise SystemExit("DreamLite and Reader must use distinct CUDA devices for the formal run.")

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise SystemExit("A fresh run refuses a non-empty --output-dir; use --resume or choose a new directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(args)
    manifest["strict_determinism"] = determinism_report
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    installed = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    (args.output_dir / "environment.txt").write_text("\n".join(installed) + "\n", encoding="utf-8")

    if args.dataset_format == "synthetic":
        raw_train_records: list[Any] = read_episode_jsonl(args.train)
        dev_records: list[Any] = read_episode_jsonl(args.dev)[: args.eval_limit]
    elif args.dataset_format == "prefeval-export":
        raw_train_records = read_prefeval_adapted_jsonl(args.train, allowed_splits={"adapt_train"})
        dev_records = read_prefeval_adapted_jsonl(args.dev, allowed_splits={"adapt_dev"})[: args.eval_limit]
    else:
        raw_train_records = read_prefeval_supervised_jsonl(args.train, allowed_splits={"adapt_train"})
        dev_records = read_prefeval_supervised_jsonl(args.dev, allowed_splits={"adapt_dev"})[: args.eval_limit]
    train_records, curriculum_audit = select_curriculum_episodes(
        raw_train_records,
        curriculum=args.curriculum,
    )
    if args.max_train_episodes is not None:
        train_records = train_records[: args.max_train_episodes]
    if not train_records or not dev_records:
        raise SystemExit("Train and dev datasets must remain non-empty after limits are applied.")
    curriculum_record = {
        **curriculum_audit.to_dict(),
        "selected_after_max_train_episodes": len(train_records),
        "selection_scope": "training_only",
        "turns_rewritten": False,
    }
    (args.output_dir / "curriculum.json").write_text(
        json.dumps(curriculum_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    updater_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    updater_dtype = compute_dtype(updater_device)
    reader_dtype = compute_dtype(reader_device)
    set_all_seeds(args.seed)

    from diffusers import DreamLiteMobilePipeline
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    pipe = DreamLiteMobilePipeline.from_pretrained(
        args.dreamlite,
        local_files_only=True,
        torch_dtype=updater_dtype,
    ).to(updater_device)
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

    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=256 * 256,
        max_pixels=256 * 256,
    )
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("A fast tensor-native Qwen image processor is required.")
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=reader_dtype,
        attn_implementation="sdpa",
    ).to(reader_device)
    freeze_module(reader)
    reader.config.use_cache = False

    source_pil, _source_metadata = load_initial_image(
        args.initial_state_mode,
        args.source_image,
        resolution=args.resolution,
    )
    source_image = pipe.image_processor.preprocess(source_pil, height=args.resolution, width=args.resolution)
    with torch.no_grad():
        initial_state = pipe.prepare_image_latents(source_image, dtype=updater_dtype, device=updater_device)
    model = DreamLiteEpisodeModel(
        pipeline=pipe,
        initial_state=initial_state,
        global_seed=args.seed,
        checkpoint_unet=args.checkpoint_unet,
        learn_initial_state=args.learn_initial_state,
    )
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    allowed_trainable = {
        name
        for name in trainable_names
        if ".lora_A." in name or ".lora_B." in name or (args.learn_initial_state and name == "initial_state")
    }
    unexpected_trainable = sorted(trainable_names - allowed_trainable)
    if unexpected_trainable:
        raise RuntimeError(
            f"Unexpected trainable parameters outside the LoRA/initial-state whitelist: {unexpected_trainable}"
        )
    named_trainable = [(name, parameter) for name, parameter in model.named_parameters() if name in allowed_trainable]
    trainable = [parameter for _name, parameter in named_trainable]
    if not trainable:
        raise RuntimeError("No trainable LoRA/initial-state parameters were found.")
    if args.initialize_from is not None:
        parent_payload = load_trainable_weights(args.initialize_from, trainable_module=model)
        parent_lineage = parent_payload["manifest"]["training_lineage"]
        if args.training_regime == "qa_only" and parent_lineage.get("training_regime") != "qa_only":
            raise RuntimeError("Refusing to load teacher-lineage weights into a qa_only model.")
        if args.training_regime == "teacher_assisted" and parent_lineage.get("objective_stage") != "distill":
            raise RuntimeError("Teacher-assisted QA must initialize from a distill-only checkpoint.")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 0
    start_cursor = 0
    optimizer_step = 0
    best_dev = float("inf")
    stale_evals = 0
    if args.resume:
        payload = load_training_checkpoint(
            args.resume,
            trainable_module=model,
            optimizer=optimizer,
            expected_manifest=manifest,
        )
        start_epoch = int(payload["epoch"])
        start_cursor = int(payload["episode_cursor"])
        optimizer_step = int(payload["optimizer_step"])
        saved_trainer_state = payload.get("trainer_state", {})
        best_dev = float(saved_trainer_state.get("best_dev", float("inf")))
        stale_evals = int(saved_trainer_state.get("stale_evals", 0))
    if args.max_optimizer_steps is not None and optimizer_step >= args.max_optimizer_steps:
        raise RuntimeError("Resume checkpoint already reached or exceeded --max-optimizer-steps.")

    gradient_audit_tensors: list[tuple[str, torch.Tensor]] = []
    gradient_audit_accumulator: dict[str, list[float]] = {}
    gradient_audit_expected: Counter[str] = Counter()
    gradient_audit_multi_update_episodes = 0
    gradient_audit_contract = manifest["state_gradient_audit_contract"]
    train_target_reader = target_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=reader_device,
        require_grad=True,
        deterministic_ce=args.strict_determinism,
        gradient_audit_tensors=gradient_audit_tensors if args.audit_state_gradients else None,
    )
    eval_target_reader = target_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=reader_device,
        require_grad=False,
        deterministic_ce=args.strict_determinism,
    )
    train_choice_reader = choice_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=reader_device,
        require_grad=True,
        deterministic_ce=args.strict_determinism,
        gradient_audit_tensors=gradient_audit_tensors if args.audit_state_gradients else None,
    )
    eval_choice_reader = choice_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=reader_device,
        require_grad=False,
        deterministic_ce=args.strict_determinism,
    )
    state_supervision_fn = build_state_supervision_provider(
        args=args,
        model=model,
        pipe=pipe,
        reader=reader,
        processor=processor,
        expected_lineage=manifest["training_lineage"],
        gradient_audit_tensors=gradient_audit_tensors if args.audit_state_gradients else None,
    )
    if args.training_regime == "qa_only" and state_supervision_fn is not None:
        raise RuntimeError("qa_only provider contract violation: a state supervision provider was configured.")
    if args.objective_stage == "qa" and state_supervision_fn is not None:
        raise RuntimeError("QA objective must unload the teacher provider before training begins.")
    if (
        args.training_regime == "teacher_assisted"
        and args.objective_stage == "distill"
        and state_supervision_fn is None
    ):
        raise RuntimeError("teacher_assisted distillation requires a configured, locked state supervision provider.")
    distill_initial: dict[str, float] | None = None
    if state_supervision_fn is not None:
        distill_initial = evaluate_distillation(
            model=model,
            records=train_records,
            recurrence_mode=args.recurrence_mode,
            detach_between_events=args.detach_between_events,
            noop_policy=args.noop_policy,
            state_supervision_fn=state_supervision_fn,
            require_mixed_delayed_probe=args.require_mixed_delayed_probe,
        )
    metrics_path = args.output_dir / "metrics.jsonl"
    prior_elapsed = truncate_metrics_for_resume(metrics_path, optimizer_step=optimizer_step) if args.resume else 0.0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(updater_device)
    torch.cuda.reset_peak_memory_stats(reader_device)

    final_epoch = start_epoch
    final_cursor = start_cursor
    budget_reached = False
    epoch_iterator = (
        range(start_epoch, args.epochs)
        if args.disable_early_stopping or stale_evals < args.early_stopping_patience
        else ()
    )
    for epoch in epoch_iterator:
        final_epoch = epoch
        order = episode_order(train_records, args.seed, epoch)
        cursor = start_cursor if epoch == start_epoch else 0
        accumulation_count = 0
        accumulation_loss_sum = 0.0
        accumulation_qa_loss_sum = 0.0
        accumulation_state_loss_sum = 0.0
        accumulation_latent_loss_sum = 0.0
        accumulation_image_loss_sum = 0.0
        accumulation_feature_loss_sum = 0.0
        accumulation_choice_rotation_counts = [0, 0, 0, 0]
        accumulation_gradient_audit: dict[str, list[float]] = {}
        accumulation_gradient_expected: Counter[str] = Counter()
        accumulation_multi_update_episodes = 0
        for position in range(cursor, len(order)):
            gradient_audit_tensors.clear()
            final_cursor = position + 1
            episode = train_records[order[position]]
            choice_rotation = choice_rotation_for_training(
                args.choice_view_schedule,
                epoch=epoch,
                position=position,
                episodes_per_epoch=len(order),
                schedule_key=episode_choice_schedule_key(episode),
            )
            episode_kwargs: dict[str, Any] = {
                "episode": episode,
                "initial_state": model.reset_state(),
                "update_fn": model.updater,
                "decode_fn": model.updater.decode_for_reader,
                "reencode_fn": model.updater.reencode_posterior_mean,
                "reencode_decode_fn": model.updater.decode_for_reencode,
                "reader_loss_mode": args.reader_loss_mode,
                "recurrence_mode": args.recurrence_mode,
                "detach_between_events": args.detach_between_events,
                "collect_states": args.audit_state_gradients,
                "noop_policy": args.noop_policy,
                "choice_view_fn": choice_view_for_rotation(args.choice_view_schedule, choice_rotation),
                "training_regime": args.training_regime,
                "state_supervision_fn": state_supervision_fn,
                "objective_stage": args.objective_stage,
                "require_mixed_delayed_probe": args.require_mixed_delayed_probe,
            }
            if args.objective_stage == "qa":
                if args.reader_loss_mode == "listwise-choice":
                    episode_kwargs["choice_reader_loss_fn"] = train_choice_reader
                else:
                    episode_kwargs["reader_loss_fn"] = train_target_reader
            else:
                episode_kwargs["choice_view_fn"] = None
            result = run_episode(**episode_kwargs)
            if args.audit_state_gradients:
                retain_gradient_audit_tensor(
                    gradient_audit_tensors,
                    category="final_state",
                    tensor=result.final_state,
                )
                if len(result.states) >= 2:
                    retain_gradient_audit_tensor(
                        gradient_audit_tensors,
                        category="first_intermediate_state",
                        tensor=result.states[0],
                    )
                    gradient_audit_multi_update_episodes += 1
                    accumulation_multi_update_episodes += 1
                category_counts = Counter(category for category, _tensor in gradient_audit_tensors)
                if args.objective_stage == "qa" and category_counts["query_image"] != result.query_count:
                    raise RuntimeError("R3 micro query-image audit count differs from the episode query count.")
                if args.objective_stage == "distill":
                    if category_counts["state_image"] != len(result.states):
                        raise RuntimeError("R3 micro state-image audit count differs from the updater-state count.")
                    if category_counts["student_visual_feature"] != len(result.states):
                        raise RuntimeError("R3 micro visual-feature audit count differs from the updater-state count.")
                expected_categories = set(gradient_audit_contract["required_always"])
                if len(result.states) >= 2:
                    expected_categories.update(gradient_audit_contract["required_when_two_or_more_updates"])
                if set(category_counts) != expected_categories:
                    raise RuntimeError(
                        "R3 micro gradient-audit categories differ from the manifest contract: "
                        f"expected={sorted(expected_categories)}, observed={sorted(category_counts)}."
                    )
                gradient_audit_expected.update(category_counts)
                accumulation_gradient_expected.update(category_counts)
            (result.loss / args.gradient_accumulation).backward()
            if args.audit_state_gradients:
                observed = audit_episode_gradients(gradient_audit_tensors, gradient_audit_accumulator)
                for category, values in observed.items():
                    accumulation_gradient_audit.setdefault(category, []).extend(values)
            assert_frozen_contract(pipe, reader)
            accumulation_count += 1
            accumulation_loss_sum += float(result.loss.detach().item())
            accumulation_qa_loss_sum += 0.0 if result.qa_loss is None else float(result.qa_loss.detach().item())
            accumulation_state_loss_sum += (
                0.0 if result.state_supervision_loss is None else float(result.state_supervision_loss.detach().item())
            )
            accumulation_latent_loss_sum += (
                0.0 if result.latent_distill_loss is None else float(result.latent_distill_loss.detach().item())
            )
            accumulation_image_loss_sum += (
                0.0 if result.image_distill_loss is None else float(result.image_distill_loss.detach().item())
            )
            accumulation_feature_loss_sum += (
                0.0
                if result.visual_feature_distill_loss is None
                else float(result.visual_feature_distill_loss.detach().item())
            )
            accumulation_choice_rotation_counts[choice_rotation] += 1

            is_last = position + 1 == len(order)
            if accumulation_count == args.gradient_accumulation or is_last:
                # Every optimizer step represents the mean of its actual episode group.
                # A final partial group was divided by the nominal accumulation size
                # above, so restore the correct 1/actual_count scale before clipping.
                accumulation_rescale = args.gradient_accumulation / accumulation_count
                if accumulation_rescale != 1.0:
                    for parameter in trainable:
                        if parameter.grad is not None:
                            parameter.grad.mul_(accumulation_rescale)
                raw_gradient_sha256 = gradient_payload_sha256(named_trainable) if args.audit_gradient_sha else None
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
                if not torch.isfinite(gradient_norm) or gradient_norm.item() <= 0:
                    raise RuntimeError(f"Invalid trainable gradient norm: {gradient_norm.item()}")
                clipped_gradient_sha256 = gradient_payload_sha256(named_trainable) if args.audit_gradient_sha else None
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                group_episode_count = accumulation_count
                group_mean_loss = accumulation_loss_sum / group_episode_count
                group_mean_qa_loss = accumulation_qa_loss_sum / group_episode_count
                group_mean_state_loss = (
                    accumulation_state_loss_sum / group_episode_count if args.objective_stage == "distill" else None
                )
                group_mean_latent_loss = (
                    accumulation_latent_loss_sum / group_episode_count if args.objective_stage == "distill" else None
                )
                group_mean_image_loss = (
                    accumulation_image_loss_sum / group_episode_count if args.objective_stage == "distill" else None
                )
                group_mean_feature_loss = (
                    accumulation_feature_loss_sum / group_episode_count if args.objective_stage == "distill" else None
                )
                group_choice_rotation_counts = tuple(accumulation_choice_rotation_counts)
                group_gradient_audit = gradient_audit_summary(
                    accumulation_gradient_audit,
                    accumulation_gradient_expected,
                    contract=gradient_audit_contract,
                    multi_update_episode_count=accumulation_multi_update_episodes,
                )
                accumulation_count = 0
                accumulation_loss_sum = 0.0
                accumulation_qa_loss_sum = 0.0
                accumulation_state_loss_sum = 0.0
                accumulation_latent_loss_sum = 0.0
                accumulation_image_loss_sum = 0.0
                accumulation_feature_loss_sum = 0.0
                accumulation_choice_rotation_counts = [0, 0, 0, 0]
                accumulation_gradient_audit = {}
                accumulation_gradient_expected = Counter()
                accumulation_multi_update_episodes = 0
                append_jsonl(
                    metrics_path,
                    {
                        "kind": "train",
                        "epoch": epoch,
                        "episode_cursor": position + 1,
                        "optimizer_step": optimizer_step,
                        "loss": group_mean_loss,
                        "qa_loss": group_mean_qa_loss,
                        "state_supervision_loss": group_mean_state_loss,
                        "latent_distill_loss": group_mean_latent_loss,
                        "image_distill_loss": group_mean_image_loss,
                        "visual_feature_distill_loss": group_mean_feature_loss,
                        "training_regime": args.training_regime,
                        "objective_stage": args.objective_stage,
                        "reader_loss_mode": args.reader_loss_mode,
                        "choice_rotation_counts": group_choice_rotation_counts,
                        "group_episode_count": group_episode_count,
                        "gradient_norm": float(gradient_norm.item()),
                        "loss_hex": float(group_mean_loss).hex(),
                        "gradient_norm_hex": float(gradient_norm.item()).hex(),
                        "raw_gradient_sha256": raw_gradient_sha256,
                        "clipped_gradient_sha256": clipped_gradient_sha256,
                        "state_gradient_audit": group_gradient_audit,
                        "elapsed_seconds": prior_elapsed + time.monotonic() - started,
                    },
                )

                stop_after_step = False
                if (
                    args.objective_stage == "qa"
                    and optimizer_step >= args.eval_start_step
                    and (optimizer_step - args.eval_start_step) % args.eval_every == 0
                ):
                    dev_loss = evaluate_dev(
                        model=model,
                        records=dev_records,
                        recurrence_mode=args.recurrence_mode,
                        detach_between_events=args.detach_between_events,
                        noop_policy=args.noop_policy,
                        reader_loss_mode=args.reader_loss_mode,
                        reader_loss_fn=eval_target_reader,
                        choice_reader_loss_fn=eval_choice_reader,
                        choice_view_schedule=args.choice_view_schedule,
                        require_mixed_delayed_probe=args.require_mixed_delayed_probe,
                    )
                    append_jsonl(metrics_path, {"kind": "dev", "optimizer_step": optimizer_step, "loss": dev_loss})
                    if dev_loss < best_dev:
                        best_dev = dev_loss
                        stale_evals = 0
                        save_training_checkpoint(
                            args.output_dir / "best.pt",
                            trainable_module=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            episode_cursor=position + 1,
                            optimizer_step=optimizer_step,
                            manifest=manifest,
                            trainer_state={"best_dev": best_dev, "stale_evals": stale_evals},
                        )
                    else:
                        stale_evals += 1
                    stop_after_step = not args.disable_early_stopping and stale_evals >= args.early_stopping_patience

                if optimizer_step % args.checkpoint_every == 0:
                    save_training_checkpoint(
                        args.output_dir / f"checkpoint-{optimizer_step:06d}.pt",
                        trainable_module=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        episode_cursor=position + 1,
                        optimizer_step=optimizer_step,
                        manifest=manifest,
                        trainer_state={"best_dev": best_dev, "stale_evals": stale_evals},
                    )
                if args.max_optimizer_steps is not None and optimizer_step >= args.max_optimizer_steps:
                    budget_reached = True
                    stop_after_step = True
                if stop_after_step:
                    break
        start_cursor = 0
        if not args.disable_early_stopping and stale_evals >= args.early_stopping_patience:
            break
        if budget_reached:
            break
        final_cursor = len(order)

    state_gradient_audit = gradient_audit_summary(
        gradient_audit_accumulator,
        gradient_audit_expected,
        contract=gradient_audit_contract,
        multi_update_episode_count=gradient_audit_multi_update_episodes,
    )
    (args.output_dir / "state_gradient_audit.json").write_text(
        json.dumps(state_gradient_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.audit_state_gradients and not state_gradient_audit["passed"]:
        raise RuntimeError("R3 state-gradient audit did not satisfy its locked manifest contract.")

    save_training_checkpoint(
        args.output_dir / "last.pt",
        trainable_module=model,
        optimizer=optimizer,
        epoch=final_epoch,
        episode_cursor=final_cursor,
        optimizer_step=optimizer_step,
        manifest=manifest,
        trainer_state={"best_dev": best_dev, "stale_evals": stale_evals},
    )
    distill_final: dict[str, float] | None = None
    distill_diagnostics: dict[str, Any] | None = None
    if state_supervision_fn is not None:
        assert distill_initial is not None
        distill_final = evaluate_distillation(
            model=model,
            records=train_records,
            recurrence_mode=args.recurrence_mode,
            detach_between_events=args.detach_between_events,
            noop_policy=args.noop_policy,
            state_supervision_fn=state_supervision_fn,
            require_mixed_delayed_probe=args.require_mixed_delayed_probe,
        )
        if any(value <= 0.0 for value in distill_initial.values()):
            raise RuntimeError("Initial distillation diagnostics must be strictly positive.")
        relative = {
            name: distill_final[name] / distill_initial[name]
            for name in ("distill_loss", "latent_raw", "image_raw", "feature_raw")
        }
        distill_diagnostics = {
            "initial": distill_initial,
            "final": distill_final,
            "final_over_initial": relative,
            "checks": {
                "composite_drop_at_least_50_percent": relative["distill_loss"] <= 0.5,
                "all_raw_components_decreased": all(
                    distill_final[name] < distill_initial[name] for name in ("latent_raw", "image_raw", "feature_raw")
                ),
            },
        }
        (args.output_dir / "distill_diagnostics.json").write_text(
            json.dumps(distill_diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "optimizer_steps": optimizer_step,
        "best_dev_loss": None if best_dev == float("inf") else best_dev,
        "training_regime": args.training_regime,
        "objective_stage": args.objective_stage,
        "reader_loss_mode": args.reader_loss_mode,
        "choice_view_schedule": args.choice_view_schedule,
        "teacher_manifest_sha256": manifest["training_lineage"]["teacher_manifest_sha256"],
        "teacher_control": manifest["training_lineage"]["teacher_control"],
        "teacher_control_sha256": manifest["training_lineage"]["teacher_control_sha256"],
        "state_gradient_audit": state_gradient_audit,
        "elapsed_seconds": prior_elapsed + time.monotonic() - started,
        "peak_vram_gib": {
            str(updater_device): torch.cuda.max_memory_allocated(updater_device) / 2**30,
            str(reader_device): torch.cuda.max_memory_allocated(reader_device) / 2**30,
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "distill_diagnostics": distill_diagnostics,
        "strict_determinism": determinism_report,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
