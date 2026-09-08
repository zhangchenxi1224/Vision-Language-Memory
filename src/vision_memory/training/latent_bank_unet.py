"""Empirical-set supervision for a source-anchored DreamLite flow.

No question or answer is a U-Net input. A sampled successful endpoint is a
training target, never an averaged latent. The bridge starts at the *actual*
deployment distribution: (1-s0)*source+s0*Gaussian, with s0=0.5.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from vision_memory.repro import canonical_tensor_sha256

INSTRUCTIONS = "Use the memory image to answer.\nAnswer with a short phrase only."
START_SIGMA = 0.5
EFFECTIVE_SIGMAS = (0.5, 0.375, 0.25, 0.125)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, namespace: str, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{namespace}:{index}".encode()).digest()[:8], "big") % (2**63 - 1)


def anchored_flow_bridge(source: Tensor, noise: Tensor, target: Tensor, sigma: float,
                         start_sigma: float = START_SIGMA) -> tuple[Tensor, Tensor]:
    if source.shape != noise.shape or source.shape != target.shape:
        raise ValueError("Source, noise and target must have the same shape")
    if not 0 < start_sigma <= 1 or not math.isfinite(sigma) or not 0 <= sigma <= start_sigma:
        raise ValueError("Bridge sigma must lie in [0, start_sigma]")
    start = (1 - start_sigma) * source + start_sigma * noise
    velocity = (start - target) / start_sigma
    return target + sigma * velocity, velocity


def predict_velocity(sampler: Any, state: Tensor, source: Tensor, sigma: float,
                     prompt_embeds: Tensor, attention_mask: Tensor) -> Tensor:
    """Use the same spatial concatenation, timestep units and crop as inference."""
    if not 0 <= sigma <= START_SIGMA:
        raise ValueError("Unexpected effective sigma")
    count = int(getattr(sampler.scheduler.config, "num_train_timesteps", 1000))
    timestep = torch.tensor(sigma * count, device=state.device, dtype=state.dtype)
    time_ids = torch.tensor([[state.shape[-1] * sampler.vae_scale_factor,
                              state.shape[-2] * sampler.vae_scale_factor]],
                            device=state.device, dtype=state.dtype)
    prediction = sampler._unet_step(torch.cat((state, source), dim=3), timestep,
                                   prompt_embeds, attention_mask, time_ids)
    if prediction.shape != torch.cat((state, source), dim=3).shape:
        raise RuntimeError("Unexpected DreamLite velocity shape")
    return prediction[..., :state.shape[-1]]


def member_split(teacher_ids: list[str]) -> tuple[list[str], list[str]]:
    """Hash-stable target holdout, only if the question has >=5 distinct targets.

    This is a held-out *target* diagnostic, never a held-out-question claim.
    """
    ordered = sorted(teacher_ids, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    held = ordered[:max(1, len(ordered) // 5)] if len(ordered) >= 5 else []
    return [x for x in ordered if x not in held], held


def balanced_draw(groups: list[dict[str, Any]], seed: int, step: int) -> tuple[dict[str, Any], str, int, float]:
    """Each complete cycle visits every question once; sample a member uniformly."""
    cycle, offset = divmod(step, len(groups))
    order_generator = torch.Generator().manual_seed(stable_seed(seed, "question-order", cycle))
    order = torch.randperm(len(groups), generator=order_generator).tolist()
    group = groups[order[offset]]
    train, _ = member_split(group["teacher_ids"])
    rng = torch.Generator().manual_seed(stable_seed(seed, "bank-member-and-sigma", step))
    teacher_id = train[int(torch.randint(len(train), (), generator=rng))]
    # Exact zero carries no training probability. Keep float32 strictly positive.
    sigma = max(float(torch.rand((), generator=rng)), 1e-6) * START_SIGMA
    return group, teacher_id, stable_seed(seed, "training-noise", step), sigma


def bank_geometry(outputs: Tensor, teachers: Tensor, teacher_ids: list[str],
                  heldout_ids: list[str] | None = None) -> dict[str, Any]:
    """Report distances and nearest-member diversity; these are not semantic clusters."""
    x = outputs.detach().float().cpu().flatten(1)
    bank = teachers.detach().float().cpu().flatten(1)
    if x.shape[1] != bank.shape[1] or bank.shape[0] != len(teacher_ids):
        raise ValueError("Geometry shape mismatch")
    distances = torch.cdist(x.double(), bank.double()) / math.sqrt(bank.shape[1])
    nearest = distances.argmin(1)
    counts = torch.bincount(nearest, minlength=len(teacher_ids))
    probs = counts[counts > 0].double() / len(x)
    pair = torch.pdist(x.double()) / math.sqrt(x.shape[1])
    bank_pair = torch.pdist(bank.double()) / math.sqrt(bank.shape[1])
    radius = float(bank_pair.median() * .25) if bank_pair.numel() else 0.0
    coverage = distances.min(0).values <= radius
    held = [teacher_ids.index(t) for t in heldout_ids or []]
    return {
        "generated_count": len(x), "bank_count": len(bank),
        "nearest_bank_rms": distances.min(1).values.tolist(),
        "nearest_teacher_ids": [teacher_ids[i] for i in nearest.tolist()],
        "nearest_member_counts": dict(zip(teacher_ids, counts.tolist())),
        "nearest_member_fraction_visited": float((counts > 0).double().mean()),
        "nearest_member_entropy": float(-(probs * probs.log()).sum()),
        "generated_pairwise_rms_mean": float(pair.mean()) if pair.numel() else None,
        "bank_pairwise_rms_mean": float(bank_pair.mean()) if bank_pair.numel() else None,
        "coverage_radius_rms": radius,
        "coverage_radius_definition": "0.25 * median pairwise bank RMS, fixed before inspecting generated outputs",
        "bank_fraction_within_radius": float(coverage.double().mean()),
        "heldout_target_count": len(held),
        "heldout_bank_fraction_within_radius": float(coverage[held].double().mean()) if held else None,
        "interpretation": "Finite-sample coordinate diagnostics, not established semantic modes or intrinsic dimension",
    }


def load_teacher_bank(path: Path) -> tuple[dict[str, Any], dict[str, Tensor]]:
    bank = json.loads(path.read_text(encoding="utf-8"))
    if bank.get("schema") != "latent-teacher-bank/v1" or bank.get("bank_status") != "sealed":
        raise ValueError("U-Net training requires a sealed latent-teacher-bank/v1")
    if bank.get("route") not in {"direct", "frozen"}:
        raise ValueError("Direct and Frozen banks must be separate arms")
    teachers = bank.get("teachers", [])
    groups = bank.get("groups", [])
    if not teachers or not groups:
        raise ValueError("A bank with no verified successful targets cannot train a Writer")
    tensors: dict[str, Tensor] = {}
    records = {}
    for member in teachers:
        tid = member["teacher_id"]
        if tid in tensors:
            raise ValueError("Duplicate teacher ID")
        if member.get("endpoint_step") != 256 or not member.get("target", {}).get("strict_correct"):
            raise ValueError("Only independently validated raw step256 EOS endpoints may supervise U-Net")
        if (member.get("gold_eos_appended") is not True
                or member.get("evaluation_generation") != {"do_sample": False, "max_new_tokens": 32}):
            raise ValueError("An MCQ, no-EOS or altered-decoding teacher cannot enter the bank")
        payload_path = Path(member["latent_path"])
        if not payload_path.is_absolute():
            payload_path = path.parent / payload_path
        if file_sha256(payload_path) != member["latent_file_sha256"]:
            raise ValueError(f"Teacher file changed: {tid}")
        tensor = torch.load(payload_path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, Tensor) or tensor.dtype != torch.float32 or tuple(tensor.shape) != (1, 4, 128, 128):
            raise ValueError("Teacher must be a raw FP32 model-space [1,4,128,128] tensor")
        if not torch.isfinite(tensor).all() or canonical_tensor_sha256(tensor) != member["latent_sha256"]:
            raise ValueError(f"Teacher tensor changed: {tid}")
        tensors[tid], records[tid] = tensor, member
    seen = []
    for group in groups:
        if not isinstance(group.get("event_text"), str) or not group["event_text"].strip():
            raise ValueError("Every group needs the original event-only condition")
        prompts = group.get("question_variants", {})
        if len(prompts) != 5 or "original_open" not in prompts or len(set(prompts.values())) != 5:
            raise ValueError("Require original plus four distinct question-only paraphrases")
        for query in prompts.values():
            if not query.endswith("\n" + INSTRUCTIONS) or "\n" in query[:-len(INSTRUCTIONS)-1]:
                raise ValueError("Question must be one line, followed by the exact two instruction lines")
        ids = group["teacher_ids"]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Each question must have distinct successful members")
        for tid in ids:
            if tid not in records or records[tid]["question_id"] != group["question_id"] or records[tid]["answer"] != group["answer"]:
                raise ValueError("Teacher does not belong to this question/answer")
        seen.extend(ids)
    if sorted(seen) != sorted(tensors) or len({g["question_id"] for g in groups}) != len(groups):
        raise ValueError("Every teacher must belong to exactly one distinct group")
    return bank, tensors
