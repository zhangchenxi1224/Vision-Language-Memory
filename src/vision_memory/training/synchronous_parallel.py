"""Explicit synchronous data parallelism without changing the upstream U-Net.

Each global microbatch keeps its original draw index. Gradients of loss/global
batch are summed across ranks before clipping and AdamW. Native inference stays
batch one and is sharded by complete condition group.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def microbatch_indices(global_batch, world_size, rank):
    if global_batch < world_size or global_batch % world_size or not 0 <= rank < world_size:
        raise ValueError("Global microbatch count must be divisible by the number of ranks")
    return range(rank, global_batch, world_size)


def reduce_gradients(parameters, *, bucket_bytes=32 * 1024 * 1024):
    """SUM already globally scaled gradients; never divide a second time."""
    bucket, size = [], 0
    def flush():
        nonlocal bucket, size
        if not bucket:
            return
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for parameter in bucket:
            count = parameter.numel()
            parameter.grad.copy_(flat[offset:offset + count].view_as(parameter))
            offset += count
        bucket, size = [], 0
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("Every full U-Net parameter must participate on every rank")
        if bucket and size + parameter.numel() * parameter.element_size() > bucket_bytes:
            flush()
        bucket.append(parameter)
        size += parameter.numel() * parameter.element_size()
    flush()


class ParallelExecution:
    def __init__(self, args):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.enabled = bool(getattr(args, "data_parallel", False))
        if self.enabled != (self.world > 1):
            raise ValueError("Multi-process launch requires explicit --data-parallel and world size > 1")
        self.root = self.rank == 0
        if self.enabled:
            if os.environ.get("NCCL_ALGO") != "Ring" or os.environ.get("NCCL_PROTO") != "Simple":
                raise ValueError("Parallel training requires fixed NCCL_ALGO=Ring and NCCL_PROTO=Simple")
            if args.model_variant != "base" or args.trainable_scope != "full_unet" or not args.colocate_models:
                raise ValueError("Synchronous parallel training requires colocated official Base/full U-Net")
            microbatch_indices(args.gradient_accumulation_steps, self.world, self.rank)
            if self.world > torch.cuda.device_count():
                raise ValueError("One local CUDA device per rank is required")
            torch.cuda.set_device(self.local_rank)
            args.dreamlite_device = args.reader_device = f"cuda:{self.local_rank}"
            dist.init_process_group("nccl", timeout=timedelta(minutes=30))

    def gather(self, value):
        if not self.enabled:
            return [value]
        values = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def agree(self, value, label):
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        if len(set(self.gather(digest))) != 1:
            raise RuntimeError(f"Ranks disagree about {label}")

    def any_stop(self, local_stop):
        return any(self.gather(bool(local_stop)))

    def check_parameters(self, module):
        if not self.enabled:
            return
        digest = hashlib.sha256()
        for name, parameter in module.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        values = self.gather(digest.hexdigest())
        if len(set(values)) != 1:
            raise RuntimeError("Synchronized U-Net parameters differ between ranks")
        devices = self.gather({"rank": self.rank, "local_rank": self.local_rank, "pid": os.getpid(),
                               "device": str(torch.cuda.current_device()), "name": torch.cuda.get_device_name()})
        return {"world_size": self.world, "parameter_sha256_by_rank": values, "bitwise_rank_agreement": True,
                "devices": devices, "backend": "nccl", "nccl_algorithm": "Ring", "nccl_protocol": "Simple"}

    def evaluate(self, args, runtime, bank, teachers, phase, training):
        if not self.enabled:
            return training.evaluate(args, runtime, bank, teachers, phase)
        shard_args = copy.copy(args)
        shard_args.output_dir = args.output_dir / "parallel-shards" / f"rank-{self.rank:02d}"
        shard_bank = {**bank, "groups": bank["groups"][self.rank::self.world]}
        training.evaluate(shard_args, runtime, shard_bank, teachers, phase)
        self.barrier()
        if self.root:
            merge_evaluation(args, bank, phase, self.world, training)
        self.barrier()
        complete = json.loads((args.output_dir / phase / "complete.json").read_text())
        return complete["summary"]

    def teacher_readback(self, args, runtime, groups, teachers, training):
        if not self.enabled:
            return training.verify_training_teacher_readback(args, runtime, groups, teachers)
        shard_args = copy.copy(args)
        shard_args.output_dir = args.output_dir / "parallel-shards" / f"rank-{self.rank:02d}"
        training.verify_training_teacher_readback(shard_args, runtime, groups[self.rank::self.world], teachers)
        self.barrier()
        if self.root:
            rows = []
            for rank in range(self.world):
                path = args.output_dir / "parallel-shards" / f"rank-{rank:02d}" / "teacher-readback.json"
                rows.extend(json.loads(path.read_text())["rows"])
            order = {group["question_id"]: i for i, group in enumerate(groups)}
            rows.sort(key=lambda row: (order[row["question_id"]], row["teacher_id"]))
            training.write_json(args.output_dir / "teacher-readback.json",
                {"scope": "direct oracle positive control, not Writer output", "rows": rows})
        self.barrier()


def merge_evaluation(args, bank, phase, world_size, training):
    """Verify each sealed shard and the full cell matrix before sealing a phase."""
    directory = args.output_dir / phase
    directory.mkdir(parents=True, exist_ok=True)
    rows, geometry, tensor_names = [], {}, set()
    for rank in range(world_size):
        shard = args.output_dir / "parallel-shards" / f"rank-{rank:02d}" / phase
        seal = json.loads((shard / "complete.json").read_text())
        for name, expected in seal["artifact_hashes"].items():
            if Path(name).name != name or training.file_sha256(shard / name) != expected:
                raise RuntimeError("Parallel evaluation shard changed")
            if name.endswith(".pt"):
                if name in tensor_names:
                    raise RuntimeError("Duplicate generated tensor between ranks")
                tensor_names.add(name)
                destination = directory / name
                if destination.exists():
                    if training.file_sha256(destination) != expected:
                        raise RuntimeError("Previously merged evaluation tensor differs")
                else:
                    os.link(shard / name, destination)
        shard_rows = [json.loads(line) for line in (shard / "generations.jsonl").read_text().splitlines()]
        assigned = {group["question_id"] for group in bank["groups"][rank::world_size]}
        if any(row["question_id"] not in assigned or row["phase"] != phase for row in shard_rows):
            raise RuntimeError("Evaluation rank produced another rank's group")
        rows.extend(shard_rows)
        geometry.update(seal["summary"]["geometry"])
    expected = set()
    order = {}
    for group in bank["groups"]:
        for kind, seed in [("blank", None), ("donor", None)] + [
            ("matched", training.stable_seed(args.seed, "heldout-evaluation-noise", i)) for i in range(args.eval_seeds)]:
            for prompt_id in group["question_variants"]:
                key = (group["question_id"], kind, seed, prompt_id)
                expected.add(key)
                order[key] = len(order)
    key = lambda row: (row["question_id"], row["condition"], row["noise_seed"], row["prompt_id"])
    if len(rows) != len(expected) or {key(row) for row in rows} != expected:
        raise RuntimeError("Parallel evaluation has missing or duplicate cells")
    if len(tensor_names) != len(bank["groups"]) * args.eval_seeds or set(geometry) != {g["question_id"] for g in bank["groups"]}:
        raise RuntimeError("Parallel evaluation tensor/geometry coverage differs")
    rows.sort(key=lambda row: order[key(row)])
    temporary = directory / "generations.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    temporary.replace(directory / "generations.jsonl")
    summary = training.summarize_evaluation(rows, geometry)
    training.write_json(directory / "summary.json", summary)
    training.write_json(directory / "complete.json", {"summary": summary, "generation_rows": len(rows),
        "artifact_hashes": {name: training.file_sha256(directory / name)
                            for name in sorted(tensor_names | {"generations.jsonl", "summary.json"})}})
