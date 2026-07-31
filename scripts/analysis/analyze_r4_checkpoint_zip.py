"""R4 checkpoint analyzer for Python environments with NumPy but without PyTorch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import struct
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


class StorageRef:
    def __init__(self, key: str, dtype: str, numel: int) -> None:
        self.key = key
        self.dtype = dtype
        self.numel = numel


class Dummy:
    def __setstate__(self, state: Any) -> None:
        return None


class TensorMeta:
    def __init__(self, storage: StorageRef, offset: int, size: tuple[int, ...], stride: tuple[int, ...]) -> None:
        self.storage = storage
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)


class Unpickler(pickle.Unpickler):
    def persistent_load(self, pid: Any) -> StorageRef:
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise ValueError(f"unsupported persistent id: {pid!r}")
        return StorageRef(pid[2], getattr(pid[1], "__name__", str(pid[1])), int(pid[4]))

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch" and name.endswith("Storage"):
            return type(name, (), {})
        if module == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return lambda storage, offset, size, stride, *args: TensorMeta(
                storage, int(offset), tuple(size), tuple(stride)
            )
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda tensor, requires_grad, hooks: tensor
        if module == "torch.torch_version" and name == "TorchVersion":
            return str
        if module.startswith("torch"):
            raise RuntimeError(f"unsupported torch pickle global: {module}.{name}")
        return super().find_class(module, name)


def _root_name(names: list[str]) -> str:
    for name in names:
        if name.endswith("/data.pkl"):
            return name[: -len("/data.pkl")]
    raise ValueError("checkpoint has no data.pkl")


def _decode_storage(raw: bytes, dtype: str, numel: int) -> list[float]:
    if dtype.endswith("FloatStorage"):
        return list(struct.unpack("<" + "f" * numel, raw[: 4 * numel]))
    if dtype.endswith("DoubleStorage"):
        return list(struct.unpack("<" + "d" * numel, raw[: 8 * numel]))
    if dtype.endswith("HalfStorage"):
        return [float(value) for value in struct.unpack("<" + "e" * numel, raw[: 2 * numel])]
    raise ValueError(f"unsupported checkpoint dtype without torch: {dtype}")


def _values(meta: TensorMeta, storages: dict[str, list[float]]) -> list[float]:
    values = storages[meta.storage.key]
    if not meta.size:
        return [values[meta.offset]]
    expected = 1
    for size in meta.size:
        expected *= size
    if meta.stride == tuple(
        reversed(tuple(
            __import__("math").prod(meta.size[index + 1:]) for index in range(len(meta.size))
        ))
    ):
        return values[meta.offset: meta.offset + expected]
    result: list[float] = []
    def visit(dim: int, offset: int) -> None:
        if dim == len(meta.size):
            result.append(values[offset])
            return
        for index in range(meta.size[dim]):
            visit(dim + 1, offset + index * meta.stride[dim])
    visit(0, meta.offset)
    return result


def _load_state(path: Path) -> dict[str, list[float]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        root = _root_name(names)
        payload = Unpickler(io.BytesIO(archive.read(root + "/data.pkl"))).load()
        raw_storages: dict[str, bytes] = {}
        for name in names:
            if name.startswith(root + "/data/"):
                raw_storages[name.rsplit("/", 1)[1]] = archive.read(name)
        storage_values: dict[str, list[float]] = {}
        refs: dict[str, StorageRef] = {}
        for value in payload.get("trainable_state", {}).values():
            if isinstance(value, TensorMeta):
                refs[value.storage.key] = value.storage
        for key, ref in refs.items():
            storage_values[key] = _decode_storage(raw_storages[key], ref.dtype, ref.numel)
        return {
            name: _values(value, storage_values)
            for name, value in payload["trainable_state"].items()
        }


def _group(name: str) -> str:
    projection = next((item for item in ("to_q", "to_k", "to_v", "to_out") if f".{item}." in f".{name}."), "other")
    factor = next((item for item in ("lora_A", "lora_B") if f".{item}." in f".{name}."), "other")
    return f"{projection}|{factor}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _delta(left: list[float], right: list[float]) -> tuple[float, float | None]:
    if len(left) != len(right):
        return float("nan"), None
    delta = [a - b for a, b in zip(left, right)]
    left_norm = _norm(left)
    right_norm = _norm(right)
    cosine = None if left_norm == 0.0 or right_norm == 0.0 else sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return _norm(delta), cosine


def analyze(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    paths = sorted((run_dir / "output" / "checkpoints").glob("step-*.pt"))
    if not paths:
        raise FileNotFoundError("no checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    previous: dict[str, list[float]] | None = None
    for path in paths:
        state = _load_state(path)
        groups: dict[str, list[float]] = defaultdict(list)
        for name, values in state.items():
            groups[_group(name)].extend(values)
        for group, values in sorted(groups.items()):
            change_norm = None
            change_cosine = None
            if previous is not None:
                old_values: list[float] = []
                for name, current in state.items():
                    if _group(name) == group and name in previous:
                        old_values.extend(previous[name])
                change_norm, change_cosine = _delta(values, old_values)
            group_rows.append({
                "checkpoint": path.name,
                "optimizer_step": int(path.stem.split("-")[-1]),
                "group": group,
                "parameter_norm": _norm(values),
                "delta_norm_from_previous": change_norm,
                "delta_cosine_from_previous": change_cosine,
                "elements": len(values),
            })
        summary.append({
            "checkpoint": path.name,
            "optimizer_step": int(path.stem.split("-")[-1]),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "parameter_count": sum(len(values) for values in state.values()),
            "parameter_norm": math.sqrt(sum(value * value for values in state.values() for value in values)),
        })
        previous = state
    with (output_dir / "checkpoint_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "checkpoint_parameter_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    report = {
        "schema": "vision_memory.r4-checkpoint-zip-analysis.v1",
        "checkpoint_count": len(paths),
        "summary": summary,
        "notes": [
            "This parser reads only trainable_state from PyTorch zip checkpoints.",
            "Deltas are relative to the preceding saved checkpoint.",
            "No checkpoint selection is performed.",
        ],
    }
    (output_dir / "checkpoint_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_dir, args.output_dir)
    print(json.dumps({"checkpoint_count": report["checkpoint_count"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

