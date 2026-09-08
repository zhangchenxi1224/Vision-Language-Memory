"""Observe actual 4xH200 runtime; no package install and no scientific outcome."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    from diffusers import DreamLiteMobilePipeline
    from transformers import Qwen3VLForConditionalGeneration
    assert DreamLiteMobilePipeline and Qwen3VLForConditionalGeneration
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"Expected exactly 4 visible GPUs, observed {torch.cuda.device_count()}")
    devices = []
    for i in range(4):
        d = torch.cuda.get_device_properties(i)
        if "H200" not in d.name or d.total_memory < 140000 * 1024 ** 2:
            raise RuntimeError(f"Unexpected accelerator {i}: {d.name}, {d.total_memory}")
        devices.append({"index": i, "name": d.name, "memory_bytes": d.total_memory})
    if shutil.disk_usage(args.output.parent).free < 150 * 1024 ** 3:
        raise RuntimeError("Less than 150 GiB free for complete trajectory artifacts")
    result = {"passed": True, "python": sys.version, "python_executable": sys.executable,
              "torch": torch.__version__, "torch_file": torch.__file__, "cuda_runtime": torch.version.cuda,
              "platform": platform.platform(), "devices": devices,
              "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                                       "--format=csv,noheader"], text=True),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("numpy", "diffusers", "transformers", "safetensors", "accelerate", "peft")},
              "scope": "infrastructure only; real full-chain gradient probe follows",
              "image": "ngc-pytorch:25.02-cuda12.8.0-py3",
              "model_manifest_sha256": {name: os.environ[name] for name in
                ("VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256", "VLM_READER_SNAPSHOT_MANIFEST_SHA256")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
