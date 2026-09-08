"""CPU-only P0/P4 audit of actual Qwen chat terminator and continuation tokens."""
from pathlib import Path
import argparse
import json
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from transformers import AutoProcessor
from vision_memory.reader.open_eos import assistant_termination_contract
from scripts.experiments.run_r11_open_eos_paired import load_config, tokenization_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    generation = json.loads((args.reader / "generation_config.json").read_text())
    model = SimpleNamespace(generation_config=SimpleNamespace(**generation))
    processor = AutoProcessor.from_pretrained(args.reader, local_files_only=True, use_fast=True)
    config = load_config(ROOT / "configs/experiments/r11_open_eos_paired.json")
    result = {"termination": assistant_termination_contract(model, processor),
              "examples": tokenization_audit({"processor": processor}, config),
              "model_forward_calls": 0, "optimizer_steps": 0, "scientific_memory_success": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
