from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import read_jsonl  # noqa: E402
from vision_memory.dreamlite import assert_no_frozen_parameter_grads, freeze_module  # noqa: E402
from vision_memory.lightweight import LightweightVisualUpdater  # noqa: E402
from vision_memory.reader import qwen3vl_choice_nll, qwen3vl_target_only_ce  # noqa: E402
from vision_memory.training import format_mcq_query, run_episode  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(episodes, updater, reader, processor, device: torch.device) -> float:
    correct = 0
    total = 0
    updater.eval()
    with torch.no_grad():
        for episode in episodes:
            state = updater.initial_state(batch_size=1, device=device, dtype=torch.float32)
            for turn in episode.turns:
                if turn.calls_updater:
                    state = updater.update(state, turn.event_text)
                if turn.calls_reader:
                    query = turn.query
                    image = updater.render(state)[0]
                    scored = qwen3vl_choice_nll(
                        model=reader,
                        processor=processor,
                        image=image,
                        query=format_mcq_query(query.text, query.choices),
                        choices=query.choices,
                        device=device,
                    )
                    correct += int(scored.predicted_index == query.target_index)
                    total += 1
    updater.train()
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Overfit 64 episodes through the frozen real Qwen Reader")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-size", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("The real Qwen overfit gate requires CUDA.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    episodes = read_jsonl(args.dataset)[: args.limit]
    if len(episodes) != args.limit:
        raise SystemExit(f"Requested {args.limit} episodes, found {len(episodes)}")

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=256 * 256,
        max_pixels=256 * 256,
    )
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(reader)
    reader.config.use_cache = False
    updater = LightweightVisualUpdater(output_size=args.output_size).to(device)
    optimizer = torch.optim.AdamW(updater.parameters(), lr=args.learning_rate, weight_decay=0.01)

    def update_fn(state, event_text, _episode_id, _turn_id):
        return updater.update(state, event_text)

    def reader_loss(image, query, target):
        return qwen3vl_target_only_ce(
            model=reader,
            processor=processor,
            image=image[0],
            query=query,
            target=target,
            device=device,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    best_accuracy = 0.0
    passed_step = None
    last_loss = None
    updater.train()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        episode = episodes[(step - 1) % len(episodes)]
        optimizer.zero_grad(set_to_none=True)
        result = run_episode(
            episode=episode,
            initial_state=updater.initial_state(batch_size=1, device=device, dtype=torch.float32),
            update_fn=update_fn,
            decode_fn=updater.render,
            reader_loss_fn=reader_loss,
            collect_states=False,
        )
        result.loss.backward()
        assert_no_frozen_parameter_grads(reader, "Qwen Reader")
        gradient_norm = torch.nn.utils.clip_grad_norm_(updater.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Non-finite lightweight updater gradient.")
        optimizer.step()
        last_loss = float(result.loss.item())
        if step % args.eval_every == 0 or step == args.steps:
            accuracy = evaluate(episodes, updater, reader, processor, device)
            best_accuracy = max(best_accuracy, accuracy)
            if accuracy >= args.threshold:
                passed_step = step
                break

    checkpoint = args.output_dir / "lightweight_qwen_overfit.pt"
    torch.save(
        {
            "schema_version": 1,
            "updater": updater.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": passed_step or args.steps,
            "seed": args.seed,
            "dataset_sha256": sha256_file(args.dataset),
            "reader_revision": (args.reader / ".locked_revision").read_text(encoding="utf-8").strip(),
        },
        checkpoint,
    )
    report = {
        "reader": "Qwen/Qwen3-VL-4B-Instruct",
        "episodes": len(episodes),
        "steps": passed_step or args.steps,
        "passed": passed_step is not None,
        "threshold": args.threshold,
        "best_accuracy": best_accuracy,
        "last_loss": last_loss,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed_step is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
