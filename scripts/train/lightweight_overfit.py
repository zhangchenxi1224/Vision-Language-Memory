from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import read_jsonl  # type: ignore[attr-defined] # noqa: E402
from vision_memory.data.episode import run_episode  # noqa: E402
from vision_memory.lightweight import HashChoiceReader, LightweightVisualUpdater  # noqa: E402


def _accuracy(episodes, updater, reader, device: torch.device) -> float:
    correct = 0
    total = 0
    updater.eval()
    with torch.no_grad():
        for episode in episodes:
            output = run_episode(episode, updater=updater, reader=reader, device=device)
            for prediction in output.reader_outputs:
                correct += int(prediction.logits.argmax(dim=-1).item() == prediction.target_index)
                total += 1
    updater.train()
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Overfit 64 episodes with the lightweight updater and fixed surrogate Reader"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to train.jsonl")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state-size", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "lightweight" / "overfit.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    episodes = read_jsonl(args.dataset)[: args.limit]
    if len(episodes) != args.limit:
        raise SystemExit(f"Requested {args.limit} episodes, found {len(episodes)}")

    updater = LightweightVisualUpdater(state_size=args.state_size, output_size=args.output_size).to(device)
    reader = HashChoiceReader().to(device)
    reader.requires_grad_(False)
    optimizer = torch.optim.AdamW(updater.parameters(), lr=args.lr, weight_decay=0.01)
    started = time.perf_counter()
    best_accuracy = 0.0
    passed_step: int | None = None
    last_loss = float("nan")

    for step in range(1, args.steps + 1):
        episode = episodes[(step - 1) % len(episodes)]
        optimizer.zero_grad(set_to_none=True)
        output = run_episode(episode, updater=updater, reader=reader, device=device)
        if not torch.isfinite(output.loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(updater.parameters(), 1.0)
        optimizer.step()
        last_loss = output.loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            accuracy = _accuracy(episodes, updater, reader, device)
            best_accuracy = max(best_accuracy, accuracy)
            if accuracy >= args.threshold:
                passed_step = step
                break

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": updater.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": args.seed,
            "step": passed_step or args.steps,
            "dataset": str(args.dataset.resolve()),
        },
        args.checkpoint,
    )
    report = {
        "reader": "fixed_hash_choice_surrogate_not_scientific_qwen",
        "episodes": len(episodes),
        "steps": passed_step or args.steps,
        "passed": passed_step is not None,
        "threshold": args.threshold,
        "best_accuracy": best_accuracy,
        "last_loss": last_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(args.checkpoint),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed_step is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
