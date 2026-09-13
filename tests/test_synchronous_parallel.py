"""Actual multiprocess reduction, optimizer recovery and evaluation integrity."""
import copy
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vision_memory.training.synchronous_parallel import microbatch_indices, reduce_gradients, merge_evaluation
from vision_memory.training.checkpoint import save_training_checkpoint, load_training_checkpoint


def _worker(rank, port, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4)
    torch.manual_seed(123)
    module = torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.Tanh(), torch.nn.Linear(11, 3))
    serial = copy.deepcopy(module)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.003)
    reference = torch.optim.AdamW(serial.parameters(), lr=0.003)
    data = torch.randn(4, 7)
    target = torch.randn(4, 3)
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        for micro in microbatch_indices(4, 4, rank):
            ((module(data[micro:micro+1]) - target[micro:micro+1]).square().mean() / 4).backward()
        reduce_gradients(list(module.parameters()), bucket_bytes=100)
        for micro in range(4):
            ((serial(data[micro:micro+1]) - target[micro:micro+1]).square().mean() / 4).backward()
        for actual, expected in zip(module.parameters(), serial.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-7, rtol=2e-5)
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(serial.parameters(), 1.)
        optimizer.step()
        reference.step()
        for actual, expected in zip(module.parameters(), serial.parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        if step == 1:
            checkpoint = Path(output) / f"rank-{rank}.pt"
            save_training_checkpoint(checkpoint, trainable_module=module, optimizer=optimizer,
                epoch=0, episode_cursor=2, optimizer_step=2, manifest={"world": 4})
            restored = copy.deepcopy(module)
            restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.9)
            load_training_checkpoint(checkpoint, trainable_module=restored, optimizer=restored_opt,
                                     expected_manifest={"world": 4})
        elif step > 1:
            restored_opt.zero_grad(set_to_none=True)
            for micro in microbatch_indices(4, 4, rank):
                ((restored(data[micro:micro+1]) - target[micro:micro+1]).square().mean() / 4).backward()
            reduce_gradients(list(restored.parameters()), bucket_bytes=100)
            torch.nn.utils.clip_grad_norm_(restored.parameters(), 1.)
            restored_opt.step()
            for actual, expected in zip(restored.parameters(), module.parameters()):
                assert torch.equal(actual, expected)
    values = [None] * 4
    dist.all_gather_object(values, [p.detach().tolist() for p in module.parameters()])
    assert all(value == values[0] for value in values)
    dist.destroy_process_group()


def test_actual_four_process_gradient_adam_and_exact_recovery(tmp_path):
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        port = endpoint.getsockname()[1]
    mp.spawn(_worker, args=(port, str(tmp_path)), nprocs=4, join=True)


def test_global_draw_indices_are_conserved():
    assert [list(microbatch_indices(4, 4, rank)) for rank in range(4)] == [[0], [1], [2], [3]]
    assert sorted(i for rank in range(2) for i in microbatch_indices(8, 2, rank)) == list(range(8))
    with pytest.raises(ValueError):
        microbatch_indices(3, 4, 0)


def test_merge_rejects_missing_and_duplicate_cells(tmp_path):
    from scripts.train import train_latent_bank_unet as training
    args = SimpleNamespace(output_dir=tmp_path, seed=10, eval_seeds=2)
    bank = {"groups": [{"question_id": "g", "question_variants": {"q": "question"}}]}
    shard = tmp_path / "parallel-shards/rank-00/baseline"
    shard.mkdir(parents=True)
    rows = [{"question_id": "g", "phase": "baseline", "condition": "matched", "noise_seed": 1, "prompt_id": "q"}] * 4
    (shard / "generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    training.write_json(shard / "complete.json", {"artifact_hashes": {
        "generations.jsonl": training.file_sha256(shard / "generations.jsonl")}, "summary": {"geometry": {"g": {}}}})
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        merge_evaluation(args, bank, "baseline", 1, training)
