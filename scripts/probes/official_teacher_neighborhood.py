"""Measure oracle readback robustness; these images are NOT Writer outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    from scripts.train import train_latent_bank_unet as train
    from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
    from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV
    from vision_memory.training.latent_bank_unet import load_teacher_bank, file_sha256, stable_seed
    command = json.loads((a.run / "commands.json").read_text())["commands"][-1]
    args = train.parser().parse_args(command[3:])
    bank, teachers = load_teacher_bank(args.bank_manifest)
    if not a.worker:
        env = {**os.environ, **snapshot_environment(bank), **REQUIRED_DETERMINISM_ENV,
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        return subprocess.call([sys.executable, str(Path(__file__).resolve()),
            "--run", str(a.run), "--output", str(a.output), "--worker"], env=env)
    terminal = json.loads((a.run / "terminal.json").read_text())
    if terminal["state"] != "completed" or file_sha256(a.run / "train/result.json") != terminal["training_result_sha256"]:
        raise RuntimeError("A verified completed parent run is required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Probe source must be immutable")
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    from PIL import Image
    from vision_memory.repro import configure_strict_cuda_determinism, canonical_tensor_sha256
    configure_strict_cuda_determinism(args.seed)
    runtime = train.load_runtime(args, bank)
    rows = []
    identity = {"schema": "official-teacher-neighborhood/v1", "probe_commit": commit,
        "parent_result_sha256": terminal["training_result_sha256"],
        "bank_sha256": file_sha256(args.bank_manifest), "runtime_snapshots": runtime["snapshots"],
        "additional_protocol_binding": runtime.get("protocol_binding", {}),
        "scope": "oracle perturbation diagnostic, not Writer accuracy; no teacher selection or optimization",
        "isotropic_rms": [.001, .003, .01, .03, .1, .3], "direction_seeds": [0, 1, 2],
        "writer_seed_index": 0, "writer_interpolation_fractions": [.01, .03, .1, .3, 1.]}
    train.write_json(a.output / "identity.json", identity)
    with torch.no_grad():
        for group in train.training_groups(bank, args.target_mode):
            tid = group["teacher_ids"][0]
            target = teachers[tid].float().cpu()
            filename = hashlib.sha256(group["question_id"].encode()).hexdigest()[:16] + "-seed-00.pt"
            artifact = a.run / "train/trained" / filename
            complete = json.loads((artifact.parent / "complete.json").read_text())
            if file_sha256(artifact) != complete["artifact_hashes"][filename]:
                raise RuntimeError("Parent generated latent changed")
            generated = torch.load(artifact, map_location="cpu", weights_only=True)["latent"]
            samples = [("exact_teacher", 0., None, target)]
            for index in identity["direction_seeds"]:
                seed = stable_seed(args.seed, "oracle-neighborhood-diagnostic", index)
                direction = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed))
                direction /= direction.square().mean().sqrt()
                samples.extend(("isotropic", rms, seed, target + rms * direction) for rms in identity["isotropic_rms"])
            samples.extend(("toward_writer_seed0", f, None, (1-f)*target + f*generated)
                           for f in identity["writer_interpolation_fractions"])
            for kind, amplitude, seed, latent in samples:
                pixels = train.decode_model_latents_unit_interval(runtime["pipe"].vae,
                    latent.to(runtime["vae_device"]), clamp=True)
                if kind == "exact_teacher":
                    image = (pixels[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()*255).round().astype("uint8")
                    Image.fromarray(image).save(a.output / f"teacher-{tid}.png")
                # Fixed original and lexically last held-out variant; all retained.
                prompts = ["original_open", sorted(k for k in group["question_variants"] if k != "original_open")[-1]]
                for prompt_id in prompts:
                    query = group["question_variants"][prompt_id]
                    ce = train.qwen3vl_answer_eos_ce(model=runtime["reader"], processor=runtime["processor"],
                        image=pixels[0].to(runtime["reader_device"]), device=runtime["reader_device"],
                        query=query, target=group["answer"], termination=runtime["termination"], lambda_eos=1.,
                        require_image_grad=False, deterministic_ce=True,
                        reader_resize_contract=train.R3_QWEN_READER_RESIZE_CONTRACT)
                    generation = train.generate_short_answer(model=runtime["reader"], processor=runtime["processor"],
                        image=pixels.to(runtime["reader_device"]), query=query,
                        device=runtime["reader_device"], max_new_tokens=32, do_sample=False)
                    row = {"kind": kind, "amplitude": amplitude, "noise_seed": seed, "teacher_id": tid,
                        "prompt_id": prompt_id, "query": query, "gold": group["answer"],
                        "latent_rms_error": float((latent-target).square().mean().sqrt()),
                        "latent_sha256": canonical_tensor_sha256(latent),
                        "image_sha256": canonical_tensor_sha256(pixels.cpu()),
                        "answer_ce": float(ce.answer_loss), "eos_ce": float(ce.eos_loss), **generation,
                        "scorer": train.generation_diagnostics(generation, group["answer"],
                            ce.target_ids[0, :ce.answer_token_count].cpu().tolist())}
                    rows.append(row)
                    train.append_jsonl(a.output / "generations.jsonl", row)
                    print(json.dumps({k: row[k] for k in ("kind", "amplitude", "prompt_id", "latent_rms_error", "scorer")}), flush=True)
    train.write_json(a.output / "complete.json", {"rows": len(rows), "identity": identity,
        "artifact_hashes": {p.name: file_sha256(p) for p in a.output.iterdir() if p.is_file()}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
