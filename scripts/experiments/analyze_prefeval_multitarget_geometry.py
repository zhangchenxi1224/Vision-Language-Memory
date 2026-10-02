"""Read-only, CPU-only geometry audit of the qualified teacher pools."""
import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    x = torch.tensor(values, dtype=torch.float64)
    return {"n": len(values), "min": x.min().item(), "median": x.quantile(.5).item(),
            "mean": x.mean().item(), "max": x.max().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    paths = {"F8": args.root / "pilot64/F8-bank.json",
             "S8": args.root / "pilot64/S8-bank.json",
             "C8": args.root / "round2/pilot64/C8-bank.json"}
    f1 = json.loads((args.root / "pilot64/F1-bank.json").read_text())["targets"]
    result = {"time_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha(__file__), "device": "cpu", "threads": 2,
              "scope": "Qualified training-side targets only; latent RMS is not semantic diversity.",
              "arms": {}}
    for arm, path in paths.items():
        bank = json.loads(path.read_text())
        per_pref, pair_values, anchor_values = {}, [], []
        for pid, entries in bank["targets"].items():
            tensors, tensor_hashes, pixel_hashes, file_hashes = [], set(), set(), set()
            anchor = None
            for entry in entries:
                assert sha(entry["latent"]) == entry["latent_sha256"]
                assert sha(entry["png"]) == entry["png_sha256"]
                tensor = torch.load(entry["latent"], map_location="cpu", weights_only=True)
                assert isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all()
                tensor = tensor.to(torch.float64).contiguous()
                tensors.append(tensor)
                tensor_hashes.add(hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
                with Image.open(entry["png"]) as image:
                    pixel_hashes.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
                file_hashes.add(entry["png_sha256"])
                if arm == "C8" and entry["target_index"] == 0:
                    anchor = tensor
                    assert entry["latent_sha256"] == f1[pid][0]["latent_sha256"]
                    assert entry["png_sha256"] == f1[pid][0]["png_sha256"]
            distances = [((a - b).square().mean().sqrt()).item()
                         for a, b in itertools.combinations(tensors, 2)]
            pair_values.extend(distances)
            values = {"qualified_targets": len(entries), "unique_tensor_values": len(tensor_hashes),
                      "unique_png_pixels": len(pixel_hashes), "unique_png_files": len(file_hashes),
                      "pair_rms": stats(distances), "pair_rms_values": distances,
                      "target_rms": stats([x.square().mean().sqrt().item() for x in tensors])}
            if arm == "C8":
                assert anchor is not None
                offsets = [(x - anchor).square().mean().sqrt().item()
                           for x, entry in zip(tensors, entries) if entry["target_index"] != 0]
                assert max(offsets) <= .100001
                anchor_values.extend(offsets)
                values["new_target_to_anchor_rms"] = stats(offsets)
                values["anchor_byte_identical_to_F1"] = True
            per_pref[pid] = values
        assert len(per_pref) == 64
        result["arms"][arm] = {"bank_sha256": sha(path), "per_preference": per_pref,
            "qualified_targets": sum(x["qualified_targets"] for x in per_pref.values()),
            "pair_rms": stats(pair_values),
            "preferences_with_exact_tensor_duplicates": sum(x["unique_tensor_values"] < x["qualified_targets"] for x in per_pref.values()),
            "preferences_with_exact_pixel_duplicates": sum(x["unique_png_pixels"] < x["qualified_targets"] for x in per_pref.values())}
        if anchor_values:
            result["arms"][arm]["new_target_to_anchor_rms"] = stats(anchor_values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps({arm: {k: v for k, v in data.items() if k != "per_preference"}
                      for arm, data in result["arms"].items()}))


if __name__ == "__main__":
    main()
