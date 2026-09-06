"""Read-only raw tensor/file audit; run from the repository root, prints JSON."""

import argparse
import hashlib
import json
import pathlib
import sys
import torch

sys.path.insert(0, "src")
from vision_memory.repro import canonical_tensor_sha256 as th

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--raw-root", type=pathlib.Path, required=True)
parser.add_argument("--source-parent-raw", type=pathlib.Path, required=True)
args = parser.parse_args()
root = args.raw_root / "probe"
if not __debug__:
    raise RuntimeError("This evidence audit requires assertions enabled; do not run with -O.")


def h(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


cfg = read(pathlib.Path("configs/experiments/r11_new_identity_condition_probe.json"))
m = read(root / "manifest.json")
r = read(root / "result.json")
t = read(root / "terminal.json")
inv = read(root / "artifact_inventory.json")
assert h(root.parent / "probe-raw.tar.gz") == "fb77f5bac2d3c96816b5451254ea5fdb9288c7792f3b69f37f9af1b44f68206f"
assert (
    h(root / "manifest.json")
    == "6dc0652d6d1750c51341a55849757bcc3fa86b201a451eb020803e0912e0aaa2"
    == r["manifest_sha256"]
    == t["manifest_sha256"]
)
assert (
    h(root / "result.json") == "e414b9ed035f102a9ef7838da6948071e9e735ee6bdb39b2965c164c3ec02ecf" == t["result_sha256"]
)
assert h(root / "artifact_inventory.json") == "85ad0e79b98626ccb63ef760b09b9ee02d20720500ece5df79cd8ad54b37e087"
decl = set()
for rec in inv["artifacts"]:
    p = pathlib.PurePosixPath(rec["path"])
    assert not p.is_absolute() and ".." not in p.parts and rec["path"] not in decl
    decl.add(rec["path"])
    f = root.joinpath(*p.parts)
    assert f.stat().st_size == rec["bytes"] and h(f) == rec["sha256"]
assert decl == {
    p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "artifact_inventory.json"
}
assert inv["artifact_count"] == len(decl)
assert read(root / "config.json") == cfg
assert t["status"] == "technical_completed" and t["exit_code"] == 0 and t["lock_release"]["released"] is True
counters = {
    "condition_encoder_calls": 2,
    "unet_forward_calls": 0,
    "full_dreamlite_forward_calls": 0,
    "reader_forward_calls": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
}
assert m["counters"] == r["counters"] == t["counters"] == counters
assert m["validation"]["git_commit"] == r["git_commit"] == t["git_commit"] == "55c355f769c21ceb9b86e5090cdafa4090751d21"
assert m["probe_script_sha256"] == h(pathlib.Path("scripts/experiments/probe_r11_new_identity_condition.py"))
assert (
    m["frozen_before"] == m["frozen_after"]
    and not m["frozen_after"]["trainable_names"]
    and not m["frozen_after"]["gradient_names"]
)
assert (
    m["model_snapshot_start"]
    == m["model_snapshot_end"]
    == read(root / "model_snapshot_verification_start.json")
    == read(root / "model_snapshot_verification_end.json")
)
assert m["model_snapshot_start"]["manifest_sha256"] == cfg["dreamlite"]["snapshot_manifest_sha256"]
p = torch.load(root / "condition.pt", weights_only=True, map_location="cpu")
q = torch.load(root / "condition-repeat.pt", weights_only=True, map_location="cpu")
s = torch.load(root / "source.pt", weights_only=True, map_location="cpu")
for n, f in [("condition", "condition.pt"), ("repeat", "condition-repeat.pt"), ("source", "source.pt")]:
    a = m["artifacts"][n]
    assert h(root / f) == a["sha256"] and (root / f).stat().st_size == a["bytes"]
for name in ["prompt_embeds", "attention_mask"]:
    assert torch.equal(p[name], q[name])
    assert (
        th(p[name])
        == m["artifacts"]["condition"]["tensor_sha256"][name]
        == m["artifacts"]["repeat"]["tensor_sha256"][name]
        == r["artifact_audit"]["tensor_sha256"][name]
    )
    assert (
        str(p[name].dtype) == m["artifacts"]["condition"]["tensor_dtypes"][name]
        and list(p[name].shape) == m["artifacts"]["condition"]["tensor_shapes"][name]
    )
assert torch.isfinite(p["prompt_embeds"]).all() and torch.count_nonzero(p["prompt_embeds"]) > 0
assert p["prompt_embeds"].dtype == torch.bfloat16 and p["attention_mask"].dtype == torch.int64
assert (((p["attention_mask"] == 0) | (p["attention_mask"] == 1)).all()) and p["attention_mask"].any()
assert p["metadata"] == q["metadata"] == m["condition_metadata"]
for name in ["original_event_text", "actual_conditioning_text", "full_prompt"]:
    assert p["metadata"][name] == cfg["condition"][name]
    assert (
        hashlib.sha256(p["metadata"][name].encode()).hexdigest()
        == cfg["condition"][name + "_sha256"]
        == p["metadata"][name + "_sha256"]
    )
assert (
    th(s["source_rgb"]) == cfg["source"]["rgb_sha256"]
    and th(s["source_latents_fp32"]) == cfg["source"]["latents_fp32_sha256"]
)
assert th(p["prompt_embeds"]) != cfg["condition"]["old_prompt_embeds_sha256"]
parent = args.source_parent_raw
parent_files = {
    "comparison": parent / "aggregation-v1/comparison.json",
    "raw_artifacts": parent / "aggregation-v1/RAW_ARTIFACTS.json",
    "formal_manifest": parent / "formal-target01/run/manifest.json",
}
for k, f in parent_files.items():
    assert h(f) == cfg["parent_source_init"][k + "_sha256"] == m["parent_binding"]["artifacts"][k]["sha256"]
assert read(parent_files["formal_manifest"])["target_segment"] == m["parent_binding"]["target_segment"]
for obj in [r, t, m["information_boundary"]]:
    assert obj["formal_success"] is False and obj["phase2_allowed"] is False
assert r["engineering_gate"] is True and r["bridge_result_evaluated"] is False
print(
    json.dumps(
        {
            "schema": "vision_memory.r11-new-identity-condition-probe-local-audit.v1",
            "passed": True,
            "inventory_files_verified": len(decl),
            "archive_sha256": h(root.parent / "probe-raw.tar.gz"),
            "manifest_sha256": h(root / "manifest.json"),
            "result_sha256": h(root / "result.json"),
            "terminal_sha256": h(root / "terminal.json"),
            "inventory_sha256": h(root / "artifact_inventory.json"),
            "git_commit": t["git_commit"],
            "condition_tensor_sha256": r["artifact_audit"]["tensor_sha256"],
            "condition_shapes": m["artifacts"]["condition"]["tensor_shapes"],
            "condition_dtypes": m["artifacts"]["condition"]["tensor_dtypes"],
            "repeat_bitwise_equal": True,
            "embedding_differs_from_parent": True,
            "source_hashes_match": True,
            "original_target_matches_parent": True,
            "snapshot_records_match": True,
            "snapshot_files_rehashed_locally": False,
            "actual_model_calls_rerun_locally": False,
            "counts_verified_from_bound_records": counters,
            "elapsed_seconds": r["elapsed_seconds"],
            "started_at_utc": t["started_at_utc"],
            "completed_at_utc": t["completed_at_utc"],
            "formal_success": False,
            "phase2_allowed": False,
        },
        indent=2,
    )
)
