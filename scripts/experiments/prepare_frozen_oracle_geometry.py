"""Write the prospective config and outcome-independent run manifest."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vision_memory.training.frozen_oracle_geometry import build_manifest, canonical_hash, CHECKPOINT_STEPS


def main():
    parent = json.loads((ROOT / "configs/experiments/r11_new_frozen_dreamlite_oracle_phase1a.json").read_text())
    manifest = build_manifest()
    config = {
        "schema": "vision_memory.frozen-oracle-geometry-config.v1",
        "protocol": "Frozen-DreamLite-Oracle-Geometry-20260908",
        "base_commit": "7c27edb",
        "fixed_data": parent["fixed_data"], "target_selection": parent["target_selection"],
        "source_state": parent["source_state"], "dreamlite_path": parent["dreamlite_path"],
        "choice_views": parent["choice_views"],
        "optimization": {"optimizer": "Adam", "learning_rate": .05, "weight_decay": 0.,
                         "optimizer_steps": 256, "gradient_clipping": None,
                         "checkpoint_steps": list(CHECKPOINT_STEPS), "save_latents_every_step": True,
                         "primary_endpoint": "raw_step256", "best_checkpoint_selection_forbidden": True},
        "precision": {"x_T": "float32", "dreamlite": "float32", "reader": "bfloat16",
                      "strict_determinism": True, "tf32": False, "attention": "math_sdpa"},
        "initialization": {"rng": "numpy.PCG64", "draw_dtype": "float64", "storage_dtype": "float32",
                           "matching": "population mean=0 variance=1; log empirical moments",
                           "gaussian": "N(0,1), no sample normalization",
                           "uniform": "U(-sqrt(3),sqrt(3))", "rademacher": "equiprobable +/-1",
                           "sphere": "Gaussian divided by sample RMS; radius=sqrt(D)",
                           "heavy_tail": "Student-t df5 times sqrt(3/5); no sample normalization",
                           "paired_seeds_across_tasks_and_scales": True},
        "primary_success": {"endpoint_step": 256, "reverse_choice_views_correct": 4,
                            "minimum_margin_gt": 0, "technical_pass_required": True,
                            "old_phase1a_gate_reported_separately": True},
        "deployment": {"gpus": 4, "gpu_type": "H200", "worker_gpu_pairs": [[0,1],[2,3]],
                       "stages": ["probe", "A1", "A2", "A3", "A4", "A9", "A5_A8", "analysis"],
                       "reproducibility_failure_stops_multistart": True,
                       "model_hash_failure_stops_all": True,
                       "maximum_runtime_hours": 72, "resume_complete_runs_only_after_hash_verification": True},
        "scientific_limits": ["sampled reachability is not geometric basin volume",
                              "linear interpolation failure does not prove disconnected basins",
                              "64 centered samples have rank at most 63",
                              "choice success is not open-answer or recurrent-memory success"],
        "stage_bcd": {"bank_tasks": 32, "bank_selection": "original locked train F1 hash order",
                      "failed_task_replacement": False, "requires_geometry_decision": True,
                      "controller_inputs": ["source_latent", "event_condition_embedding"],
                      "split_unit": "source_episode/event", "full_ft_default": False},
        "manifest_sha256": canonical_hash(manifest),
    }
    for name, value in (("frozen_oracle_geometry.json", config),
                        ("frozen_oracle_geometry_manifest.json", manifest)):
        path = ROOT / "configs/experiments" / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(manifest["runs"]), "manifest_sha256": canonical_hash(manifest)}))


if __name__ == "__main__":
    main()
