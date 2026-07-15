from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CLUSTER_SCRIPTS = ROOT / "scripts" / "cluster"
sys.path.insert(0, str(CLUSTER_SCRIPTS))

from submit_experiment_dag import (  # noqa: E402
    DEFAULT_RESOURCES,
    Paths,
    build_jobs,
    render_sbatch,
)
from compare_checkpoints import compare_values  # noqa: E402


class ExperimentDagTest(unittest.TestCase):
    def paths(self) -> Paths:
        return Paths(
            project=Path("/remote/project"),
            environment=Path("/remote/env_torch271"),
            model_root=Path("/remote/models"),
            prefeval_root=Path("/remote/data/PrefEval"),
            run_root=Path("/remote/runs/unit"),
        )

    def test_core_dag_contains_strict_gates_and_three_seed_matrix(self):
        jobs, specification = build_jobs(
            self.paths(),
            fetch_prefeval=False,
            include_ablations=False,
            set_only_train=None,
            set_only_dev=None,
            include_prefeval_adaptation=False,
        )
        by_name = {job.name: job for job in jobs}
        self.assertEqual(by_name["D1_qwen_sanity"].dependencies, ("D0_data",))
        self.assertEqual(by_name["D2_lightweight_overfit"].dependencies, ("D1_qwen_sanity",))
        overfit_command = "\n".join(by_name["D2_lightweight_overfit"].commands)
        self.assertIn("lightweight_episode.py", overfit_command)
        self.assertIn("--overfit-gate", overfit_command)
        self.assertIn("--overfit-episodes 64", overfit_command)
        self.assertIn("--max-optimizer-steps 2000", overfit_command)
        self.assertIn("--gradient-accumulation 1", overfit_command)
        self.assertNotIn("lightweight_qwen_overfit.py", overfit_command)
        self.assertEqual(by_name["P5_resume_gate"].dependencies, ("P4_select",))
        pilot_commands = "\n".join(
            command
            for name in ("P0_blank", "P0_frozen", "P1_lr_3em5", "P2_lr_1em4", "P3_lr_3em4")
            for command in by_name[name].commands
        ).replace("\\", "/")
        self.assertIn("synthetic_v2/dev.jsonl", pilot_commands)
        self.assertNotIn("synthetic_v2/test_id.jsonl", pilot_commands)
        for seed in range(3):
            self.assertIn(f"F_dreamlite_seed_{seed}", by_name)
            self.assertIn(f"F_lightweight_seed_{seed}", by_name)
            self.assertIn(f"F_static_seed_{seed}", by_name)
            self.assertIn(f"E_text_seed_{seed}", by_name)
            self.assertIn(f"E_frozen_seed_{seed}", by_name)
            self.assertIn(f"E_dreamlite_seed_{seed}", by_name)
            static_command = "\n".join(by_name[f"F_static_seed_{seed}"].commands)
            self.assertNotIn("--learn-initial-state", static_command)
        self.assertEqual(len(specification["candidates"]), 3)
        self.assertEqual(specification["selection_split"], "dev")
        self.assertTrue(
            all("dev_predictions.jsonl" in item["predictions"] for item in specification["candidates"])
        )
        self.assertNotIn("A_set_only_seed_0", by_name)
        self.assertEqual(self.paths().synthetic.name, "synthetic_v2")
        score_commands = "\n".join(by_name["E_score_main"].commands)
        self.assertIn("prefeval_protocol_predictions.jsonl", score_commands)
        self.assertIn("prefeval_forced_k10_scores_implicit_choice.json", score_commands)
        self.assertIn("--headline-forced-write-k 10", score_commands)
        self.assertIn("synthetic_ood_predictions.jsonl", score_commands)
        self.assertIn("synthetic_ood.jsonl", score_commands)

    def test_full_ablation_matrix_generates_independent_set_only_and_scores_every_output(self):
        jobs, _ = build_jobs(
            self.paths(),
            fetch_prefeval=False,
            include_ablations=True,
            set_only_train=None,
            set_only_dev=None,
            include_prefeval_adaptation=True,
        )
        by_name = {job.name: job for job in jobs}
        data_commands = "\n".join(by_name["D0_data"].commands)
        self.assertIn("synthetic_set_only_v2", data_commands)
        self.assertIn("--transition-profile set-only", data_commands)
        self.assertIn("synthetic_v2_validation.json", data_commands)
        self.assertIn("synthetic_set_only_v2_validation.json", data_commands)
        self.assertIn("--expected-records 3000", data_commands)
        self.assertIn("--expected-records 2400", data_commands)
        self.assertIn("--expected-records 2160", data_commands)
        self.assertIn("A_set_only_seed_0", by_name)
        self.assertIn("A_rank4_blank_1k_seed_0", by_name)
        self.assertIn("E_score_ablations", by_name)
        self.assertIn("E_score_noise", by_name)
        self.assertIn("E_prefeval_adapted_seed_0", by_name)
        noop_commands = "\n".join(by_name["A_noop_skip_seed_0"].commands)
        self.assertIn("--noop-policy skip", noop_commands)
        ablation_score_commands = "\n".join(by_name["E_score_ablations"].commands)
        self.assertIn("ablation_summary.json", ablation_score_commands)
        self.assertIn("rank4_blank_1k", ablation_score_commands)
        self.assertIn("dreamlite_main", ablation_score_commands)
        self.assertIn("E_dreamlite_seed_0", by_name["E_score_ablations"].dependencies)
        self.assertIn("noise_robustness_summary.json", "\n".join(by_name["E_score_noise"].commands))
        adapted_commands = "\n".join(by_name["E_prefeval_adapted_seed_0"].commands)
        self.assertIn("PrefEval adapted state-streaming protocol", adapted_commands)

    def test_every_sbatch_is_single_node_and_pins_runtime(self):
        paths = self.paths()
        jobs, _ = build_jobs(
            paths,
            fetch_prefeval=False,
            include_ablations=False,
            set_only_train=None,
            set_only_dev=None,
            include_prefeval_adaptation=False,
        )
        for job in jobs:
            rendered = render_sbatch(
                job,
                DEFAULT_RESOURCES[job.resource_class],
                paths=paths,
                partition="a800",
                expected_commit="a" * 40,
                expected_torch="2.7.1+cu118",
            )
            self.assertIn("#SBATCH --nodes=1", rendered)
            self.assertIn("2.7.1+cu118", rendered)
            self.assertIn('test "$(git rev-parse HEAD)"', rendered)
            self.assertIn('test -z "$(git status --porcelain)"', rendered)

    def test_default_cli_is_a_real_dry_run_without_sbatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            (project / "README.md").write_text("fixture\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(project)], check=True)
            subprocess.run(["git", "-C", str(project), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(project), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(project), "add", "README.md"], check=True)
            subprocess.run(["git", "-C", str(project), "commit", "-q", "-m", "fixture"], check=True)
            runs = root / "runs"
            result = subprocess.run(
                [
                    sys.executable,
                    str(CLUSTER_SCRIPTS / "submit_experiment_dag.py"),
                    "--project-root",
                    str(project),
                    "--runs-root",
                    str(runs),
                    "--run-name",
                    "dry-run",
                    "--through",
                    "data",
                    "--environment",
                    str(root / "env"),
                    "--model-root",
                    str(root / "models"),
                    "--prefeval-root",
                    str(root / "PrefEval"),
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            emitted = json.loads(result.stdout)
            manifest = json.loads((runs / "dry-run" / "submission.json").read_text(encoding="utf-8"))
            self.assertTrue(emitted["dry_run"])
            self.assertEqual(manifest["status"], "dry_run_complete")
            self.assertEqual(list(manifest["jobs"]), ["D0_data"])
            self.assertIsNone(manifest["jobs"]["D0_data"]["job_id"])
            sbatch = (runs / "dry-run" / "sbatch" / "D0_data.sbatch").read_text(encoding="utf-8")
            self.assertIn("#SBATCH --nodes=1", sbatch)
            self.assertNotIn("#SBATCH --gres=gpu:", sbatch)

    def test_checkpoint_comparison_detects_resume_drift(self):
        reference = {"weight": torch.tensor([1.0, 2.0]), "step": 10}
        self.assertEqual(compare_values(reference, reference, atol=0.0, rtol=0.0, path="root"), [])
        resumed = {"weight": torch.tensor([1.0, 2.1]), "step": 10}
        mismatches = compare_values(reference, resumed, atol=1e-6, rtol=1e-5, path="root")
        self.assertTrue(any("max_abs_difference" in message for message in mismatches))


if __name__ == "__main__":
    unittest.main()
