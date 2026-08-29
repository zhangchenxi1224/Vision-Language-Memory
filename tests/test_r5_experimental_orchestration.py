from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_script(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


selector = load_script("select_r5_pilot_under_test", "scripts/experiments/select_r5_pilot.py")
topology = load_script("assess_r5_topology_under_test", "scripts/experiments/assess_r5_topology.py")
resume_assessor = load_script("assess_r5_resume_under_test", "scripts/experiments/assess_r5_resume.py")
controller = load_script("run_r5_pipeline_under_test", "scripts/inspire/run_r5_compose_pipeline.py")
reporter = load_script("render_r5_report_under_test", "scripts/reporting/render_r5_compose_report.py")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def pilot_summary(*, state: str, horizon: int, delayed: float, formal: float, eligible: bool):
    endpoint = {
        "breakdowns": {
            "query_gap": {
                "ema_step128|mechanism_select_32|normal|3": {"mean_margin": 0.4}
            }
        }
    }
    return {
        "persistent_state": state,
        "tbptt_horizon": horizon,
        "gradient_mode": "drtune_stateful",
        "selected_step_count": 2,
        "elapsed_seconds": 100.0 + horizon,
        "pilot_selection": {
            "technical_gate_passed": eligible,
            "mechanism_gate_passed": eligible,
            "eligible_for_selection": eligible,
            "delayed_mechanism_ce": {"m0": 2.0, "endpoint": delayed, "delta": delayed - 2.0},
            "formal_select_ce": {"m0": 2.0, "endpoint": formal, "delta": formal - 2.0},
            "endpoint_normal_minus_reset_ce": -0.2,
        },
        "pilot_endpoint_evaluation": endpoint,
        "technical_gate": {"passed": eligible},
    }


class R5ExperimentalOrchestrationTest(unittest.TestCase):
    def test_resume_audit_requires_exact_endpoint_state_and_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct = root / "direct"
            resumed = root / "resumed"
            for directory in (direct, resumed):
                directory.mkdir()
                payload = {
                    "schema_version": 1,
                    "optimizer_step": 2,
                    "trainable_state": {"lora": torch.tensor([1.0, 2.0])},
                    "optimizer": {"state": {0: {"step": torch.tensor(2.0)}}, "param_groups": []},
                    "trainer_state": {"ema_state": {"lora": torch.tensor([0.9, 1.9])}},
                }
                torch.save(payload, directory / "endpoint_raw.pt")
                (directory / "metrics.jsonl").write_text(
                    json.dumps({"kind": "optimizer_step", "optimizer_step": 2, "loss_mean": 1.25}) + "\n",
                    encoding="utf-8",
                )
            result = resume_assessor.assess(direct, resumed)
            self.assertTrue(result["passed"])
            payload = torch.load(resumed / "endpoint_raw.pt", map_location="cpu", weights_only=False)
            payload["trainable_state"]["lora"][0] += 1.0
            torch.save(payload, resumed / "endpoint_raw.pt")
            self.assertFalse(resume_assessor.assess(direct, resumed)["passed"])

    def test_pilot_selector_is_gate_then_lexicographic(self):
        values = [
            {
                "name": "rgb-h2",
                "eligible": False,
                "delayed_ce": 0.1,
                "formal_ce": 0.1,
                "longest_gap_margin": 1.0,
                "elapsed_seconds": 1.0,
            },
            {
                "name": "latent-h2",
                "eligible": True,
                "delayed_ce": 1.2,
                "formal_ce": 1.1,
                "longest_gap_margin": 0.3,
                "elapsed_seconds": 100.0,
            },
            {
                "name": "latent-h4",
                "eligible": True,
                "delayed_ce": 1.1,
                "formal_ce": 1.5,
                "longest_gap_margin": 0.9,
                "elapsed_seconds": 120.0,
            },
        ]
        result = selector.select(values)
        self.assertEqual(result["winner"]["name"], "latent-h4")
        self.assertIn("gap=3", result["gap4_note"])

    def test_topology_requires_memory_throughput_parity_and_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            same = root / "same"
            split = root / "split"
            for directory, same_device, throughput, peak in (
                (same, True, 0.7, 90.0),
                (split, False, 1.0, 30.0),
            ):
                write_json(
                    directory / "summary.json",
                    {
                        "same_device": same_device,
                        "technical_gate": {"passed": True},
                        "micro_segments_per_second": throughput,
                        "updater_peak_memory_gib": peak,
                        "reader_peak_memory_gib": peak,
                    },
                )
                (directory / "metrics.jsonl").write_text(
                    "\n".join(
                        json.dumps({"kind": "optimizer_step", "loss_mean": value})
                        for value in (2.0, 1.9)
                    )
                    + "\n",
                    encoding="utf-8",
                )
            result = topology.assess(same, split)
            self.assertTrue(result["passed"])
            self.assertEqual(result["decision"], "single_h200_parallel_arms")

    def test_controller_builds_locked_same_and_split_device_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                repo=ROOT,
                output_root=root / "experiment",
                python=Path(sys.executable),
                report_python=Path(sys.executable),
                train=Path("train.jsonl"),
                dev=Path("dev.jsonl"),
                dreamlite=Path("dreamlite"),
                reader=Path("reader"),
                dreamlite_manifest_sha256="a" * 64,
                reader_manifest_sha256="b" * 64,
            )
            value = controller.Controller(args)
            same = value.base_train_command(
                profile="pilot",
                output_dir=root / "same",
                persistent_state="latent",
                horizon=4,
                gradient_mode="drtune_stateful",
                selected_step_count=2,
                seed=0,
                same_device=True,
                checkpoint_every=64,
            )
            split = value.base_train_command(
                profile="pilot",
                output_dir=root / "split",
                persistent_state="latent",
                horizon=4,
                gradient_mode="drtune_stateful",
                selected_step_count=2,
                seed=0,
                same_device=False,
                checkpoint_every=64,
            )
            self.assertEqual(Path(same[0]), Path(sys.executable))
            self.assertEqual(Path(split[0]), Path(sys.executable))
            self.assertEqual(same[same.index("--reader-device") + 1], "cuda:0")
            self.assertEqual(split[split.index("--reader-device") + 1], "cuda:1")
            self.assertIn("--strict-determinism", same)
            environment = value.environment("0")
            self.assertEqual(environment["PYTHONHASHSEED"], "0")
            self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertEqual(environment["OMP_NUM_THREADS"], "1")
            self.assertEqual(environment["MKL_NUM_THREADS"], "1")

    def test_controller_uses_separate_reporting_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                repo=ROOT,
                output_root=root / "experiment",
                python=Path("training-python"),
                report_python=Path("reporting-python"),
                train=Path("train.jsonl"),
                dev=Path("dev.jsonl"),
                dreamlite=Path("dreamlite"),
                reader=Path("reader"),
                dreamlite_manifest_sha256="a" * 64,
                reader_manifest_sha256="b" * 64,
            )
            value = controller.Controller(args)
            report = args.output_root / "delivery" / "FINAL_REPORT.md"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("test\n", encoding="utf-8")
            with mock.patch.object(controller.subprocess, "run") as run:
                value.render_report()
            self.assertEqual(Path(run.call_args.args[0][0]), args.report_python)

    def test_full_gradient_winner_forces_split_two_gpu_execution(self):
        single_h200 = {"decision": "single_h200_parallel_arms"}
        self.assertTrue(
            controller.Controller._same_device_parallel_allowed(
                single_h200, {"gradient_mode": "drtune_stateful"}
            )
        )
        self.assertFalse(
            controller.Controller._same_device_parallel_allowed(
                single_h200, {"gradient_mode": "full"}
            )
        )
        self.assertFalse(
            controller.Controller._same_device_parallel_allowed(
                {"decision": "split_h200_serial"}, {"gradient_mode": "drtune_stateful"}
            )
        )

    def test_report_renderer_handles_pilot_only_negative_delivery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            output = root / "delivery"
            write_json(root / "rescue_decision.json", {"reason": "all pilots failed"})
            write_json(root / "pilot_selection.json", {"decision": "no_eligible_pilot", "winner": None})
            write_json(
                root / "runs" / "pilot-latent-h2" / "summary.json",
                pilot_summary(state="latent", horizon=2, delayed=2.1, formal=2.2, eligible=False),
            )
            result = reporter.render(root, output)
            self.assertEqual(result["status"], "completed")
            self.assertTrue((output / "FINAL_REPORT.md").is_file())
            self.assertTrue((output / "DELIVERY_MANIFEST.json").is_file())


if __name__ == "__main__":
    unittest.main()
