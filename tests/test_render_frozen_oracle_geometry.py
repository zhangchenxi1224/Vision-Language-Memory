"""Rendering tests use tiny labeled synthetic fixtures, never experiment data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.experiments import render_frozen_oracle_geometry as plot


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_missing_results_render_pending_without_fabrication(tmp_path):
    result = plot.render(tmp_path, tmp_path / "figures", dpi=45)
    assert not result["errors"]
    assert all(row["status"] == "pending" for row in result["plots"].values())
    assert len(list((tmp_path / "figures").glob("*.png"))) == 7
    assert result["scientific_success_claim"] is False


def test_common_pca_preserves_full_distances_when_rank_is_two():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0]], dtype=np.float32)
    projected, fraction = plot.common_pca(points)
    expected = np.sqrt(np.mean((points[:, None] - points[None]) ** 2, axis=2))
    actual = np.sqrt(np.sum((projected[:, None] - projected[None]) ** 2, axis=2))
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert abs(fraction - 1) < 1e-12


def test_complete_synthetic_inputs_render_all_six_scientific_panels(tmp_path):
    analysis = tmp_path / "analysis"
    write(analysis / "success_rates.json", [{"study": "A2", "distribution": "gaussian", "scale": 1.,
                                              "planned": 4, "completed": 3, "successes": 2}])
    geometry = {space: {"within_rmse": .2, "between_rmse": .4,
                        "pca": {"r90": 1, "r95": 2, "r99": 3, "random_null_r95": 3, "sample_rank_limit": 3}}
                for space in plot.SPACES}
    write(analysis / "geometry_statistics.json", {"all_completed": geometry, "successful_only": geometry})
    for seed in range(2):
        folder = tmp_path / "runs" / f"A2-s{seed}"
        folder.mkdir(parents=True)
        trajectory = []
        for step in plot.SNAPSHOTS:
            record = {"step": step}
            for space in ("xT", "z"):
                path = folder / f"{space}-{step}.npy"
                np.save(path, np.array([seed, step / 256, seed + step / 256, 0], dtype=np.float32).reshape(1, 1, 2, 2))
                record[space] = {"path": str(path), "file_sha256": plot.file_hash(path)}
            trajectory.append(record)
        write(folder / "trajectory_index.json", trajectory)
        write(folder / "summary.json", {"seed": seed, "technical_pass": True, "optimizer_steps": 256,
                                         "trajectory_index_path": str(folder / "trajectory_index.json")})
        metrics = [{"step": step, "ce": 1 / (step + 1), "gradient": {"norm": 1 / (step + 1)}} for step in range(3)]
        (folder / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics) + "\n")
    cases = []
    for space in ("xT", "z"):
        for coefficient in (0., .5, 1.):
            cases.append({"case": {"kind": "interpolation", "space": space, "lambda": coefficient,
                                    "left_run_id": "synthetic-0", "right_run_id": "synthetic-1"},
                          "qa_pass": coefficient != .5,
                          "evaluation_rows": [{"condition": "normal", "ce": .5 + coefficient} for _ in range(4)]})
    for rho in (.01, .05, .1, .25):
        cases.append({"case": {"kind": "perturbation", "rho": rho}, "qa_pass": rho < .1,
                      "full_dreamlite_path_executed": True})
    write(analysis / "functional_geometry.json", {"technical_pass": True, "cases": cases, "synthetic_test_fixture": True})
    result = plot.render(tmp_path, tmp_path / "figures", dpi=55)
    assert not result["errors"]
    assert all(row["status"] == "observed" for row in result["plots"].values())
    assert result["plots"]["B_common_pca_trajectories"]["xT"]["common_basis_fit_samples"] == 22
    assert result["plots"]["F_local_robustness"]["reoptimization_included"] is False


def test_modified_tensor_cannot_generate_observed_trajectory(tmp_path):
    folder = tmp_path / "runs" / "A2-s0"
    folder.mkdir(parents=True)
    path = folder / "x.npy"
    np.save(path, np.ones((1, 1, 2, 2), np.float32))
    record = {"path": str(path), "file_sha256": "0" * 64}
    write(folder / "trajectory_index.json", [{"step": step, "xT": record, "z": record} for step in plot.SNAPSHOTS])
    write(folder / "summary.json", {"seed": 0, "technical_pass": True, "optimizer_steps": 256,
                                     "trajectory_index_path": str(folder / "trajectory_index.json")})
    result = plot.render(tmp_path, tmp_path / "figures", dpi=45)
    assert result["plots"]["B_common_pca_trajectories"]["status"] == "artifact_error"
    assert result["errors"]
