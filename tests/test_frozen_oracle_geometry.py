import numpy as np
import pytest

from vision_memory.training.frozen_oracle_geometry import (
    build_manifest, determinism_gate, geometry_statistics, make_initial_array, wilson,
)


def test_sphere_does_not_erase_gaussian_radial_control():
    g = make_initial_array((1, 4, 128, 128), "gaussian", 0)
    s = make_initial_array(g.shape, "sphere", 0)
    assert np.array_equal(g, make_initial_array(g.shape, "gaussian", 0))
    assert not np.array_equal(g, s)
    assert np.sqrt(np.mean(s.astype(float) ** 2)) == pytest.approx(1, abs=1e-8)
    assert np.dot(g.ravel(), s.ravel()) / np.linalg.norm(g) / np.linalg.norm(s) > .99999
    assert np.array_equal(2 * g, make_initial_array(g.shape, "gaussian", 0, 2))


def test_design_pairing_and_independent_counts():
    m = build_manifest()
    assert len(m["runs"]) == 158
    assert len({r["run_id"] for r in m["runs"]}) == 158
    assert {k: len(v) for k, v in m["study_memberships"].items()} == {
        "A1": 6, "A2": 32, "A3": 40, "A4": 40, "A9": 64}
    assert len(set(m["study_memberships"]["A2"]) & set(m["study_memberships"]["A9"])) == 8


def test_reproducibility_gate_rejects_missing_and_mismatched_evidence():
    keys = ("initial_xT_sha256", "optimized_xT_sha256", "endpoint_z_sha256",
            "loss_trajectory_sha256", "gradient_trajectory_sha256")
    rows = [{"target_index": target, "technical_pass": True, **{k: str(target) for k in keys}}
            for target in (0, 1) for _ in range(3)]
    assert determinism_gate(rows)["passed"]
    rows[-1]["gradient_trajectory_sha256"] = "different"
    assert not determinism_gate(rows)["passed"]
    assert not determinism_gate([])["passed"]


def test_rank_reports_sample_limit_and_random_control():
    x = np.random.default_rng(1).normal(size=(8, 128))
    stats = geometry_statistics(x, [0] * 4 + [1] * 4, list(range(4)) * 2)
    assert stats["pca"]["sample_rank_limit"] == 7
    assert stats["pca"]["r95"] <= 7
    assert stats["pca"]["random_null_r95"] <= 7
    assert stats["between_same_seed_rmse"] > 0
    assert wilson(0, 8)[0] == 0
    assert wilson(8, 8)[1] == pytest.approx(1)


@pytest.mark.parametrize("distribution", ["gaussian", "uniform", "rademacher", "sphere", "heavy_tail"])
def test_population_moments_and_finite_values(distribution):
    x = make_initial_array((65536,), distribution, 8)
    assert abs(float(x.mean())) < .03
    assert .90 < float(np.mean(x * x)) < 1.10
    assert np.isfinite(x).all()
