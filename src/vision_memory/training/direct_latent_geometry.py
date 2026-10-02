"""Prospective direct-z initialization panel; no Reader outcomes select members."""
from __future__ import annotations
import math
import numpy as np

DISTRIBUTIONS = ("gaussian", "uniform", "sphere", "rademacher", "heavy_tail")
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256)
INSTRUCTIONS = "Use the memory image to answer.\nAnswer with a short phrase only."


def question_prompts():
    prefix = "R3 Train Standard Templates 03: "
    questions = {
        "original_open": "At this later check, what is the current music preference for the indigo desk train 001123?",
        "paraphrase_1": "At this later check, which music does the indigo desk train 001123 currently prefer?",
        "paraphrase_2": "What music does the indigo desk train 001123 prefer now, at this later check?",
        "paraphrase_3": "For the indigo desk train 001123, what is the music preference currently in effect at this later check?",
        "paraphrase_4": "Based on the stored memory, name the current preferred music for the indigo desk train 001123 at this later check.",
    }
    return {name: prefix + question + "\n" + INSTRUCTIONS for name, question in questions.items()}


def panel():
    rows = []
    def add(distribution, seed, scale, study):
        rows.append({"run_id": f"direct-{distribution}-s{seed:02d}-a{scale:g}",
                     "arm": "open_gold_eos", "distribution": distribution, "seed": seed,
                     "scale": scale, "study": study, "init_id": f"{distribution}-{seed}-{scale:g}",
                     "init_seed": seed, "included_in_random_start_statistics": True})
    # Interleave distributions so both device lanes cover the same study early.
    for seed in range(8):
        for distribution in DISTRIBUTIONS:
            add(distribution, seed, 1., "distribution")
    for seed in range(8):
        for scale in (.25, .5, 2., 4.):
            add("gaussian", seed, scale, "scale")
    for seed in range(8, 32):
        add("gaussian", seed, 1., "gaussian_density")
    assert len(rows) == 96 and len({r["run_id"] for r in rows}) == 96
    return rows


def draw(shape, distribution, seed):
    if distribution not in DISTRIBUTIONS or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid prospective distribution or seed")
    rng = np.random.Generator(np.random.PCG64(seed))
    if distribution == "gaussian":
        value = rng.standard_normal(shape)
    elif distribution == "uniform":
        value = rng.uniform(-math.sqrt(3), math.sqrt(3), shape)
    elif distribution == "rademacher":
        value = rng.integers(0, 2, shape).astype(np.float64) * 2 - 1
    elif distribution == "sphere":
        value = rng.standard_normal(shape)
        value /= np.sqrt(np.mean(value ** 2))
    else:
        value = rng.standard_t(5, shape) * math.sqrt(3 / 5)
    return value


def initial_array(reference, spec):
    reference = np.asarray(reference, dtype=np.float64)
    scale = float(spec["scale"])
    if not np.isfinite(reference).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Nonfinite reference or invalid scale")
    rms = float(np.sqrt(np.mean(reference ** 2)))
    if rms <= 0:
        raise ValueError("A nonzero model-space blank reference is required")
    noise = draw(reference.shape, spec["distribution"], spec["seed"])
    value = (reference + .1 * scale * rms * noise).astype(np.float32)
    return value, {"reference_rms": rms, "relative_scale": .1 * scale,
                   "epsilon_mean": float(noise.mean()), "epsilon_rms": float(np.sqrt(np.mean(noise ** 2))),
                   "epsilon_kurtosis": float(np.mean((noise-noise.mean())**4)/np.var(noise)**2)}


def study_members(rows, study, label):
    if study == "distribution":
        return [r for r in rows if r["seed"] < 8 and r["scale"] == 1 and r["distribution"] == label]
    if study == "scale":
        return [r for r in rows if r["seed"] < 8 and r["distribution"] == "gaussian" and r["scale"] == label]
    return [r for r in rows if r["distribution"] == "gaussian" and r["scale"] == 1]


def spectrum(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return {"sample_count": len(values), "rank_cap": max(0, len(values)-1), "r95": None}
    centered = values - values.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered @ centered.T)[::-1].clip(min=0)
    total = float(eigenvalues.sum())
    explained = np.cumsum(eigenvalues)/total if total else np.zeros_like(eigenvalues)
    return {"sample_count": len(values), "rank_cap": min(len(values)-1, values.shape[1]),
            "r95": int(np.searchsorted(explained, .95)+1) if total else 0,
            "cumulative_variance": explained.tolist(), "total_centered_energy": total}
