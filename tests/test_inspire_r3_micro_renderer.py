from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSPIRE = ROOT / "scripts" / "inspire"
sys.path.insert(0, str(INSPIRE))

from materialize_r3_dag import _load_micro_command_contract  # noqa: E402
from r3_dag_contract import atomic_json  # noqa: E402
from render_r3_micro_contract import RenderInputs, render_contract  # noqa: E402


READER_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DREAMLITE_REVISION = "6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4"


def _commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture_inputs(tmp_path: Path, suite: str, regime: str) -> RenderInputs:
    models = tmp_path / "models"
    dreamlite = models / "DreamLite-mobile"
    reader = models / "Qwen3-VL-4B-Instruct"
    dreamlite.mkdir(parents=True)
    reader.mkdir()
    (dreamlite / ".locked_revision").write_text(DREAMLITE_REVISION + "\n", encoding="utf-8")
    (reader / ".locked_revision").write_text(READER_REVISION + "\n", encoding="utf-8")

    teacher_cache = None
    teacher_calibration = None
    if regime == "teacher_assisted":
        teacher_cache = tmp_path / "teacher-cache"
        teacher_cache.mkdir()
        (teacher_cache / "manifest.json").write_text("{}\n", encoding="utf-8")
        (teacher_cache / "transitions.jsonl").write_text("{}\n", encoding="utf-8")
        teacher_calibration = tmp_path / "teacher-calibration.json"
        teacher_calibration.write_text("{}\n", encoding="utf-8")

    return RenderInputs(
        repo=ROOT,
        python=Path(sys.executable),
        model_root=models,
        run_root=tmp_path / "immutable-micro-run",
        suite=suite,
        training_regime=regime,
        train=ROOT / "data" / "r3_micro_v1" / f"{suite}_train.jsonl",
        gate=ROOT / "data" / "r3_micro_v1" / f"{suite}_gate.jsonl",
        expected_commit=_commit(),
        reader_revision=READER_REVISION,
        dreamlite_revision=DREAMLITE_REVISION,
        teacher_cache=teacher_cache,
        teacher_calibration=teacher_calibration,
    )


@pytest.mark.parametrize(
    ("suite", "regime", "shape", "arm_ids", "final_script"),
    [
        ("set8", "qa_only", "single", ["A"], "score_r3_micro.py"),
        (
            "transition16",
            "qa_only",
            "paired-replica",
            ["A", "B"],
            "validate_r3_micro_replication.py",
        ),
        (
            "set8",
            "teacher_assisted",
            "teacher-control-composite",
            ["correct", "shuffled", "random"],
            "score_r3_teacher_attribution.py",
        ),
        (
            "transition16",
            "teacher_assisted",
            "paired-replica",
            ["A", "B"],
            "validate_r3_micro_replication.py",
        ),
    ],
)
def test_renderer_produces_loader_valid_arm_aware_contracts(
    tmp_path: Path,
    suite: str,
    regime: str,
    shape: str,
    arm_ids: list[str],
    final_script: str,
) -> None:
    inputs = _fixture_inputs(tmp_path, suite, regime)
    contract = render_contract(inputs)
    path = tmp_path / "contract.json"
    atomic_json(path, contract)

    loaded = _load_micro_command_contract(path)
    assert loaded == contract
    assert loaded["execution_shape"] == shape
    assert [arm["arm_id"] for arm in loaded["arms"]] == arm_ids
    assert Path(loaded["commands"][-1][1]).name == final_script
    assert all(command[0] == str(inputs.python.absolute()) for command in loaded["commands"])
    assert len({output["path"] for output in loaded["outputs"]}) == len(loaded["outputs"])
    assert len({output["label"] for output in loaded["outputs"]}) == len(loaded["outputs"])

    evaluation_commands = [
        command for command in loaded["commands"] if Path(command[1]).name == "dreamlite_mcq.py"
    ]
    assert len(evaluation_commands) == len(arm_ids)
    for command in evaluation_commands:
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        expected_step = (512 if regime == "qa_only" else 256) * (8 if suite == "set8" else 16) // 8
        assert checkpoint.name == f"checkpoint-{expected_step:06d}.pt"
        assert checkpoint.name != "last.pt"

    if regime == "qa_only":
        assert loaded["teacher_calibration_binding"] is None
        assert all("--teacher-manifest" not in command for command in loaded["commands"])
    else:
        assert loaded["teacher_calibration_binding"]["suite"] == suite
        distill_commands = [
            command
            for command in loaded["commands"]
            if Path(command[1]).name == "dreamlite_episode.py"
            and command[command.index("--objective-stage") + 1] == "distill"
        ]
        assert len(distill_commands) == len(arm_ids)
        assert all("--teacher-calibration" in command for command in distill_commands)


def test_renderer_is_deterministic_and_keeps_all_outputs_under_run_root(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path, "transition16", "qa_only")
    first = render_contract(inputs)
    second = render_contract(inputs)
    assert first == second
    run_root = inputs.run_root.resolve()
    assert all(Path(output["path"]).resolve().is_relative_to(run_root) for output in first["outputs"])


def test_renderer_rejects_teacher_cache_on_qa_only(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path, "set8", "qa_only")
    cache = tmp_path / "unexpected-teacher"
    cache.mkdir()
    with pytest.raises(ValueError, match="qa_only rendering forbids"):
        render_contract(RenderInputs(**{**inputs.__dict__, "teacher_cache": cache}))
