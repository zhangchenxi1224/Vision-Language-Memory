import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("multiq_launch", ROOT / "scripts/inspire/run_r11_open_eos_multiquestion_h200x4.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def panel():
    return json.loads((ROOT / "configs/experiments/r11_open_eos_multiquestion_targets.json").read_text())


def test_two_lanes_cover_panel_once_and_share_no_output(tmp_path):
    config = panel()
    commands = launcher.lane_commands(config, Path("targets.json"), tmp_path, "python")
    assert len(commands) == 2
    indices = [i for lane, _ in commands for i in lane["target_indices"]]
    assert len(indices) == len(set(indices)) == 16
    assert sorted(indices) == list(range(16))
    outputs = [command[command.index("--output-dir") + 1] for _, command in commands]
    assert len(set(outputs)) == 2
    assert all(len(lane["target_indices"]) * len(config["seeds"]) * len(config["arms"]) == 64 for lane, _ in commands)


def test_duplicate_or_missing_question_stops_launch(tmp_path):
    config = panel()
    config["lanes"][1]["target_indices"][0] = 0
    with pytest.raises(ValueError, match="exactly once"):
        launcher.lane_commands(config, Path("targets.json"), tmp_path)


def test_overlapping_physical_gpus_stop_launch(tmp_path):
    config = panel()
    config["lanes"][1]["gpu_pair"] = [1, 2]
    with pytest.raises(ValueError, match="disjoint"):
        launcher.lane_commands(config, Path("targets.json"), tmp_path)
