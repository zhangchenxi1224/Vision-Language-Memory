from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.eval.qwen_history_r4 import synthetic_queries  # noqa: E402
from vision_memory.data.r5_baseline_lockbox import build_transition32  # noqa: E402
from vision_memory.data.schema import write_jsonl  # noqa: E402


class R5HistoryEvaluatorTest(unittest.TestCase):
    def test_valid_r5_transition32_is_checked_before_query_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "transition32.jsonl"
            write_jsonl(path, build_transition32())
            queries = list(synthetic_queries(path, limit=None))
        self.assertEqual(len(queries), 48)
        self.assertEqual(sum(query.metadata["probe_role"] == "delayed" for query in queries), 32)

    def test_tampered_r5_pair_map_is_rejected_before_query_expansion(self) -> None:
        episodes = list(build_transition32())
        first = episodes[0]
        cross_scope = next(
            episode
            for episode in episodes
            if episode.semantic_group_id != first.semantic_group_id
            and episode.turns[-1].query.target != first.turns[-1].query.target
        )
        episodes[0] = replace(
            first,
            counterfactual_episode_id=cross_scope.episode_id,
            pair_id="r5-tampered-cross-scope-pair",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tampered.jsonl"
            write_jsonl(path, tuple(episodes))
            with self.assertRaisesRegex(ValueError, "not reciprocal"):
                list(synthetic_queries(path, limit=None))


if __name__ == "__main__":
    unittest.main()
