from __future__ import annotations

import unittest

from cognitive_probe.evaluation.probe_prune.online import simulate_online_group
from cognitive_probe.evaluation.probe_prune.strategies import build_strategy

class StrictOnlineReplayTest(unittest.TestCase):
    def test_group_quorum_snapshot_floor_keeps_short_scheduled_sequence(self) -> None:
        records = [
            {"output_ids": [10, 11, 12, 13], "probs": [0.9, 0.1, 0.1, 0.8]},
            {"output_ids": [20, 21, 22, 23, 24, 25, 26], "probs": [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1]},
        ]
        generation_strategy = build_strategy(
            strategy_name="consecutive_below",
            strategy_args={"threshold": 0.15, "window_size": 2, "min_keep_tokens": 0},
        )

        outcomes = simulate_online_group(
            records=records,
            generation_strategy=generation_strategy,
            input_strategy_name="group_quorum_snapshot_floor",
            input_strategy_args={"min_pruned_count": 2, "quantile": 1.0},
        )

        self.assertEqual(len(outcomes), 2)

        short_outcome = outcomes[0]
        self.assertFalse(short_outcome.decision.should_prune)
        self.assertEqual(short_outcome.final_state, "finished")
        self.assertEqual(short_outcome.candidate_count, 1)
        self.assertIn("sequence ended before scheduled online stop", short_outcome.decision.reason)

        long_outcome = outcomes[1]
        self.assertTrue(long_outcome.decision.should_prune)
        self.assertEqual(long_outcome.decision.cut_index, 7)
        self.assertEqual(long_outcome.final_state, "committed")


if __name__ == "__main__":
    unittest.main()
