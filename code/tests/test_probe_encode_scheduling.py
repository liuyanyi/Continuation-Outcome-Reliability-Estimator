import runpy
import unittest
from pathlib import Path

_iter_probe_batches = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/prob/run_probe_encode.py"
    )
)["_iter_probe_batches"]


class ProbeBatchSchedulingTest(unittest.TestCase):
    def test_resumed_unequal_rollouts_preserve_points_and_prefix_order(self):
        # Remaining work after a checkpoint: unequal lengths and gaps are valid.
        remaining = {0: [384, 640, 768], 3: [128], 8: [0, 256], 11: [512, 640]}
        tasks = [
            {"record_idx": rid, "token_pos": pos}
            for rid, positions in remaining.items()
            for pos in positions
        ]
        batches = list(
            _iter_probe_batches(tasks, batch_size=2, interleave_records=True)
        )
        seen = {rid: [] for rid in remaining}
        for batch in batches:
            self.assertLessEqual(len(batch), 2)
            self.assertEqual(len({t["record_idx"] for t in batch}), len(batch))
            for task in batch:
                seen[task["record_idx"]].append(task["token_pos"])
        self.assertEqual(seen, remaining)
        self.assertEqual(sum(map(len, batches)), len(tasks))
        # The active window retains unfinished rollouts instead of sweeping all
        # records, which would evict their prefix caches on long datasets.
        self.assertEqual([t["record_idx"] for t in batches[0]], [0, 3])
        self.assertEqual([t["record_idx"] for t in batches[1]], [0, 8])

    def test_default_order_is_unchanged(self):
        tasks = [{"record_idx": 0, "token_pos": p} for p in [0, 128, 256]]
        self.assertEqual(
            list(_iter_probe_batches(tasks, batch_size=2, interleave_records=False)),
            [tasks[:2], tasks[2:]],
        )

    def test_empty_and_single_record(self):
        self.assertEqual(
            list(_iter_probe_batches([], batch_size=8, interleave_records=True)), []
        )
        tasks = [{"record_idx": 7, "token_pos": p} for p in [0, 128, 256]]
        self.assertEqual(
            list(_iter_probe_batches(tasks, batch_size=8, interleave_records=True)),
            [[task] for task in tasks],
        )
        with self.assertRaises(ValueError):
            list(_iter_probe_batches(tasks, batch_size=0, interleave_records=True))


if __name__ == "__main__":
    unittest.main()
