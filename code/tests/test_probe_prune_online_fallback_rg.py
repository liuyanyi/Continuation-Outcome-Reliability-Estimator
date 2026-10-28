from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESS_ROOT = PROJECT_ROOT / "scripts" / "evaluation" / "postprocess"
if str(POSTPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(POSTPROCESS_ROOT))

from run_probe_prune_online_fallback_rg import (  # noqa: E402
    _build_fallback_regeneration_config,
    _compute_resume_index,
)


class FallbackRollbackConfigTest(unittest.TestCase):
    def test_default_uses_fixed_rollback_tokens(self) -> None:
        config = _build_fallback_regeneration_config(
            original_args={},
            strategy_args={"api_url": "http://localhost:3000", "model_name": "demo"},
        )

        self.assertEqual(config.rollback_tokens, 32)
        self.assertIsNone(config.rollback_ratio)

    def test_ratio_mode_rolls_back_fraction_of_stop_index(self) -> None:
        config = _build_fallback_regeneration_config(
            original_args={},
            strategy_args={
                "api_url": "http://localhost:3000",
                "model_name": "demo",
                "rollback_ratio": 0.5,
            },
        )

        resume_index, rollback_applied_tokens, rollback_mode = _compute_resume_index(
            bounded_stop_index=7,
            config=config,
        )

        self.assertEqual((resume_index, rollback_applied_tokens, rollback_mode), (3, 4, "ratio"))

    def test_rejects_tokens_and_ratio_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one of rollback_tokens or rollback_ratio"):
            _build_fallback_regeneration_config(
                original_args={},
                strategy_args={
                    "api_url": "http://localhost:3000",
                    "model_name": "demo",
                    "rollback_tokens": 32,
                    "rollback_ratio": 0.5,
                },
            )


if __name__ == "__main__":
    unittest.main()
