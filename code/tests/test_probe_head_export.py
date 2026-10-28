from __future__ import annotations

import tempfile
import unittest
import importlib.util
import json
from pathlib import Path

import torch
from transformers import Qwen3Config
from safetensors.torch import save_file

from cognitive_probe.probe_model.probe_model import Qwen3ForProbing, SingleProbeHead


class ProbeExportTests(unittest.TestCase):
    def test_encode_projection_matches_training_for_untied_weights(self):
        path = Path(__file__).resolve().parents[1] / "scripts/evaluation/prob/run_probe_encode.py"
        spec = importlib.util.spec_from_file_location("probe_encode_projection", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        embedding = torch.randn(32, 16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"tie_word_embeddings": False}))
            for sharded in (False, True):
                filename = "shard.safetensors" if sharded else "model.safetensors"
                save_file({"model.embed_tokens.weight": embedding, "lm_head.weight": embedding + 10}, root / filename)
                if sharded:
                    (root / "model.safetensors.index.json").write_text(json.dumps({
                        "weight_map": {"model.embed_tokens.weight": filename, "lm_head.weight": filename}
                    }))
                torch.testing.assert_close(module._extract_lm_head_weight(directory), embedding)

    def test_export_matches_training_head_with_raw_hidden_states(self):
        torch.manual_seed(42)
        for logit_features in (False, True):
            with self.subTest(logit_features=logit_features):
                config = Qwen3Config(
                    vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                    head_dim=8, num_labels=1,
                    task_specific_params={
                        "pooling_mode": "last_token", "segment_mean_window_tokens": 1,
                        "prob_num_layers": 2, "prob_hidden_size": 16,
                        "use_logit_margin_feature": logit_features,
                        "use_logit_entropy_feature": logit_features,
                        "normalize_hidden_states_before_head": True,
                    },
                )
                model = Qwen3ForProbing(config).eval()
                hidden = torch.randn(2, 16) * 3
                with tempfile.TemporaryDirectory() as directory, torch.no_grad():
                    expected = model._score_from_state(hidden).sigmoid()
                    model.export_score_head(directory)
                    head = SingleProbeHead.from_pretrained(directory, device="cpu").eval()
                    actual = head(hidden.unsqueeze(1), lm_head_weight=model.model.get_input_embeddings().weight)
                    torch.testing.assert_close(actual, expected)
