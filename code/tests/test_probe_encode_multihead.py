from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orjson

from scripts.evaluation.prob.run_probe_encode_multihead import read_journal


class JournalTest(unittest.TestCase):
    def test_partial_last_commit_is_recomputed_for_all_heads(self):
        row = {
            "record_idx": 0,
            "token_pos": 128,
            "skipped": False,
            "probs": {"a": 0.1, "b": 0.9},
        }
        good = orjson.dumps(row) + b"\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "checkpoint.jsonl"
            p.write_bytes(good + b'{"record_idx":1,"probs":{"a":0.2')
            result = read_journal(p, ["a", "b"], {(0, 128), (1, 128)})
            self.assertEqual(set(result), {(0, 128)})
            self.assertEqual(p.read_bytes(), good)

    def test_missing_head_or_invalid_probability_rejected(self):
        for probs in [{"a": 0.1}, {"a": 0.1, "b": float("inf")}]:
            with self.subTest(probs=probs), tempfile.TemporaryDirectory() as d:
                p = Path(d) / "checkpoint.jsonl"
                # JSON infinity is rejected by the probability range check.
                import json

                p.write_text(
                    json.dumps(
                        {
                            "record_idx": 0,
                            "token_pos": 0,
                            "skipped": False,
                            "probs": probs,
                        }
                    )
                    + "\n"
                )
                with self.assertRaises((ValueError, orjson.JSONDecodeError)):
                    read_journal(p, ["a", "b"], {(0, 0)})

    def test_context_skip_is_a_completed_commit(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "checkpoint.jsonl"
            row = {"record_idx": 0, "token_pos": 0, "skipped": True}
            p.write_bytes(orjson.dumps(row) + b"\n")
            self.assertEqual(read_journal(p, ["a", "b"], {(0, 0)})[(0, 0)], row)

    def test_duplicate_commit_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "checkpoint.jsonl"
            row = {"record_idx": 0, "token_pos": 0, "skipped": True}
            p.write_bytes((orjson.dumps(row) + b"\n") * 2)
            with self.assertRaises(ValueError):
                read_journal(p, ["a"], {(0, 0)})

class SharedEncodingTest(unittest.TestCase):
    def test_encode_once_matches_training_and_resumes(self):
        import asyncio
        import json
        from types import SimpleNamespace
        from unittest.mock import patch

        import httpx
        import torch
        from safetensors.torch import save_file
        from transformers import Qwen3Config

        from cognitive_probe.probe_model.probe_model import Qwen3ForProbing
        from scripts.evaluation.prob import run_probe_encode_multihead as shared

        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input'
            source.mkdir()
            records = [
                {'input_ids': [1], 'output_ids': [4, 5, 4], 'example_id': 'a'},
                {'input_ids': [2, 3], 'output_ids': [4, 5, 4], 'example_id': 'b'},
            ]
            (source / 'generation_results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            (source / 'args.json').write_text('{"model_name": "tiny"}')
            token_map = root / 'tokens.json'
            token_map.write_text('{"4": "\\n"}')
            base = root / 'base'
            base.mkdir()
            models, paths = {}, {}
            embedding = torch.randn(32, 16)
            for name in ('a', 'b'):
                config = Qwen3Config(
                    vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                    head_dim=8, num_labels=1, tie_word_embeddings=False,
                    task_specific_params={
                        'pooling_mode': 'last_token', 'prob_num_layers': 2,
                        'prob_hidden_size': 16, 'normalize_hidden_states_before_head': True,
                        'use_logit_margin_feature': True, 'use_logit_entropy_feature': True,
                    },
                )
                model = Qwen3ForProbing(config).eval()
                with torch.no_grad():
                    model.model.get_input_embeddings().weight.copy_(embedding)
                models[name] = model
                paths[name] = str(root / name)
                model.export_score_head(paths[name])
            config.save_pretrained(base)
            # A different untied LM head must never be used to compute features.
            save_file({'model.embed_tokens.weight': embedding, 'lm_head.weight': embedding * 10}, base / 'model.safetensors')
            heads = root / 'heads.json'
            heads.write_text(json.dumps(paths))
            args = SimpleNamespace(
                input_dir=str(source), output_dir=str(root / 'out'), heads_json=str(heads),
                base_model_path=str(base), track_token_map=str(token_map), max_records=0,
                probe_point_mode='track_token', track_token_stride=1,
                verify_single_head=True, batch_size=2, device='cpu', sgl_url='http://test',
            )
            requested = []
            def hidden(ids):
                return torch.arange(16, dtype=torch.float32) * (sum(ids) / 20) + len(ids)
            def respond(request):
                ids = json.loads(request.content)['input_ids']
                requested.append(ids)
                if len(ids) > 4:
                    return httpx.Response(400, text="longer than the model's context length")
                return httpx.Response(200, json={'embedding': hidden(ids).tolist()})
            original_client = httpx.AsyncClient
            def client():
                return original_client(transport=httpx.MockTransport(respond))
            with patch.object(shared.httpx, 'AsyncClient', side_effect=client):
                asyncio.run(shared.evaluate(args))
            self.assertEqual(len(requested), 4)  # Four prefixes, not four per head.
            self.assertEqual(len({tuple(ids) for ids in requested}), 4)
            outputs = {}
            for name, model in models.items():
                path = root / 'out' / name / 'generation_results.probed.jsonl'
                outputs[name] = path.read_bytes()
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for source_row, row in zip(records, rows):
                    self.assertEqual(row['probs'][1], -1)
                    for pos in (0, 2):
                        ids = source_row['input_ids'] + source_row['output_ids'][:pos + 1]
                        if len(ids) > 4:
                            self.assertEqual(row['probs'][pos], -1)
                        else:
                            with torch.no_grad():
                                expected = model._score_from_state(hidden(ids).unsqueeze(0)).sigmoid().item()
                            self.assertAlmostEqual(row['probs'][pos], expected, places=6)
            # A completed run makes no service calls.
            requested.clear()
            with patch.object(shared.httpx, 'AsyncClient', side_effect=client):
                asyncio.run(shared.evaluate(args))
            self.assertEqual(requested, [])
            # Recompute only an interrupted final commit, including context skips.
            journal = root / 'out' / 'checkpoint.jsonl'
            lines = journal.read_bytes().splitlines(keepends=True)
            journal.write_bytes(b''.join(lines[:-1]) + lines[-1][:12])
            with patch.object(shared.httpx, 'AsyncClient', side_effect=client):
                asyncio.run(shared.evaluate(args))
            self.assertEqual(len(requested), 1)
            for name in models:
                self.assertEqual((root / 'out' / name / 'generation_results.probed.jsonl').read_bytes(), outputs[name])
            # Changed head weights must not reuse the journal.
            weights = Path(paths['a']) / 'model.safetensors'
            weights.write_bytes(weights.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
                asyncio.run(shared.evaluate(args))

    def test_incompatible_features_are_rejected(self):
        from types import SimpleNamespace
        from scripts.evaluation.prob.run_probe_encode_multihead import check_compatible
        def head(normalized):
            return SimpleNamespace(
                pooling_mode='last_token', use_hidden_state_feature=True,
                normalize_hidden_states_before_head=normalized,
                use_logit_margin_feature=False, use_logit_entropy_feature=False,
                logit_feature_dim=0,
                config=SimpleNamespace(hidden_size=16, vocab_size=32, model_type='qwen3'),
            )
        with self.assertRaisesRegex(ValueError, 'incompatible feature'):
            check_compatible({'a': head(True), 'b': head(False)},
                             {'hidden_size': 16, 'vocab_size': 32, 'model_type': 'qwen3'})


if __name__ == "__main__":
    unittest.main()
