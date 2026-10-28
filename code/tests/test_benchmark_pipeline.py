"""CPU artifact-contract check using a fake SGLang HTTP service."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config
from cognitive_probe.probe_model.probe_model import Qwen3ForProbing

ROOT = Path(__file__).resolve().parents[1]

class FakeSGLang(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if not request.get('input_ids'):
            self.send_error(400)
            return
        if self.path == '/generate':
            result = {'text': '\\boxed{42}\n\n', 'output_ids': [3, 4, 3, 4], 'meta_info': {'finish_reason': {'type': 'stop'}}}
        elif self.path == '/encode':
            result = {'embedding': [0.0] * 16}
        else:
            self.send_error(404)
            return
        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

class BenchmarkPipelineTest(unittest.TestCase):
    def test_generate_encode_prune_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, '\\boxed{42}': 3, '\n': 4}, unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
            fast.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
            fast.save_pretrained(root / 'tokenizer')
            model = Qwen3ForProbing(Qwen3Config(vocab_size=5, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8, task_specific_params={'pooling_mode': 'last_token', 'segment_mean_window_tokens': 1}))
            with torch.no_grad():
                for parameter in model.score.parameters(): parameter.zero_()
            model.export_score_head(str(root / 'head'))
            (root / 'tokens.json').write_text(json.dumps({'4': '\n'}))
            (root / 'questions.jsonl').write_text(json.dumps({'id': 'q1', 'prompt': [{'role': 'user', 'content': 'Compute 6 times 7.'}], 'solution': '42'}) + '\n')
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
            def run(script, *args):
                result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=root, env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-4000:])
            server = ThreadingHTTPServer(('127.0.0.1', 0), FakeSGLang)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f'http://127.0.0.1:{server.server_port}'
                run('scripts/evaluation/generation/run_generation_api.py', '--api-url', url, '--model-name', 'fake', '--tokenizer-path', root / 'tokenizer', '--data', root / 'questions.jsonl', '--output', root / 'runs', '--num-generation', 2, '--max-length', 8, '--eval-k', '1,2')
                run_dir = next((root / 'runs').rglob('generation_results.jsonl')).parent
                self.assertTrue(list(run_dir.glob('eval_report_v2_*.json')))
                run('scripts/evaluation/prob/run_probe_encode.py', '--input-dir', run_dir, '--output-dir', root / 'probed', '--track-token-map', root / 'tokens.json', '--head-model-path', root / 'head', '--sgl-url', url, '--device', 'cpu', '--sgl-max-workers', 2, '--track-token-stride', 1)
                scored = next((root / 'probed').glob('*.probed.jsonl'))
                rows = [json.loads(line) for line in scored.read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                for row in rows: self.assertEqual(row['probs'], [-1.0, 0.5, -1.0, 0.5])
                run('scripts/evaluation/postprocess/run_probe_prune_online.py', '--input-dir', root / 'probed', '--output-dir', root / 'pruned', '--strategy', 'consecutive_below', '--threshold', '0.6', '--window-size', '1', '--tokenizer-path', root / 'tokenizer')
                run('scripts/evaluation/analysis/run_eval.py', '--dir', root / 'pruned', '-k', '1,2')
                self.assertTrue((root / 'pruned/final_outputs.jsonl').exists())
                self.assertTrue(list((root / 'pruned').glob('eval_report_v2_*.json')))
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()
