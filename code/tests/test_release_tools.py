import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class ReleaseToolsTest(unittest.TestCase):
    def test_added_token_is_included(self):
        module = load('token_map_release_test', 'scripts/evaluation/prob/build_all_track_tokens.py')
        class FakeTokenizer:
            vocab_size = 1
            def get_vocab(self): return {'base': 0, 'added': 7}
            def decode(self, ids, **kwargs): return '\n' if ids == [7] else 'x'
        self.assertEqual(module.build_track_tokens(FakeTokenizer(), ['\n']), {7: '\n'})

    def test_scoring_batch_reports_partial_failure(self):
        module = load('score_release_test', 'scripts/score_batch_file.py')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(FileNotFoundError):
                module.process_batch_files(directory, score_only=True)
            (path / 'bad.json').write_text('invalid json')
            (path / 'good.json').write_text(json.dumps({'id': 'q', 'answer': '42', 'branches': [{'text': '\\boxed{42}'}]}))
            for workers in (1, 2):
                with self.subTest(workers=workers), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'Scoring failed for 1 file'):
                        module.process_batch_files(directory, score_only=True, max_file_workers=workers)
                self.assertEqual(json.loads((path / 'good.json').read_text())['branches'][0]['score'], 1.0)
