from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/evaluation/prob/check_sglang_qwen3_encode_patch.py"
spec = importlib.util.spec_from_file_location("encode_patch", path)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


class SGLangPatchTests(unittest.TestCase):
    def test_targeted_idempotent_patch_and_backup(self):
        source = '''class Other:
    def __init__(self):
        self.pooler = Pooler(normalize=True)
class Qwen3ForCausalLM:
    def __init__(self):
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
'''
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "qwen3.py"
            file.write_text(source)
            self.assertTrue(patch.apply_patch(file, "0.5.9"))
            self.assertFalse(patch._find_qwen3_pooler_normalize(file)[0])
            self.assertIn("self.pooler = Pooler(normalize=True)", file.read_text())
            self.assertEqual(file.with_suffix(".py.probe-original").read_text(), source)
            self.assertFalse(patch.apply_patch(file, "0.5.9"))

    def test_unknown_version_or_layout_is_not_modified(self):
        sources = [
            ("0.6.0", "Pooler(normalize=True)"),
            ("0.5.9", "Pooler()"),
            ("0.5.9", "UnknownPooler(normalize=True)"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "qwen3.py"
            for version, expression in sources:
                source = f"class Qwen3ForCausalLM:\n    def __init__(self):\n        self.pooler = {expression}\n"
                file.write_text(source)
                with self.assertRaises(RuntimeError):
                    patch.apply_patch(file, version)
                self.assertEqual(file.read_text(), source)
