import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tree_gen_config", ROOT / "scripts/run_tree_gen_batch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TreeConfigTests(unittest.TestCase):
    def test_presets_and_cli_override(self):
        tiny = module.parse_args(["--config", str(ROOT / "configs/tree_gen/tiny.json")])
        self.assertEqual((tiny.max_total_branches, tiny.max_new_tokens), (8, 512))
        self.assertFalse(tiny.disable_refill)
        large = module.parse_args(["--config", str(ROOT / "configs/tree_gen/qwen3_4b.json"), "--max-total-branches", "16"])
        self.assertEqual((large.max_total_branches, large.max_new_tokens), (16, 16000))
        self.assertEqual(module.parse_args([]).max_total_branches, 512)

    def test_boolean_overrides_in_both_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"disable_refill": True, "disable_timeout": True}))
            args = module.parse_args(["--config", str(path), "--enable-refill", "--enable-timeout"])
            self.assertFalse(args.disable_refill)
            self.assertFalse(args.disable_timeout)
        args = module.parse_args(["--config", str(ROOT / "configs/tree_gen/tiny.json"), "--disable-refill"])
        self.assertTrue(args.disable_refill)

    def test_invalid_config_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for config in ([1], {"max_total_branch": 8}, {"initial": True}, {"disable_refill": "false"}, {"batch_concurrent_tasks": 0}, {"expand_prob": 2}, {"top_p": float("nan")}):
                path.write_text(json.dumps(config))
                with self.subTest(config=config), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        module.parse_args(["--config", str(path)])
                    self.assertEqual(error.exception.code, 2)
