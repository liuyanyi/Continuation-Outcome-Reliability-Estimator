from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from cognitive_probe.evaluation.refrain import build_step_ranges, token_char_offsets
from cognitive_probe.evaluation.stats import compute_per_problem_metrics_from_run
from cognitive_probe.evaluation.v2 import (
    compute_eval_report, load_final_outputs, load_generation_segments,
    materialize_v2_run_from_rows, migrate_legacy_run,
)

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / 'scripts/evaluation/analysis' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RefrainMappingTests(unittest.TestCase):
    def setUp(self):
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        backend = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>')
        self.tokenizer.add_tokens(['<added>'])

    def test_split_unicode_added_token_and_hidden_eos(self):
        text = '想法。\n\nAnswer <added> 好'
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids.append(self.tokenizer.eos_token_id)
        steps = build_step_ranges(response_text=text, output_ids=ids, tokenizer=self.tokenizer)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[-1].token_end, len(ids) - 1)
        for step in steps:
            decoded = self.tokenizer.decode(ids[step.token_start:step.token_end])
            self.assertIn(step.text, decoded)
        offsets = token_char_offsets(self.tokenizer, response_text=text, output_ids=ids)
        self.assertEqual(offsets[:3], [(0, 1)] * 3)
        self.assertEqual(offsets[-1], (len(text), len(text)))

    def test_truncated_utf8(self):
        ids = self.tokenizer.encode('好', add_special_tokens=False)[:2]
        text = self.tokenizer.decode(ids)
        self.assertEqual(text, '\ufffd')
        steps = build_step_ranges(response_text=text, output_ids=ids, tokenizer=self.tokenizer)
        self.assertEqual(steps[0].token_end, 2)

    def test_non_bytelevel_exact_special_alignment_and_refusal(self):
        class ExactTokenizer:
            all_special_ids = [99]
            def __call__(self, *args, **kwargs):
                return {'input_ids': [1, 2], 'offset_mapping': [(0, 1), (1, 2)]}
        tokenizer = ExactTokenizer()
        self.assertEqual(token_char_offsets(tokenizer, response_text='ab', output_ids=[1, 2, 99]),
                         [(0, 1), (1, 2), (2, 2)])
        with self.assertRaisesRegex(ValueError, 'Cannot exactly align'):
            token_char_offsets(tokenizer, response_text='ab', output_ids=[3])


class EvaluationFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lcb = load_script('run_eval_livebench')
        cls.lcb_stats = load_script('run_livebench_bootstrap_stats')

    def make_run(self, root, rows):
        args = {'probe_prune': {'strategy': 'refrain'}}
        (root / 'args.json').write_text(json.dumps(args))
        (root / 'generation_results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        migrate_legacy_run(root, force=True)

    def row(self, decision='keep', prefix=None):
        row = {'example_id': 'toy', 'answer': '1', 'response': r'\boxed{1}',
               'output_ids': [1, 2, 3], 'prune_meta': {'decision': decision, 'original_len': 200}}
        if prefix is not None:
            row['refrain_postprocess'] = {'decision': 'continue_with_direct_answer',
                                         'kept_prefix_tokens': prefix}
        return row

    def test_pruned_correct_answer_is_abstention(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            self.make_run(root, [self.row('prune')])
            report = compute_eval_report(root, 1)
            stats = compute_per_problem_metrics_from_run(root, 1)
            self.assertEqual((report.mean_at_k, report.best_score_mean, report.maj_at_k), (0, 0, 0))
            for metric in ('mean', 'pass', 'maj'):
                self.assertEqual(stats[metric].tolist(), [0])
            self.assertEqual(report.billed_output_tokens_total, 3)

    def test_continuation_cost_migration_and_all_consumers(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            self.make_run(root, [self.row('prune_then_regenerate', 100), self.row()])
            for force in (False, True, True):
                migrate_legacy_run(root, force=force)
                report = compute_eval_report(root, 2)
                self.assertEqual(report.billed_output_tokens_total, 106)
                self.assertEqual(report.final_visible_tokens_total, 106)
                self.assertEqual(report.request_count_total, 3)
                self.assertEqual(report.mean_at_k, 1)
                self.assertEqual(len(load_generation_segments(root)), 3)
                stats = compute_per_problem_metrics_from_run(root, 2)
                self.assertEqual(stats['tok'].tolist(), [53])
                self.assertEqual(stats['mean'].tolist(), [1])
            outputs = [o.to_dict() for o in load_final_outputs(root)]
            self.assertFalse(any(o['was_pruned'] for o in outputs))
            self.check_lcb(root, outputs, {'toy__0': self.judge(), 'toy__1': self.judge()}, 1, 53)

    @staticmethod
    def judge():
        return {'passed': True, 'test_pass_vector': [True], 'behavior_signature': '1:1'}

    def check_lcb(self, root, outputs, judgments, expected_mean, expected_tokens):
        report = self.lcb.compute_report(run_dir=root, ks=[len(outputs)], grouped_outputs={'toy': outputs},
                                        targets_by_sample={}, judge_results=judgments)
        k = len(outputs)
        self.assertEqual(report['mean_at_k'][str(k)], expected_mean)
        self.assertEqual(report['lcb_pass_at_1'], expected_mean)
        self.assertEqual(report['avg_final_visible_tokens'], expected_tokens)
        (root / 'livebench_eval_report.json').write_text(json.dumps(report))
        (root / 'livebench_judge_results.jsonl').write_text(''.join(
            json.dumps(dict(result, output_id=output_id)) + '\n' for output_id, result in judgments.items()))
        stats = self.lcb_stats.compute_per_problem_metrics(root, k)
        self.assertEqual(stats['per_problem']['mean'].tolist(), [expected_mean])
        self.assertEqual(stats['per_problem']['tok'].tolist(), [expected_tokens])
        self.assertTrue(stats['livebench_alignment']['passed'])
        return report

    def test_lcb_abstention_without_judge_and_fixed_denominator(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            self.make_run(root, [self.row('prune'), self.row()])
            outputs = [o.to_dict() for o in load_final_outputs(root)]
            report = self.check_lcb(root, outputs, {'toy__1': self.judge()}, .5, 3)
            self.assertEqual(report['missing_judgements'], 0)
            self.assertEqual(report['maj_at_k']['2'], 1)
            # A cached passing judgment must not resurrect an abstained output.
            self.check_lcb(root, outputs, {'toy__0': self.judge(), 'toy__1': self.judge()}, .5, 3)

    def test_lcb_all_abstentions_need_no_judge(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            self.make_run(root, [self.row('prune')])
            outputs = [o.to_dict() for o in load_final_outputs(root)]
            report = self.check_lcb(root, outputs, {}, 0, 3)
            self.assertEqual(report['maj_at_k']['1'], 0)
            self.assertEqual(report['status'], 'judged')
            self.assertEqual(report['missing_judgements'], 0)

    def test_invalid_prefix_length_is_rejected(self):
        with TemporaryDirectory() as d:
            for value in (-1, True, '100'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    materialize_v2_run_from_rows(Path(d), args_data={},
                                                rows=[self.row('prune_then_regenerate', value)])


if __name__ == '__main__':
    unittest.main()
