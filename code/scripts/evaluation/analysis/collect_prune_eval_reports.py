#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_k_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError('No k values provided.')
    return values


def _fmt(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(fieldnames) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(fieldnames)) + ' |\n')
        for row in rows:
            f.write('| ' + ' | '.join(_fmt(row.get(name)) for name in fieldnames) + ' |\n')


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_row(root_dir: Path, report_path: Path) -> dict[str, Any]:
    with report_path.open('r', encoding='utf-8') as f:
        report = json.load(f)

    rel_parts = report_path.relative_to(root_dir).parts
    if len(rel_parts) < 3:
        raise ValueError(f'Unexpected report path layout: {report_path}')
    experiment = rel_parts[0]
    case_name = rel_parts[1]

    comparison = report.get('comparison_metrics') if isinstance(report.get('comparison_metrics'), dict) else {}
    strategy_metrics = report.get('strategy_metrics') if isinstance(report.get('strategy_metrics'), dict) else {}
    prune = strategy_metrics.get('prune') if isinstance(strategy_metrics.get('prune'), dict) else {}

    num_outputs_total = int(report.get('num_outputs_total') or 0)
    would_pass_total = int(prune.get('would_pass_total') or 0)
    total_fail = max(num_outputs_total - would_pass_total, 0)
    tp = int(prune.get('pruned_would_fail_count') or 0)
    fp = int(prune.get('pruned_would_pass_count') or 0)
    tn = max(would_pass_total - fp, 0)
    fn = max(total_fail - tp, 0)

    precision = _safe_float(prune.get('pruning_precision'))
    recall = (tp / total_fail) if total_fail > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    row = {
        'experiment': experiment,
        'case_name': case_name,
        'k': int(report['k']),
        'mean_at_k': _safe_float(report.get('mean_at_k')),
        'mean_at_k_delta': _safe_float(comparison.get('mean_at_k_delta')),
        'maj_at_k': _safe_float(report.get('maj_at_k')),
        'maj_at_k_delta': _safe_float(comparison.get('maj_at_k_delta')),
        'best_score_mean': _safe_float(report.get('best_score_mean')),
        'distinct_4': _safe_float(report.get('distinct_4')),
        'avg_final_visible_tokens': _safe_float(report.get('avg_final_visible_tokens')),
        'avg_cost_per_output': _safe_float(report.get('avg_cost_per_output')),
        'avg_cost_per_output_delta': _safe_float(comparison.get('avg_cost_per_output_delta')),
        'cost_saving_rate': _safe_float(report.get('cost_saving_rate')),
        'format_error_rollouts': int(report.get('format_error_rollouts') or 0),
        'pruned_outputs': int(prune.get('pruned_outputs') or 0),
        'prune_rate': _safe_float(prune.get('prune_rate')),
        'pruning_precision': precision,
        'false_positive_rate': _safe_float(prune.get('false_positive_rate')),
        'pruning_recall': recall,
        'pruning_f1': f1,
        'pruned_would_fail_count': tp,
        'pruned_would_pass_count': fp,
        'would_pass_total': would_pass_total,
        'would_fail_total': total_fail,
        'tn': tn,
        'fn': fn,
        'run_dir': str(report.get('run_dir') or ''),
        'report_path': str(report_path),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect prune eval reports into CSV/Markdown summaries.')
    parser.add_argument('--root-dir', required=True, help='Root directory containing experiment subdirectories.')
    parser.add_argument('--output-dir', default='', help='Output directory. Defaults to <root-dir>/summary.')
    parser.add_argument('--k-values', default='4,8,16,32', help='Comma-separated k values to emit per-k summaries for.')
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (root_dir / 'summary')
    k_values = set(_parse_k_values(args.k_values))

    report_paths = sorted(root_dir.glob('*/**/eval_report_v2_*.json'))
    if not report_paths:
        raise SystemExit(f'No eval_report_v2_*.json found under {root_dir}')

    rows = [_extract_row(root_dir, path) for path in report_paths]
    rows.sort(key=lambda row: (row['k'], row['experiment']))

    fieldnames = [
        'experiment', 'case_name', 'k', 'mean_at_k', 'mean_at_k_delta', 'maj_at_k', 'maj_at_k_delta', 'best_score_mean', 'distinct_4',
        'avg_final_visible_tokens', 'avg_cost_per_output', 'avg_cost_per_output_delta', 'cost_saving_rate',
        'format_error_rollouts', 'pruned_outputs', 'prune_rate', 'pruning_precision', 'false_positive_rate',
        'pruning_recall', 'pruning_f1', 'pruned_would_fail_count', 'pruned_would_pass_count', 'would_pass_total',
        'would_fail_total', 'tn', 'fn', 'run_dir', 'report_path'
    ]

    _write_csv(output_dir / 'summary_all.csv', rows, fieldnames)
    _write_markdown(output_dir / 'summary_all.md', rows, fieldnames[:-2])

    for k in sorted(k_values):
        k_rows = [row for row in rows if row['k'] == k]
        if not k_rows:
            continue
        k_rows.sort(
            key=lambda row: (
                -(row['pruning_f1'] if row['pruning_f1'] is not None else -1.0),
                -(row['mean_at_k'] if row['mean_at_k'] is not None else -1.0),
                row['false_positive_rate'] if row['false_positive_rate'] is not None else 1.0,
            )
        )
        _write_csv(output_dir / f'summary_k{k}.csv', k_rows, fieldnames)
        _write_markdown(output_dir / f'summary_k{k}.md', k_rows, fieldnames[:-2])

    manifest = {
        'root_dir': str(root_dir),
        'output_dir': str(output_dir),
        'report_count': len(report_paths),
        'experiment_count': len({row['experiment'] for row in rows}),
        'case_count': len({row['case_name'] for row in rows}),
        'k_values_present': sorted({int(row['k']) for row in rows}),
    }
    with (output_dir / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'Saved: {output_dir / "summary_all.csv"}')
    print(f'Saved: {output_dir / "summary_all.md"}')


if __name__ == '__main__':
    main()
