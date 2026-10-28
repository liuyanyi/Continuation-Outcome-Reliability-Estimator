#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from prettytable import PrettyTable

from cognitive_probe.evaluation.v2 import (
    compute_eval_report,
    ensure_v2_run,
    load_final_outputs,
    load_price_config,
    report_choices,
    scan_run_dirs,
)


def parse_ks(k_arg: str | None, run_dir: Path) -> list[int]:
    if k_arg:
        parts = str(k_arg).replace(",", " ").split()
        ks = sorted({int(item) for item in parts if item.strip()})
        if not ks:
            raise ValueError("解析到的 k 列表为空，请提供有效的正整数。")
        return ks

    final_outputs = load_final_outputs(run_dir)
    per_sample_counts: dict[str, int] = {}
    for output in final_outputs:
        per_sample_counts.setdefault(output.sample_id, 0)
        per_sample_counts[output.sample_id] += 1
    min_count = min(per_sample_counts.values()) if per_sample_counts else 0
    default_ks = [4, 8, 16, 32]
    ks = [value for value in default_ks if value <= min_count]
    if not ks and min_count > 0:
        ks = [min_count]
    if not ks:
        raise ValueError(f"无法自动推断 k：{run_dir} 中没有可评估的输出。")
    return ks


def _format_model_usage(report) -> list[str]:
    by_model = report.usage_summary.get("by_model", {}) if isinstance(report.usage_summary, dict) else {}
    rows: list[str] = []
    for model_name, usage in sorted(by_model.items()):
        if not isinstance(usage, dict):
            continue
        rows.append(
            f"{model_name}: in={usage.get('input_tokens', 0)}, "
            f"out={usage.get('output_tokens', 0)}, req={usage.get('request_count', 0)}"
        )
    return rows


def print_report_table(report) -> None:
    table = PrettyTable()
    table.field_names = ["指标", "数值"]
    table.align["指标"] = "l"
    table.align["数值"] = "r"
    table.add_row([f"Mean@{report.k}", f"{report.mean_at_k:.6f}"])
    table.add_row([f"Maj@{report.k}", f"{report.maj_at_k:.6f}"])
    table.add_row(["BestMean", f"{report.best_score_mean:.6f}"])
    table.add_row(["Distinct-4", f"{report.distinct_4:.6f}"])
    table.add_row(["全错样本数 / 总数", f"{report.solve_none} / {report.num_items}"])
    table.add_row(["全对样本数 / 总数", f"{report.solve_all} / {report.num_items}"])
    table.add_row(["平均最终可见长度", f"{report.avg_final_visible_tokens:.2f}"])
    table.add_row(["总成本", f"{report.total_cost:.6f} {report.total_cost_unit}"])
    table.add_row(["总 billed tokens", str(report.billed_tokens_total)])
    table.add_row(["输入 billed tokens", str(report.billed_input_tokens_total)])
    table.add_row(["输出 billed tokens", str(report.billed_output_tokens_total)])
    table.add_row(["cached input tokens", str(report.cached_input_tokens_total)])
    table.add_row(["请求数", str(report.request_count_total)])
    table.add_row(["最终可见 tokens", str(report.final_visible_tokens_total)])
    table.add_row(["平均单条成本", f"{report.avg_cost_per_output:.6f} {report.total_cost_unit}"])
    table.add_row(["成本节省率", "N/A" if report.cost_saving_rate is None else f"{report.cost_saving_rate:.2%}"])
    comparison = report.comparison_metrics if isinstance(report.comparison_metrics, dict) else {}
    if comparison:
        table.add_row(["对比基线", str(comparison.get("baseline_run_id") or "N/A")])
        if comparison.get("mean_at_k_delta") is not None:
            table.add_row(["Mean@k 变化", f"{float(comparison['mean_at_k_delta']):.6f}"])
        if comparison.get("maj_at_k_delta") is not None:
            table.add_row(["Maj@k 变化", f"{float(comparison['maj_at_k_delta']):.6f}"])
        if comparison.get("avg_cost_per_output_delta") is not None:
            table.add_row(["AvgCost 变化", f"{float(comparison['avg_cost_per_output_delta']):.6f} {report.total_cost_unit}"])
    strategy = report.strategy_metrics if isinstance(report.strategy_metrics, dict) else {}
    prune_metrics = strategy.get("prune") if isinstance(strategy.get("prune"), dict) else None
    if prune_metrics:
        table.add_row(
            ["剪枝比例", f"{prune_metrics.get('pruned_outputs', 0)} / {report.num_outputs_total} ({float(prune_metrics.get('prune_rate', 0.0)):.2%})"]
        )
        precision = prune_metrics.get("pruning_precision")
        fpr = prune_metrics.get("false_positive_rate")
        table.add_row(["剪枝精确率", "N/A" if precision is None else f"{float(precision):.2%}"])
        table.add_row(["误杀率 (FPR)", "N/A" if fpr is None else f"{float(fpr):.2%}"])
    for usage_line in _format_model_usage(report):
        table.add_row(["模型分解", usage_line])
    print(table)


def evaluate_single_dir(
    run_dir: Path,
    *,
    k_arg: str | None,
    price_config_path: Path | None,
    force_migrate: bool,
) -> None:
    overview = ensure_v2_run(run_dir, force_migrate=force_migrate)
    price_config = load_price_config(price_config_path) if price_config_path else None
    ks = parse_ks(k_arg, run_dir)
    print(f"\n评估目录：{run_dir}")
    print(f"运行类型：{overview.run_type} | 策略：{overview.strategy_name} | 数据集：{overview.dataset_name}")
    for k in ks:
        report = compute_eval_report(
            run_dir,
            k,
            price_config=price_config,
            price_source=str(price_config_path.resolve()) if price_config_path else None,
        )
        print(f"===== k={k} 的评估结果 =====")
        print_report_table(report)
        print(f"✓ 已保存评估报告到 {run_dir / f'eval_report_v2_{k}.json'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估系统 V2：迁移 legacy 结果并生成 eval_report_v2_{k}.json")
    parser.add_argument("--dir", type=str, required=True, help="单个运行目录或扫描根目录")
    parser.add_argument("--scan", action="store_true", help="启用扫描模式，递归评估多个运行目录")
    parser.add_argument("--recursive", action="store_true", default=False, help="扫描模式下递归扫描所有子目录")
    parser.add_argument("--skip-evaluated", action="store_true", default=False, help="扫描模式下跳过已有 V2 报告的运行目录")
    parser.add_argument("--force-migrate", action="store_true", default=False, help="强制重新生成 V2 artifact")
    parser.add_argument("--summary-only", action="store_true", default=False, help="仅打印扫描结果，不执行评估")
    parser.add_argument("-k", "--k", type=str, default=None, help="逗号或空格分隔的 k 列表；不提供则自动推断")
    parser.add_argument("--price-json", type=str, default="", help="价格表 JSON 文件路径")
    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    price_config_path = Path(args.price_json).resolve() if args.price_json else None

    if args.scan:
        run_dirs = scan_run_dirs(target_dir, recursive=args.recursive)
        print(f"扫描目录：{target_dir}")
        print(f"发现运行目录数：{len(run_dirs)}")
        if args.summary_only:
            for run_dir in run_dirs:
                print(run_dir)
            return

        if args.skip_evaluated:
            filtered_run_dirs: list[Path] = []
            for run_dir in run_dirs:
                if report_choices(run_dir):
                    continue
                filtered_run_dirs.append(run_dir)
            run_dirs = filtered_run_dirs
            print(f"跳过已有 V2 报告的目录后，剩余：{len(run_dirs)}")

        for run_dir in run_dirs:
            evaluate_single_dir(
                run_dir,
                k_arg=args.k,
                price_config_path=price_config_path,
                force_migrate=args.force_migrate,
            )
        return

    evaluate_single_dir(
        target_dir,
        k_arg=args.k,
        price_config_path=price_config_path,
        force_migrate=args.force_migrate,
    )


if __name__ == "__main__":
    main()
