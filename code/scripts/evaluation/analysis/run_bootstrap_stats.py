#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from cognitive_probe.evaluation.stats import COMPARISON_MODES, METRICS, PAIRED_METRICS, run_full_bootstrap_evaluation


def parse_method_spec(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "--method must use METHOD=RUN_DIR, for example vanilla=evaluation/results/run"
        )
    method, run_dir = raw.split("=", 1)
    method = method.strip()
    run_dir = run_dir.strip()
    if not method:
        raise argparse.ArgumentTypeError("Method name cannot be empty.")
    if not run_dir:
        raise argparse.ArgumentTypeError("Run directory cannot be empty.")
    return {"method": method, "run_dir": run_dir}


def parse_method_label(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--method-label must use METHOD=LABEL.")
    method, label = raw.split("=", 1)
    method = method.strip()
    label = label.strip()
    if not method or not label:
        raise argparse.ArgumentTypeError("--method-label requires non-empty METHOD and LABEL.")
    return method, label


def parse_metric_list(raw: str) -> tuple[str, ...]:
    metrics = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not metrics:
        raise argparse.ArgumentTypeError("At least one metric is required.")
    unknown = [metric for metric in metrics if metric not in PAIRED_METRICS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown metric(s): {', '.join(unknown)}. Expected a comma-separated subset of {PAIRED_METRICS}."
        )
    return metrics


def parse_metric_alternative(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--metric-alternative must use METRIC=ALTERNATIVE.")
    metric, alternative = raw.split("=", 1)
    metric = metric.strip()
    alternative = alternative.strip()
    if metric not in PAIRED_METRICS:
        raise argparse.ArgumentTypeError(f"Unknown metric {metric!r}; expected one of {PAIRED_METRICS}.")
    if alternative not in {"greater", "less", "two-sided"}:
        raise argparse.ArgumentTypeError("Alternative must be one of: greater, less, two-sided.")
    return metric, alternative


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V2-compatible bootstrap confidence intervals and paired tests for evaluation run dirs."
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset/run label to write into output artifacts.")
    parser.add_argument("--k", type=int, default=32, help="Number of outputs per problem to evaluate.")
    parser.add_argument(
        "--method",
        action="append",
        type=parse_method_spec,
        required=True,
        help="Method spec in METHOD=RUN_DIR form. Repeat for each method.",
    )
    parser.add_argument(
        "--method-label",
        action="append",
        type=parse_method_label,
        default=[],
        help="Optional human-readable label in METHOD=LABEL form. Repeat as needed.",
    )
    parser.add_argument("--reference", required=True, help="Reference method name for paired tests.")
    parser.add_argument(
        "--comparison-mode",
        choices=COMPARISON_MODES,
        default="reference",
        help="reference compares every method to --reference; all-pairs emits both directions for every method pair.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for summary/statistics artifacts.")
    parser.add_argument("--B", type=int, default=10000, help="Bootstrap resample count.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for np.random.default_rng.")
    parser.add_argument(
        "--alternative",
        choices=["greater", "less", "two-sided"],
        default="greater",
        help="Alternative hypothesis for paired tests.",
    )
    parser.add_argument(
        "--paired-metrics",
        type=parse_metric_list,
        default=METRICS,
        help="Comma-separated paired-test metrics. Defaults to pass,mean,maj for legacy compatibility.",
    )
    parser.add_argument(
        "--metric-alternative",
        action="append",
        type=parse_metric_alternative,
        default=[],
        help=(
            "Per-metric alternative override in METRIC=ALTERNATIVE form. "
            "By default tok uses less and other metrics use --alternative."
        ),
    )
    args = parser.parse_args()

    labels = dict(args.method_label)
    method_specs = []
    for spec in args.method:
        spec = dict(spec)
        if spec["method"] in labels:
            spec["label"] = labels[spec["method"]]
        method_specs.append(spec)

    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    run_full_bootstrap_evaluation(
        method_specs=method_specs,
        reference_method=args.reference,
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset_name,
        k=args.k,
        B=args.B,
        seed=args.seed,
        alternative=args.alternative,
        paired_metrics=args.paired_metrics,
        metric_alternatives=dict(args.metric_alternative),
        comparison_mode=args.comparison_mode,
        command=command,
    )
    print(f"已写出 bootstrap 统计结果：{Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
