#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None


PLOT_COLORS = {"overall": "#202020", "train": "#1f77b4", "test": "#d95f02"}
SPLIT_ORDER = ["train", "test", "overall"]


def _loads(line: bytes) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line.decode("utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class SplitData:
    name: str
    question_ids: list[str]
    labels: np.ndarray
    support_sizes: np.ndarray
    correct_sizes: np.ndarray
    prefix_token_lens: np.ndarray
    approx_input_token_lens: np.ndarray
    text_char_lens: np.ndarray
    samples_per_question: np.ndarray
    question_sample_counts: dict[str, int]
    question_count: int
    record_count: int


def _as_float_array(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _as_int_array(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def load_split(path: Path, name: str) -> SplitData:
    question_ids: list[str] = []
    labels: list[float] = []
    support_sizes: list[int] = []
    correct_sizes: list[int] = []
    prefix_token_lens: list[int] = []
    approx_input_token_lens: list[int] = []
    text_char_lens: list[int] = []
    question_counts: Counter[str] = Counter()

    with path.open("rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            obj = _loads(raw)
            question_id = str(obj["question_id"])
            question_ids.append(question_id)
            question_counts[question_id] += 1
            labels.append(float(obj["label"]))
            support_sizes.append(int(obj["support_size"]))
            correct_sizes.append(int(obj["correct_size"]))
            prefix_token_lens.append(int(obj["prefix_token_len"]))
            approx_input_token_lens.append(int(obj["approx_input_token_len"]))
            text_char_lens.append(len(str(obj["text"])))

    samples_per_question = list(question_counts.values())
    return SplitData(
        name=name,
        question_ids=question_ids,
        labels=_as_float_array(labels),
        support_sizes=_as_int_array(support_sizes),
        correct_sizes=_as_int_array(correct_sizes),
        prefix_token_lens=_as_int_array(prefix_token_lens),
        approx_input_token_lens=_as_int_array(approx_input_token_lens),
        text_char_lens=_as_int_array(text_char_lens),
        samples_per_question=_as_int_array(samples_per_question),
        question_sample_counts=dict(question_counts),
        question_count=len(question_counts),
        record_count=len(labels),
    )


def merge_splits(name: str, splits: list[SplitData]) -> SplitData:
    question_ids: list[str] = []
    question_sample_counts: dict[str, int] = {}
    for split in splits:
        question_ids.extend(split.question_ids)
        question_sample_counts.update(split.question_sample_counts)
    return SplitData(
        name=name,
        question_ids=question_ids,
        labels=np.concatenate([s.labels for s in splits]),
        support_sizes=np.concatenate([s.support_sizes for s in splits]),
        correct_sizes=np.concatenate([s.correct_sizes for s in splits]),
        prefix_token_lens=np.concatenate([s.prefix_token_lens for s in splits]),
        approx_input_token_lens=np.concatenate([s.approx_input_token_lens for s in splits]),
        text_char_lens=np.concatenate([s.text_char_lens for s in splits]),
        samples_per_question=np.concatenate([s.samples_per_question for s in splits]),
        question_sample_counts=question_sample_counts,
        question_count=sum(s.question_count for s in splits),
        record_count=sum(s.record_count for s in splits),
    )


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def describe_array(values: np.ndarray) -> dict[str, float]:
    return {
        "count": float(values.size),
        "min": float(np.min(values)),
        "p25": _percentile(values, 25),
        "p50": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def summarize_split(split: SplitData) -> dict[str, Any]:
    label_zero_ratio = float(np.mean(split.labels == 0.0))
    label_one_ratio = float(np.mean(split.labels == 1.0))
    return {
        "split": split.name,
        "question_count": split.question_count,
        "record_count": split.record_count,
        "samples_per_question": describe_array(split.samples_per_question),
        "labels": describe_array(split.labels),
        "support_sizes": describe_array(split.support_sizes),
        "correct_sizes": describe_array(split.correct_sizes),
        "prefix_token_lens": describe_array(split.prefix_token_lens),
        "approx_input_token_lens": describe_array(split.approx_input_token_lens),
        "text_char_lens": describe_array(split.text_char_lens),
        "label_zero_ratio": label_zero_ratio,
        "label_one_ratio": label_one_ratio,
    }


def _fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}"


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _write_markdown(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _render_row(values: list[str]) -> str:
        cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    lines = [_render_row(headers)]
    lines.append("| " + " | ".join("-" * w for w in widths) + " |")
    lines.extend(_render_row(row) for row in str_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _write_latex(path: Path, headers: list[str], rows: list[list[Any]], align: str | None = None) -> None:
    column_count = len(headers)
    alignment = align or ("l" + "r" * (column_count - 1))
    lines = [
        "% Requires \\usepackage{booktabs}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(_escape_latex(str(h)) for h in headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_escape_latex(str(cell)) for cell in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_table_bundle(base_path: Path, headers: list[str], rows: list[list[Any]], align: str | None = None) -> None:
    _write_csv(base_path.with_suffix(".csv"), headers, rows)
    _write_markdown(base_path.with_suffix(".md"), headers, rows)
    _write_latex(base_path.with_suffix(".tex"), headers, rows, align=align)


def build_overview_rows(stats_by_split: dict[str, dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "Split",
        "Questions",
        "Samples",
        "Avg samples/Q",
        "Median samples/Q",
        "Label mean",
        "Label std",
        "Label=0 (%)",
        "Label=1 (%)",
        "Median input len",
        "P90 input len",
        "Median support",
        "P90 support",
    ]
    rows: list[list[Any]] = []
    for split in SPLIT_ORDER:
        stats = stats_by_split[split]
        rows.append(
            [
                split,
                stats["question_count"],
                stats["record_count"],
                _fmt_float(stats["samples_per_question"]["mean"], 1),
                _fmt_float(stats["samples_per_question"]["p50"], 1),
                _fmt_float(stats["labels"]["mean"], 3),
                _fmt_float(stats["labels"]["std"], 3),
                _fmt_pct(stats["label_zero_ratio"]),
                _fmt_pct(stats["label_one_ratio"]),
                _fmt_float(stats["approx_input_token_lens"]["p50"], 1),
                _fmt_float(stats["approx_input_token_lens"]["p90"], 1),
                _fmt_float(stats["support_sizes"]["p50"], 1),
                _fmt_float(stats["support_sizes"]["p90"], 1),
            ]
        )
    return headers, rows


def build_quantile_rows(stats_by_split: dict[str, dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    headers = ["Metric", "Split", "Min", "P25", "Median", "P75", "P90", "P95", "P99", "Max", "Mean", "Std"]
    metric_order = [
        ("label", "labels"),
        ("support_size", "support_sizes"),
        ("correct_size", "correct_sizes"),
        ("prefix_token_len", "prefix_token_lens"),
        ("approx_input_token_len", "approx_input_token_lens"),
        ("text_char_len", "text_char_lens"),
        ("samples_per_question", "samples_per_question"),
    ]
    rows: list[list[Any]] = []
    for metric_name, key in metric_order:
        for split in SPLIT_ORDER:
            stats = stats_by_split[split][key]
            digits = 3 if metric_name == "label" else 1
            rows.append(
                [
                    metric_name,
                    split,
                    _fmt_float(stats["min"], digits),
                    _fmt_float(stats["p25"], digits),
                    _fmt_float(stats["p50"], digits),
                    _fmt_float(stats["p75"], digits),
                    _fmt_float(stats["p90"], digits),
                    _fmt_float(stats["p95"], digits),
                    _fmt_float(stats["p99"], digits),
                    _fmt_float(stats["max"], digits),
                    _fmt_float(stats["mean"], digits),
                    _fmt_float(stats["std"], digits),
                ]
            )
    return headers, rows


def _format_bin_label(left: float, right: float, is_last: bool, digits: int = 1) -> str:
    if digits == 0:
        left_text = f"{int(left)}"
        right_text = f"{int(right)}"
    else:
        left_text = f"{left:.{digits}f}"
        right_text = f"{right:.{digits}f}"
    closing = "]" if is_last else ")"
    return f"[{left_text}, {right_text}{closing}"


def _hist_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts


def build_binned_rows(
    splits: dict[str, SplitData],
    attr_name: str,
    edges: list[float],
    digits: int,
) -> tuple[list[str], list[list[Any]]]:
    headers = ["Bin", "train count", "train (%)", "test count", "test (%)", "overall count", "overall (%)"]
    histograms: dict[str, np.ndarray] = {}
    edge_arr = np.asarray(edges, dtype=np.float64)
    for split in SPLIT_ORDER:
        values = getattr(splits[split], attr_name)
        histograms[split] = _hist_counts(values, edge_arr)

    rows: list[list[Any]] = []
    for idx in range(len(edges) - 1):
        label = _format_bin_label(edges[idx], edges[idx + 1], idx == len(edges) - 2, digits)
        row: list[Any] = [label]
        for split in SPLIT_ORDER:
            counts = histograms[split]
            total = max(1, int(counts.sum()))
            row.extend([int(counts[idx]), _fmt_pct(float(counts[idx] / total))])
        rows.append(row)
    return headers, rows


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _bin_ratio(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = _hist_counts(values, edges)
    ratios = counts / max(1, counts.sum())
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, ratios


def _support_edges(splits: dict[str, SplitData]) -> np.ndarray:
    max_support = max(int(np.max(split.support_sizes)) for split in splits.values())
    return np.geomspace(8, max_support + 1, num=18)


def plot_label_distribution(splits: dict[str, SplitData], label_edges: list[float], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    edges = np.asarray(label_edges, dtype=np.float64)
    for split in ["overall", "train", "test"]:
        centers, ratios = _bin_ratio(splits[split].labels, edges)
        ax.plot(centers, ratios, marker="o", linewidth=2.0, markersize=4.5, label=split, color=PLOT_COLORS[split])
    tick_labels = [_format_bin_label(edges[i], edges[i + 1], i == len(edges) - 2, 1) for i in range(len(edges) - 1)]
    ax.set_xticks((edges[:-1] + edges[1:]) / 2)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Label bin")
    ax.set_title("Label Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, out_base)


def plot_input_len_distribution(splits: dict[str, SplitData], input_edges: list[float], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    edges = np.asarray(input_edges, dtype=np.float64)
    width = 0.24
    x = np.arange(len(edges) - 1)
    for offset, split in zip([-width, 0.0, width], ["train", "test", "overall"], strict=True):
        counts = _hist_counts(splits[split].approx_input_token_lens, edges)
        ratios = counts / max(1, counts.sum())
        ax.bar(x + offset, ratios, width=width, label=split, color=PLOT_COLORS[split], alpha=0.88)
    tick_labels = [_format_bin_label(edges[i], edges[i + 1], i == len(edges) - 2, 0) for i in range(len(edges) - 1)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Approx. input token length bin")
    ax.set_title("Approx. Input Token Length Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, out_base)


def plot_support_distribution(splits: dict[str, SplitData], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    edges = _support_edges(splits)
    for split in ["overall", "train", "test"]:
        centers, ratios = _bin_ratio(splits[split].support_sizes, edges)
        ax.plot(centers, ratios, marker="o", linewidth=2.0, markersize=4.0, label=split, color=PLOT_COLORS[split])
    ax.set_xscale("log")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Support size")
    ax.set_title("Support Size Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, out_base)


def plot_question_sample_distribution(splits: dict[str, SplitData], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for split in ["overall", "train", "test"]:
        values = np.sort(splits[split].samples_per_question)
        y = np.arange(1, values.size + 1) / values.size
        ax.step(values, y, where="post", linewidth=2.0, label=split, color=PLOT_COLORS[split])
    ax.set_xlabel("Samples per question")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Question-Level Sample Count Distribution")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, out_base)


def plot_dataset_overview(
    splits: dict[str, SplitData],
    label_edges: list[float],
    input_edges: list[float],
    out_base: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), constrained_layout=True)

    label_arr = np.asarray(label_edges, dtype=np.float64)
    for split in ["overall", "train", "test"]:
        centers, ratios = _bin_ratio(splits[split].labels, label_arr)
        axes[0, 0].plot(centers, ratios, marker="o", linewidth=2.0, markersize=4.0, label=split, color=PLOT_COLORS[split])
    axes[0, 0].set_title("A. Label Distribution")
    axes[0, 0].set_xlabel("Label bin center")
    axes[0, 0].set_ylabel("Ratio")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)

    input_arr = np.asarray(input_edges, dtype=np.float64)
    width = 0.24
    x = np.arange(len(input_arr) - 1)
    for offset, split in zip([-width, 0.0, width], ["train", "test", "overall"], strict=True):
        counts = _hist_counts(splits[split].approx_input_token_lens, input_arr)
        ratios = counts / max(1, counts.sum())
        axes[0, 1].bar(x + offset, ratios, width=width, label=split, color=PLOT_COLORS[split], alpha=0.88)
    axes[0, 1].set_title("B. Approx. Input Length")
    axes[0, 1].set_xlabel("Token length bin")
    axes[0, 1].set_ylabel("Ratio")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(
        [_format_bin_label(input_arr[i], input_arr[i + 1], i == len(input_arr) - 2, 0) for i in range(len(input_arr) - 1)],
        rotation=30,
        ha="right",
    )
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend(frameon=False)

    support_edges = _support_edges(splits)
    for split in ["overall", "train", "test"]:
        centers, ratios = _bin_ratio(splits[split].support_sizes, support_edges)
        axes[1, 0].plot(centers, ratios, marker="o", linewidth=2.0, markersize=4.0, label=split, color=PLOT_COLORS[split])
    axes[1, 0].set_title("C. Support Size")
    axes[1, 0].set_xlabel("Support size")
    axes[1, 0].set_ylabel("Ratio")
    axes[1, 0].set_xscale("log")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(frameon=False)

    for split in ["overall", "train", "test"]:
        values = np.sort(splits[split].samples_per_question)
        y = np.arange(1, values.size + 1) / values.size
        axes[1, 1].step(values, y, where="post", linewidth=2.0, label=split, color=PLOT_COLORS[split])
    axes[1, 1].set_title("D. Samples per Question")
    axes[1, 1].set_xlabel("Samples per question")
    axes[1, 1].set_ylabel("Empirical CDF")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Seq-Reg Dataset Statistics for Paper", fontsize=15)
    _save_figure(fig, out_base)


def export_record_level_raw(raw_dir: Path, splits: dict[str, SplitData]) -> None:
    path = raw_dir / "records_light.csv"
    headers = [
        "split",
        "question_id",
        "label",
        "support_size",
        "correct_size",
        "prefix_token_len",
        "approx_input_token_len",
        "text_char_len",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for split_name in ["train", "test"]:
            split = splits[split_name]
            for idx, question_id in enumerate(split.question_ids):
                writer.writerow(
                    [
                        split_name,
                        question_id,
                        _fmt_float(float(split.labels[idx]), 6),
                        int(split.support_sizes[idx]),
                        int(split.correct_sizes[idx]),
                        int(split.prefix_token_lens[idx]),
                        int(split.approx_input_token_lens[idx]),
                        int(split.text_char_lens[idx]),
                    ]
                )


def export_question_counts_raw(raw_dir: Path, splits: dict[str, SplitData]) -> None:
    rows: list[list[Any]] = []
    for split_name in ["train", "test", "overall"]:
        items = sorted(splits[split_name].question_sample_counts.items())
        for question_id, sample_count in items:
            rows.append([split_name, question_id, sample_count])
    _write_csv(raw_dir / "question_sample_counts.csv", ["split", "question_id", "sample_count"], rows)


def export_histogram_raw(
    raw_dir: Path,
    filename: str,
    splits: dict[str, SplitData],
    attr_name: str,
    edges: np.ndarray,
) -> None:
    rows: list[list[Any]] = []
    for split_name in SPLIT_ORDER:
        values = getattr(splits[split_name], attr_name)
        counts = _hist_counts(values, edges)
        total = max(1, int(counts.sum()))
        centers = (edges[:-1] + edges[1:]) / 2
        for idx, count in enumerate(counts.tolist()):
            rows.append(
                [
                    split_name,
                    idx,
                    float(edges[idx]),
                    float(edges[idx + 1]),
                    float(centers[idx]),
                    idx == len(counts) - 1,
                    int(count),
                    float(count / total),
                ]
            )
    _write_csv(
        raw_dir / filename,
        ["split", "bin_id", "left", "right", "center", "is_last_bin", "count", "ratio"],
        rows,
    )


def export_question_ecdf_raw(raw_dir: Path, splits: dict[str, SplitData]) -> None:
    rows: list[list[Any]] = []
    for split_name in SPLIT_ORDER:
        values = np.sort(splits[split_name].samples_per_question)
        denom = max(1, values.size)
        for idx, value in enumerate(values.tolist(), start=1):
            rows.append([split_name, idx, int(value), float(idx / denom)])
    _write_csv(raw_dir / "question_sample_ecdf.csv", ["split", "rank", "sample_count", "ecdf"], rows)


def export_figure_specs(raw_dir: Path, label_edges: list[float], input_edges: list[float], support_edges: np.ndarray) -> None:
    payload = {
        "split_order": SPLIT_ORDER,
        "plot_colors": PLOT_COLORS,
        "label_bin_edges": [float(x) for x in label_edges],
        "input_token_len_bin_edges": [float(x) for x in input_edges],
        "support_size_bin_edges": [float(x) for x in support_edges.tolist()],
    }
    (raw_dir / "figure_specs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_readme(
    output_dir: Path,
    dataset_dir: Path,
    stats_by_split: dict[str, dict[str, Any]],
) -> None:
    overall = stats_by_split["overall"]
    train = stats_by_split["train"]
    test = stats_by_split["test"]
    text = f"""# Paper Assets for `{dataset_dir.name}`

This directory contains paper-ready tables, figures, and raw plotting data generated from the seq-reg dataset.

## Quick Summary

- Total questions: `{overall['question_count']}`
- Total samples: `{overall['record_count']}`
- Train/test split: `{train['record_count']}` / `{test['record_count']}`
- Label mean: `{overall['labels']['mean']:.3f}`
- Median approximate input length: `{overall['approx_input_token_lens']['p50']:.1f}` tokens
- Median support size: `{overall['support_sizes']['p50']:.1f}`

## Directory Layout

- `tables/`: exported tables in `.csv`, `.md`, and `.tex`
- `figures/`: rendered figures in `.png` and `.svg`
- `raw/`: raw plotting data for redrawing every figure without re-reading the original `train_seq.jsonl` and `test_seq.jsonl`
- `dataset_stats.json`: aggregated descriptive statistics used to build the summary tables

## Raw Data Files

- `raw/records_light.csv`
  - Record-level lightweight data.
  - Columns: `split`, `question_id`, `label`, `support_size`, `correct_size`, `prefix_token_len`, `approx_input_token_len`, `text_char_len`
  - This is sufficient to recompute additional quantiles or redraw distributions from scratch.

- `raw/question_sample_counts.csv`
  - Question-level sample counts for each split.
  - Used for the question-count distribution figure.

- `raw/label_distribution.csv`
  - Binned counts and ratios for the label distribution figure.

- `raw/input_len_distribution.csv`
  - Binned counts and ratios for the approximate input length figure.

- `raw/support_distribution.csv`
  - Binned counts and ratios for the support-size figure.

- `raw/question_sample_ecdf.csv`
  - ECDF points used by the question-level sample count figure.

- `raw/figure_specs.json`
  - Bin edges, split order, and plotting colors.

## Figure Mapping

- `figures/label_distribution.*` <- `raw/label_distribution.csv`
- `figures/input_len_distribution.*` <- `raw/input_len_distribution.csv`
- `figures/support_distribution.*` <- `raw/support_distribution.csv`
- `figures/question_sample_distribution.*` <- `raw/question_sample_ecdf.csv` or `raw/question_sample_counts.csv`
- `figures/dataset_overview.*` <- the four raw files above combined

## Rebuild Command

From the repository root, rerun:

```bash
.venv/bin/python scripts/training/analyze_seq_reg_dataset.py --dataset-dir {dataset_dir.as_posix()}
```

This will regenerate `tables/`, `figures/`, `raw/`, and `dataset_stats.json`.

## Reproducibility Note

`paper_assets/` is now self-contained for plotting purposes: every figure in `figures/` has a corresponding raw data export in `raw/`. The original dataset files are only needed if you want to regenerate the raw exports themselves.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready tables and figures for seq-reg datasets.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/seq_reg_dataset_v1_optimized_8k",
        help="Directory containing summary.json, train_seq.jsonl, and test_seq.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Defaults to <dataset-dir>/paper_assets",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "paper_assets"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(dataset_dir / "summary.json")
    summary_cfg = summary["final_dataset_stats"]["summary_config"]
    label_edges = [float(x) for x in summary_cfg["label_bin_edges"]]
    input_edges = [float(x) for x in summary_cfg["input_token_len_bin_edges"]]

    train = load_split(dataset_dir / "train_seq.jsonl", "train")
    test = load_split(dataset_dir / "test_seq.jsonl", "test")
    overall = merge_splits("overall", [train, test])
    splits = {"train": train, "test": test, "overall": overall}

    stats_by_split = {name: summarize_split(split) for name, split in splits.items()}
    (output_dir / "dataset_stats.json").write_text(json.dumps(stats_by_split, indent=2), encoding="utf-8")

    overview_headers, overview_rows = build_overview_rows(stats_by_split)
    _write_table_bundle(tables_dir / "dataset_overview", overview_headers, overview_rows)

    quant_headers, quant_rows = build_quantile_rows(stats_by_split)
    _write_table_bundle(tables_dir / "dataset_quantiles", quant_headers, quant_rows)

    label_bin_headers, label_bin_rows = build_binned_rows(splits, "labels", label_edges, digits=1)
    _write_table_bundle(tables_dir / "label_bin_distribution", label_bin_headers, label_bin_rows)

    input_bin_headers, input_bin_rows = build_binned_rows(splits, "approx_input_token_lens", input_edges, digits=0)
    _write_table_bundle(tables_dir / "input_len_bin_distribution", input_bin_headers, input_bin_rows)

    _configure_matplotlib()
    plot_label_distribution(splits, label_edges, figures_dir / "label_distribution")
    plot_input_len_distribution(splits, input_edges, figures_dir / "input_len_distribution")
    plot_support_distribution(splits, figures_dir / "support_distribution")
    plot_question_sample_distribution(splits, figures_dir / "question_sample_distribution")
    plot_dataset_overview(splits, label_edges, input_edges, figures_dir / "dataset_overview")

    label_edge_arr = np.asarray(label_edges, dtype=np.float64)
    input_edge_arr = np.asarray(input_edges, dtype=np.float64)
    support_edge_arr = _support_edges(splits)
    export_record_level_raw(raw_dir, splits)
    export_question_counts_raw(raw_dir, splits)
    export_histogram_raw(raw_dir, "label_distribution.csv", splits, "labels", label_edge_arr)
    export_histogram_raw(raw_dir, "input_len_distribution.csv", splits, "approx_input_token_lens", input_edge_arr)
    export_histogram_raw(raw_dir, "support_distribution.csv", splits, "support_sizes", support_edge_arr)
    export_question_ecdf_raw(raw_dir, splits)
    export_figure_specs(raw_dir, label_edges, input_edges, support_edge_arr)
    write_readme(output_dir, dataset_dir, stats_by_split)

    print(f"Saved paper assets to: {output_dir}")


if __name__ == "__main__":
    main()
