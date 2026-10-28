#!/usr/bin/env python3
"""Build seq-reg dataset from tree generation outputs.

目标：
1. 从 tree_gen 结果中抽取“分歧点”训练样本（分歧点 = 以换行 token 结尾的共享前缀）。
2. 监督信号为该分歧点之后“答对率”（同一前缀下所有分支的正确率均值）。
3. 按原始样本 id 划分 train/test，避免样本泄漏。
"""

from __future__ import annotations

import argparse
import array
import json
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import xxhash
from tqdm import tqdm
from transformers import AutoTokenizer

from cognitive_probe.utils.math_utils import (
    extract_answer,
    grade_answer_mathd,
    grade_answer_verl,
)

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None


def _loads_bytes(buf: bytes) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(buf)
    return json.loads(buf.decode("utf-8"))


def _dumps_line(obj: dict[str, Any]) -> bytes:
    if orjson is not None:
        return orjson.dumps(obj) + b"\n"
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def load_questions_map(path: Path) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            obj = _loads_bytes(raw)
            qid = obj.get("id")
            if isinstance(qid, str):
                questions[qid] = obj
    return questions


def build_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    _messages = [dict(m) for m in messages]
    kwargs: dict[str, Any] = {"conversation": _messages, "tokenize": False}
    if _messages and _messages[-1].get("role") != "assistant":
        kwargs["add_generation_prompt"] = True
    rendered = tokenizer.apply_chat_template(**kwargs)
    if isinstance(rendered, str):
        return rendered
    return str(rendered)


@dataclass
class PrefixGroup:
    prefix_ids: list[int]
    branch_indices: list[int]


@dataclass
class SeqCandidate:
    key: str
    prefix_ids: list[int]
    label: float
    support_size: int
    correct_size: int
    sample_branch_id: str
    branch_set_hash: str


class NewlineTokenDetector:
    def __init__(self, tokenizer: Any, mode: str) -> None:
        self.tokenizer = tokenizer
        self.mode = mode
        self._cache: dict[int, int] = {}

    def _newline_count(self, token_id: int) -> int:
        cached = self._cache.get(token_id)
        if cached is not None:
            return cached

        tok = self.tokenizer.convert_ids_to_tokens(token_id)
        tok_text = tok if isinstance(tok, str) else str(tok)
        decoded = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        cnt = max(tok_text.count("\n"), decoded.count("\n"))
        if tok_text in {"<0x0A>", "Ċ"}:
            cnt = max(cnt, 1)
        self._cache[token_id] = cnt
        return cnt

    def is_split_token(self, token_id: int) -> bool:
        cnt = self._newline_count(token_id)
        if self.mode == "single_or_double":
            return cnt in {1, 2}
        if self.mode == "at_least_one":
            return cnt >= 1
        raise ValueError(f"Unknown newline mode: {self.mode}")


class SplitPointDetector:
    def __init__(
        self,
        tokenizer: Any,
        *,
        split_point_mode: str,
        newline_mode: str,
        track_characters: list[str],
        track_character_match_mode: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.split_point_mode = split_point_mode
        self.newline_detector = NewlineTokenDetector(tokenizer, mode=newline_mode)
        self.track_characters = track_characters
        self.track_character_match_mode = track_character_match_mode
        self._track_cache: dict[tuple[int, str], bool] = {}
        self._decode_cache: dict[int, str] = {}

    def _decode_token(self, token_id: int) -> str:
        cached = self._decode_cache.get(token_id)
        if cached is not None:
            return cached
        decoded = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        self._decode_cache[token_id] = decoded
        return decoded

    def _is_track_hit(self, token_id: int, character: str) -> bool:
        key = (token_id, character)
        cached = self._track_cache.get(key)
        if cached is not None:
            return cached
        decoded = self._decode_token(token_id)
        hit = character in decoded
        self._track_cache[key] = hit
        return hit

    def _stream_track_hit(self, token_id: int, previous_tail: str) -> tuple[bool, str]:
        if not self.track_characters:
            return False, ""

        decoded = self._decode_token(token_id)
        window = previous_tail + decoded
        hit = any(ch in window for ch in self.track_characters)
        max_tail_len = max(0, max(len(ch) for ch in self.track_characters) - 1)
        if max_tail_len == 0:
            return hit, ""
        return hit, window[-max_tail_len:]

    def _current_token_track_hit(self, token_id: int, previous_tail: str) -> tuple[bool, str]:
        if self.track_character_match_mode == "token":
            hit = any(self._is_track_hit(token_id, ch) for ch in self.track_characters)
            return hit, ""
        if self.track_character_match_mode == "stream":
            return self._stream_track_hit(token_id, previous_tail)
        raise ValueError(f"Unknown track character match mode: {self.track_character_match_mode}")

    def iter_split_positions(self, token_ids: list[int], max_tokens: int) -> list[int]:
        split_positions: list[int] = []
        pending_tracked_next = False
        stream_tail = ""

        for pos, token_id in enumerate(token_ids):
            # 仅保留“生成前 N token”内的分歧点，超出后直接停止该分支扫描。
            if max_tokens > 0 and (pos + 1) > max_tokens:
                break

            is_split = False
            if self.split_point_mode in {"newline", "newline_or_tracked_next_token"}:
                is_split = self.newline_detector.is_split_token(token_id)

            if self.split_point_mode in {
                "tracked_next_token",
                "newline_or_tracked_next_token",
            }:
                # tree_generate marks a tracked token and creates expansion tasks on
                # the following generated token, where top-logprobs are available.
                is_split = is_split or pending_tracked_next
                pending_tracked_next, stream_tail = self._current_token_track_hit(
                    token_id,
                    stream_tail,
                )

            if is_split:
                split_positions.append(pos)

        return split_positions


def get_file_id(path: Path) -> str:
    return path.stem


def score_branch(output_text: str, answer: Any, score_mode: str) -> float:
    if score_mode == "verl":
        return float(grade_answer_verl(output_text, answer))

    if score_mode == "fast":
        gt = answer
        if isinstance(gt, str) and "\\boxed" in gt:
            gt = extract_answer(gt)
        pred = extract_answer(output_text)
        if pred is None or gt is None:
            return 0.0
        return float(grade_answer_mathd(pred, gt))

    raise ValueError(f"Unknown score mode: {score_mode}")


def build_prefix_groups(
    all_token_ids: list[list[int]],
    detector: SplitPointDetector,
    max_generated_prefix_tokens: int,
) -> dict[str, PrefixGroup]:
    groups: dict[str, PrefixGroup] = {}
    for branch_idx, token_ids in enumerate(all_token_ids):
        hasher = xxhash.xxh64()
        split_positions = set(
            detector.iter_split_positions(token_ids, max_generated_prefix_tokens)
        )
        for pos, token_id in enumerate(token_ids):
            # 仅保留“生成前 N token”内的分歧点，超出后直接停止该分支扫描。
            if (
                max_generated_prefix_tokens > 0
                and (pos + 1) > max_generated_prefix_tokens
            ):
                break
            hasher.update(array.array("q", [token_id]))
            if pos not in split_positions:
                continue
            key = f"{pos}:{hasher.hexdigest()}"
            if key not in groups:
                groups[key] = PrefixGroup(
                    prefix_ids=token_ids[: pos + 1],
                    branch_indices=[branch_idx],
                )
            else:
                groups[key].branch_indices.append(branch_idx)
    return groups


def build_seq_candidates(
    groups: dict[str, PrefixGroup],
    *,
    min_support: int,
    correct_flags: list[int],
    branch_ids: list[str],
) -> list[SeqCandidate]:
    candidates: list[SeqCandidate] = []
    for key, group in groups.items():
        if len(group.branch_indices) < min_support:
            continue
        correct = sum(correct_flags[i] for i in group.branch_indices)
        total = len(group.branch_indices)
        target = correct / total
        candidates.append(
            SeqCandidate(
                key=key,
                prefix_ids=group.prefix_ids,
                label=float(target),
                support_size=total,
                correct_size=correct,
                sample_branch_id=branch_ids[group.branch_indices[0]],
                branch_set_hash=build_branch_set_hash(group.branch_indices),
            )
        )
    return candidates


def write_record(fp: Any, record: dict[str, Any]) -> None:
    fp.write(_dumps_line(record))


def _is_number_like(x: Any) -> bool:
    return isinstance(x, int | float)


def has_full_precomputed_scores(branches: list[dict[str, Any]]) -> bool:
    if not branches:
        return False
    for b in branches:
        if not isinstance(b, dict):
            return False
        if not _is_number_like(b.get("score")):
            return False
    return True


def is_train_id(qid: str, train_ratio: float, seed: int) -> bool:
    # Stable split by question id, independent of currently available subset.
    h = xxhash.xxh64(f"{seed}:{qid}".encode("utf-8")).intdigest()
    return (h / (2**64)) < train_ratio


def build_branch_set_hash(branch_indices: list[int]) -> str:
    hasher = xxhash.xxh64()
    for idx in sorted(branch_indices):
        hasher.update(array.array("q", [idx]))
    return hasher.hexdigest()


def prune_nearby_same_label(
    candidates: list[SeqCandidate],
    min_prefix_gap_tokens: int,
    label_round_decimals: int,
) -> tuple[list[SeqCandidate], int]:
    if min_prefix_gap_tokens <= 0:
        return candidates, 0

    kept: list[SeqCandidate] = []
    dropped = 0
    last_kept_pos: dict[str, int] = {}

    # 先按前缀长度排序，避免保留过于密集的相邻断点。
    candidates_sorted = sorted(candidates, key=lambda x: len(x.prefix_ids))
    for c in candidates_sorted:
        rounded_label = round(c.label, label_round_decimals)
        dedup_key = f"{c.branch_set_hash}:{rounded_label}"
        current_len = len(c.prefix_ids)

        prev_len = last_kept_pos.get(dedup_key)
        if prev_len is not None and (current_len - prev_len) < min_prefix_gap_tokens:
            dropped += 1
            continue

        kept.append(c)
        last_kept_pos[dedup_key] = current_len

    return kept, dropped


def _candidate_priority_key(c: SeqCandidate) -> tuple[float, int, int, str]:
    # Prefer higher support, then longer prefix (usually stronger context), then higher correct count.
    return (-c.support_size, -len(c.prefix_ids), -c.correct_size, c.key)


def _bucketize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, values, side="right") - 1
    return np.clip(bins, 0, len(edges) - 2).astype(np.int32)


def _quantile_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    if num_bins <= 1:
        lo = float(values.min())
        hi = float(values.max())
        if hi <= lo:
            hi = lo + 1e-9
        return np.array([lo, hi], dtype=np.float64)
    q = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(values, q)
    edges = np.unique(edges)
    if edges.size < 2:
        lo = float(values.min())
        hi = float(values.max())
        if hi <= lo:
            hi = lo + 1e-9
        edges = np.array([lo, hi], dtype=np.float64)
    return edges


def _take_round_robin_from_buckets(
    bucket_to_items: dict[int, list[SeqCandidate]],
    selected_keys: set[str],
    limit: int,
) -> list[SeqCandidate]:
    if limit <= 0:
        return []
    taken: list[SeqCandidate] = []
    positions: dict[int, int] = {k: 0 for k in bucket_to_items}
    # Smaller buckets first to preserve rare regions.
    bucket_order = sorted(bucket_to_items.keys(), key=lambda b: len(bucket_to_items[b]))

    while len(taken) < limit:
        progressed = False
        for b in bucket_order:
            items = bucket_to_items[b]
            pos = positions[b]
            while pos < len(items) and items[pos].key in selected_keys:
                pos += 1
            positions[b] = pos
            if pos >= len(items):
                continue
            c = items[pos]
            positions[b] = pos + 1
            selected_keys.add(c.key)
            taken.append(c)
            progressed = True
            if len(taken) >= limit:
                break
        if not progressed:
            break
    return taken


def cap_candidates_with_diversity(
    candidates: list[SeqCandidate],
    max_samples_per_question: int,
    cap_strategy: str,
    label_bins: int,
    length_bins: int,
    label_ratio: float,
    length_ratio: float,
) -> tuple[list[SeqCandidate], int]:
    if max_samples_per_question <= 0 or len(candidates) <= max_samples_per_question:
        return candidates, 0

    sorted_candidates = sorted(candidates, key=_candidate_priority_key)
    cap = min(max_samples_per_question, len(sorted_candidates))

    if cap_strategy == "none":
        kept = sorted_candidates[:cap]
        return kept, len(candidates) - len(kept)

    # diversity strategy
    labels = np.array([c.label for c in sorted_candidates], dtype=np.float64)
    lengths = np.array([len(c.prefix_ids) for c in sorted_candidates], dtype=np.float64)

    label_min, label_max = float(labels.min()), float(labels.max())
    if label_max <= label_min:
        label_max = label_min + 1e-9
    label_edges = np.linspace(label_min, label_max, max(2, label_bins) + 1, dtype=np.float64)
    length_edges = _quantile_edges(lengths, max(2, length_bins))

    label_bucket_ids = _bucketize(labels, label_edges)
    length_bucket_ids = _bucketize(lengths, length_edges)

    by_label: dict[int, list[SeqCandidate]] = defaultdict(list)
    by_length: dict[int, list[SeqCandidate]] = defaultdict(list)
    for c, lb, tb in zip(sorted_candidates, label_bucket_ids, length_bucket_ids):
        by_label[int(lb)].append(c)
        by_length[int(tb)].append(c)

    selected_keys: set[str] = set()
    kept: list[SeqCandidate] = []

    label_quota = int(cap * label_ratio)
    length_quota = int(cap * length_ratio)
    if label_quota + length_quota > cap:
        length_quota = max(0, cap - label_quota)

    stage1 = _take_round_robin_from_buckets(by_label, selected_keys, label_quota)
    kept.extend(stage1)

    stage2_target = max(0, min(length_quota, cap - len(kept)))
    stage2 = _take_round_robin_from_buckets(by_length, selected_keys, stage2_target)
    kept.extend(stage2)

    if len(kept) < cap:
        for c in sorted_candidates:
            if c.key in selected_keys:
                continue
            selected_keys.add(c.key)
            kept.append(c)
            if len(kept) >= cap:
                break

    dropped = len(candidates) - len(kept)
    return kept, dropped


class RunningValueStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_value = float("inf")
        self.max_value = float("-inf")

    def update(self, value: float) -> None:
        v = float(value)
        self.count += 1
        self.sum += v
        self.sum_sq += v * v
        if v < self.min_value:
            self.min_value = v
        if v > self.max_value:
            self.max_value = v

    def to_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }
        mean = self.sum / self.count
        var = max(0.0, self.sum_sq / self.count - mean * mean)
        return {
            "count": int(self.count),
            "min": float(self.min_value),
            "max": float(self.max_value),
            "mean": float(mean),
            "std": float(sqrt(var)),
        }


class HistogramStats:
    def __init__(self, edges: np.ndarray) -> None:
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Histogram edges must be 1D with at least 2 values")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("Histogram edges must be strictly increasing")
        self.edges = edges.astype(np.float64)
        self.counts = np.zeros(self.edges.size - 1, dtype=np.int64)
        self.underflow = 0
        self.overflow = 0

    def update(self, value: float) -> None:
        idx = int(np.searchsorted(self.edges, value, side="right") - 1)
        if idx < 0:
            self.underflow += 1
            return
        if idx >= self.counts.size:
            self.overflow += 1
            return
        self.counts[idx] += 1

    def to_dict(self, total_count: int) -> dict[str, Any]:
        denom = max(1, int(total_count))
        bins: list[dict[str, Any]] = []
        for i, c in enumerate(self.counts.tolist()):
            bins.append(
                {
                    "bin_id": i,
                    "left": float(self.edges[i]),
                    "right": float(self.edges[i + 1]),
                    "count": int(c),
                    "ratio": float(c / denom),
                }
            )
        return {
            "edges": [float(x) for x in self.edges.tolist()],
            "underflow": int(self.underflow),
            "overflow": int(self.overflow),
            "bins": bins,
        }


class DatasetSplitStats:
    def __init__(self, label_edges: np.ndarray, input_len_edges: np.ndarray) -> None:
        self.record_count = 0
        self.question_ids: set[str] = set()
        self.label_stats = RunningValueStats()
        self.prefix_len_stats = RunningValueStats()
        self.input_len_stats = RunningValueStats()
        self.text_char_len_stats = RunningValueStats()
        self.label_hist = HistogramStats(label_edges)
        self.input_len_hist = HistogramStats(input_len_edges)

    def update(
        self,
        question_id: str,
        label: float,
        prefix_token_len: int,
        approx_input_token_len: int,
        text_char_len: int,
    ) -> None:
        self.record_count += 1
        self.question_ids.add(question_id)
        self.label_stats.update(label)
        self.prefix_len_stats.update(prefix_token_len)
        self.input_len_stats.update(approx_input_token_len)
        self.text_char_len_stats.update(text_char_len)
        self.label_hist.update(label)
        self.input_len_hist.update(approx_input_token_len)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": int(self.record_count),
            "question_count_with_samples": int(len(self.question_ids)),
            "label_stats": self.label_stats.to_dict(),
            "prefix_token_len_stats": self.prefix_len_stats.to_dict(),
            "approx_input_token_len_stats": self.input_len_stats.to_dict(),
            "text_char_len_stats": self.text_char_len_stats.to_dict(),
            "label_class_distribution": self.label_hist.to_dict(self.record_count),
            "approx_input_token_len_distribution": self.input_len_hist.to_dict(self.record_count),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seq-reg dataset from tree_gen results")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/tree_gen_results_v2",
        help="Directory of tree generation json files",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default="data/dapo_question_with_suffix_17398.jsonl",
        help="Question jsonl for recovering prompt/messages",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/seq_reg_dataset_v1_0222",
        help="Output directory",
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Tokenizer path/name",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio by question id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--min-support", type=int, default=2, help="Minimum number of branches sharing a prefix")
    parser.add_argument(
        "--newline-mode",
        choices=["single_or_double", "at_least_one"],
        default="single_or_double",
        help="How to treat newline tokens as split points",
    )
    parser.add_argument(
        "--split-point-mode",
        choices=["newline", "tracked_next_token", "newline_or_tracked_next_token"],
        default="newline",
        help=(
            "Where to sample shared prefixes: newline keeps the original behavior; "
            "tracked_next_token matches tree generation expansion points after tracked characters"
        ),
    )
    parser.add_argument(
        "--track-character",
        action="append",
        default=None,
        help=(
            "Character sequence tracked by tree generation. Can be repeated. "
            "Defaults to a blank-line paragraph boundary."
        ),
    )
    parser.add_argument(
        "--track-character-match-mode",
        choices=["token", "stream"],
        default="token",
        help=(
            "How to match tracked character sequences. token matches only inside one "
            "decoded token; stream also matches sequences spanning adjacent tokens."
        ),
    )
    parser.add_argument(
        "--fallback-split-point-mode-if-empty",
        choices=["", "newline", "tracked_next_token", "newline_or_tracked_next_token"],
        default="",
        help=(
            "If a question has no candidates at --min-support, first retry with this "
            "split-point mode while keeping --min-support. Empty disables this fallback."
        ),
    )
    parser.add_argument(
        "--fallback-track-character-match-mode-if-empty",
        choices=["", "token", "stream"],
        default="",
        help=(
            "Optional track-character match mode for the split-point fallback. "
            "Empty reuses --track-character-match-mode."
        ),
    )
    parser.add_argument(
        "--fallback-min-support-if-empty",
        type=int,
        default=0,
        help=(
            "If a question still has no candidates after split-point fallback, rebuild "
            "its candidates with this lower support threshold (0 disables)."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Debug only: cap number of processed files (0 means all)",
    )
    parser.add_argument(
        "--save-hf-dataset",
        action="store_true",
        help="Also save as HuggingFace DatasetDict using datasets.load_dataset(json)",
    )
    parser.add_argument(
        "--score-mode",
        choices=["fast", "verl"],
        default="verl",
        help="Scoring mode: fast(mathd) or verl(mathd+sympy)",
    )
    parser.add_argument(
        "--prefer-precomputed-scores",
        action="store_true",
        help="Prefer branch.score from pre-scored files (e.g., after score_batch_file.py)",
    )
    parser.add_argument(
        "--skip-unscored-files",
        action="store_true",
        help="When using precomputed scores, skip files without complete branch.score",
    )
    parser.add_argument(
        "--min-prefix-gap-tokens",
        type=int,
        default=32,
        help="Prune nearby breakpoints if branch-set+label unchanged and token gap < this value (0 to disable)",
    )
    parser.add_argument(
        "--label-round-decimals",
        type=int,
        default=6,
        help="Label rounding precision used in nearby dedup",
    )
    parser.add_argument(
        "--max-samples-per-question",
        type=int,
        default=0,
        help="Cap samples per question after pruning (0 disables cap)",
    )
    parser.add_argument(
        "--max-samples-strategy",
        choices=["none", "diversity"],
        default="diversity",
        help="Selection strategy when max-samples-per-question is enabled",
    )
    parser.add_argument(
        "--cap-label-bins",
        type=int,
        default=10,
        help="Label bins used by diversity cap strategy",
    )
    parser.add_argument(
        "--cap-length-bins",
        type=int,
        default=5,
        help="Prefix-length bins used by diversity cap strategy",
    )
    parser.add_argument(
        "--cap-label-ratio",
        type=float,
        default=0.5,
        help="Quota ratio for label-bucket sampling in diversity cap strategy",
    )
    parser.add_argument(
        "--cap-length-ratio",
        type=float,
        default=0.3,
        help="Quota ratio for length-bucket sampling in diversity cap strategy",
    )
    parser.add_argument(
        "--max-generated-prefix-tokens",
        type=int,
        default=0,
        help="Only keep divergence points within first N generated tokens (0 disables this filter)",
    )
    args = parser.parse_args()

    if args.cap_label_ratio < 0 or args.cap_length_ratio < 0:
        raise ValueError("cap ratios must be non-negative")
    if args.cap_label_ratio + args.cap_length_ratio > 1.0:
        raise ValueError("cap_label_ratio + cap_length_ratio must be <= 1.0")
    if args.cap_label_bins < 2 or args.cap_length_bins < 2:
        raise ValueError("cap_label_bins and cap_length_bins must be >= 2")
    if args.max_generated_prefix_tokens < 0:
        raise ValueError("max_generated_prefix_tokens must be >= 0")
    if args.fallback_min_support_if_empty < 0:
        raise ValueError("fallback_min_support_if_empty must be >= 0")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(input_dir.glob("*.json"))
    if args.max_files > 0:
        all_files = all_files[: args.max_files]
    if not all_files:
        raise ValueError(f"No json files found in {input_dir}")

    # 可选：仅基于“已打分文件”切分（适合边打分边产出数据）
    if args.prefer_precomputed_scores and args.skip_unscored_files:
        filtered_files: list[Path] = []
        for file_path in tqdm(all_files, desc="Filtering scored files"):
            try:
                with open(file_path, "rb") as f:
                    data = _loads_bytes(f.read())
            except Exception:
                continue
            branches = data.get("branches", [])
            if isinstance(branches, list) and has_full_precomputed_scores(branches):
                filtered_files.append(file_path)
        all_files = filtered_files
        if not all_files:
            raise ValueError("No fully scored files found. Try without --skip-unscored-files.")

    file_ids = [get_file_id(p) for p in all_files]
    train_ids = {qid for qid in file_ids if is_train_id(qid, args.train_ratio, args.seed)}

    print(f"Found {len(all_files)} files")
    print(f"Split by question id: {len(train_ids)} train / {len(file_ids) - len(train_ids)} test")

    print(f"Loading questions from {args.questions_file}")
    questions_map = load_questions_map(Path(args.questions_file))
    print(f"Loaded {len(questions_map)} questions")

    print(f"Loading tokenizer: {args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    track_characters = args.track_character if args.track_character is not None else ["\n\n"]
    detector = SplitPointDetector(
        tokenizer,
        split_point_mode=args.split_point_mode,
        newline_mode=args.newline_mode,
        track_characters=track_characters,
        track_character_match_mode=args.track_character_match_mode,
    )
    fallback_split_detector = None
    fallback_split_point_mode = args.fallback_split_point_mode_if_empty or None
    fallback_track_character_match_mode = (
        args.fallback_track_character_match_mode_if_empty
        or args.track_character_match_mode
    )
    if fallback_split_point_mode:
        fallback_split_detector = SplitPointDetector(
            tokenizer,
            split_point_mode=fallback_split_point_mode,
            newline_mode=args.newline_mode,
            track_characters=track_characters,
            track_character_match_mode=fallback_track_character_match_mode,
        )

    train_seq_jsonl = output_dir / "train_seq.jsonl"
    test_seq_jsonl = output_dir / "test_seq.jsonl"

    train_seq_count = 0
    test_seq_count = 0
    dropped_nearby_same_label = 0
    dropped_by_cap = 0
    skipped_missing_question = 0
    skipped_bad_file = 0
    skipped_unscored_file = 0
    split_point_fallback_question_count = 0
    split_point_fallback_seq_count = 0
    fallback_question_count = 0
    fallback_seq_count = 0
    total_prefix_groups = 0
    total_fallback_prefix_groups = 0
    label_bin_edges = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    input_len_bin_edges = np.array(
        [0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        dtype=np.float64,
    )
    overall_stats = DatasetSplitStats(label_bin_edges, input_len_bin_edges)
    train_stats = DatasetSplitStats(label_bin_edges, input_len_bin_edges)
    test_stats = DatasetSplitStats(label_bin_edges, input_len_bin_edges)

    # 清理旧版 token-level 输出，保证目录里仅保留 seq 数据。
    legacy_train = output_dir / "train.jsonl"
    legacy_test = output_dir / "test.jsonl"
    if legacy_train.exists():
        legacy_train.unlink()
    if legacy_test.exists():
        legacy_test.unlink()

    with open(train_seq_jsonl, "wb") as f_train_seq, open(test_seq_jsonl, "wb") as f_test_seq:
        for file_path in tqdm(all_files, desc="Processing files"):
            qid = get_file_id(file_path)
            q_data = questions_map.get(qid)
            if q_data is None:
                skipped_missing_question += 1
                continue
            try:
                with open(file_path, "rb") as f:
                    data = _loads_bytes(f.read())
            except Exception:
                skipped_bad_file += 1
                continue

            branches = data.get("branches", [])
            answer = data.get("answer", q_data.get("answer", ""))
            messages = q_data.get("messages", [])
            if not isinstance(branches, list) or not branches or not isinstance(messages, list):
                skipped_bad_file += 1
                continue

            outputs = []
            branch_ids = []
            precomputed_scores: list[float] = []
            fully_scored = True
            for b in branches:
                if not isinstance(b, dict):
                    continue
                text = b.get("text", "")
                if not isinstance(text, str):
                    text = str(text)
                outputs.append(text)
                branch_ids.append(str(b.get("branch_id", "")))
                raw_score = b.get("score")
                if _is_number_like(raw_score):
                    precomputed_scores.append(float(raw_score))
                else:
                    fully_scored = False
            if not outputs:
                continue
            if args.prefer_precomputed_scores and args.skip_unscored_files and not fully_scored:
                skipped_unscored_file += 1
                continue

            prompt_text = build_prompt_text(tokenizer, messages)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_token_len = len(prompt_ids)
            output_ids_list: list[list[int]] = tokenizer(outputs)["input_ids"]
            if args.prefer_precomputed_scores and fully_scored:
                scores = precomputed_scores
            else:
                scores = [score_branch(out, answer, args.score_mode) for out in outputs]
            correct_flags = [1 if s > 0.5 else 0 for s in scores]

            groups = build_prefix_groups(
                output_ids_list,
                detector,
                max_generated_prefix_tokens=args.max_generated_prefix_tokens,
            )
            total_prefix_groups += len(groups)

            q_in_train = qid in train_ids
            seq_fp = f_train_seq if q_in_train else f_test_seq

            candidates = build_seq_candidates(
                groups,
                min_support=args.min_support,
                correct_flags=correct_flags,
                branch_ids=branch_ids,
            )
            support_policy = "min_support"
            used_split_point_fallback = False

            if not candidates and fallback_split_detector is not None:
                fallback_groups = build_prefix_groups(
                    output_ids_list,
                    fallback_split_detector,
                    max_generated_prefix_tokens=args.max_generated_prefix_tokens,
                )
                total_fallback_prefix_groups += len(fallback_groups)
                candidates = build_seq_candidates(
                    fallback_groups,
                    min_support=args.min_support,
                    correct_flags=correct_flags,
                    branch_ids=branch_ids,
                )
                if candidates:
                    used_split_point_fallback = True
                    support_policy = "min_support_relaxed_split_point"
                    split_point_fallback_question_count += 1

            used_fallback_support = False
            if not candidates and args.fallback_min_support_if_empty > 0:
                fallback_support_groups = groups
                if fallback_split_detector is not None:
                    fallback_support_groups = build_prefix_groups(
                        output_ids_list,
                        fallback_split_detector,
                        max_generated_prefix_tokens=args.max_generated_prefix_tokens,
                    )
                    total_fallback_prefix_groups += len(fallback_support_groups)
                candidates = build_seq_candidates(
                    fallback_support_groups,
                    min_support=args.fallback_min_support_if_empty,
                    correct_flags=correct_flags,
                    branch_ids=branch_ids,
                )
                if candidates:
                    used_fallback_support = True
                    support_policy = "fallback_min_support_if_empty"
                    fallback_question_count += 1

            kept_candidates, dropped_cnt = prune_nearby_same_label(
                candidates,
                min_prefix_gap_tokens=args.min_prefix_gap_tokens,
                label_round_decimals=args.label_round_decimals,
            )
            dropped_nearby_same_label += dropped_cnt
            kept_candidates, dropped_cap_cnt = cap_candidates_with_diversity(
                kept_candidates,
                max_samples_per_question=args.max_samples_per_question,
                cap_strategy=args.max_samples_strategy,
                label_bins=args.cap_label_bins,
                length_bins=args.cap_length_bins,
                label_ratio=args.cap_label_ratio,
                length_ratio=args.cap_length_ratio,
            )
            dropped_by_cap += dropped_cap_cnt
            if used_split_point_fallback:
                split_point_fallback_seq_count += len(kept_candidates)
            if used_fallback_support:
                fallback_seq_count += len(kept_candidates)

            for c in kept_candidates:
                prefix_text = tokenizer.decode(
                    c.prefix_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                prefix_token_len = len(c.prefix_ids)
                approx_input_token_len = prompt_token_len + prefix_token_len
                seq_text = prompt_text + prefix_text
                seq_record = {
                    "id": f"{qid}::{c.key}",
                    "question_id": qid,
                    "text": seq_text,
                    "label": c.label,
                    "target": c.label,
                    "support_size": c.support_size,
                    "correct_size": c.correct_size,
                    "prefix_token_len": prefix_token_len,
                    "approx_input_token_len": approx_input_token_len,
                    "sample_branch_id": c.sample_branch_id,
                    "support_policy": support_policy,
                }
                write_record(seq_fp, seq_record)
                split_stats = train_stats if q_in_train else test_stats
                overall_stats.update(
                    question_id=qid,
                    label=c.label,
                    prefix_token_len=prefix_token_len,
                    approx_input_token_len=approx_input_token_len,
                    text_char_len=len(seq_text),
                )
                split_stats.update(
                    question_id=qid,
                    label=c.label,
                    prefix_token_len=prefix_token_len,
                    approx_input_token_len=approx_input_token_len,
                    text_char_len=len(seq_text),
                )
                if q_in_train:
                    train_seq_count += 1
                else:
                    test_seq_count += 1

    summary = {
        "input_dir": str(input_dir),
        "questions_file": str(args.questions_file),
        "num_files": len(all_files),
        "train_question_count": len(train_ids),
        "test_question_count": len(file_ids) - len(train_ids),
        "min_support": args.min_support,
        "newline_mode": args.newline_mode,
        "split_point_mode": args.split_point_mode,
        "track_characters": track_characters,
        "track_character_match_mode": args.track_character_match_mode,
        "fallback_split_point_mode_if_empty": fallback_split_point_mode,
        "fallback_track_character_match_mode_if_empty": (
            fallback_track_character_match_mode if fallback_split_point_mode else None
        ),
        "split_point_fallback_question_count": split_point_fallback_question_count,
        "split_point_fallback_seq_records": split_point_fallback_seq_count,
        "fallback_min_support_if_empty": args.fallback_min_support_if_empty,
        "fallback_question_count": fallback_question_count,
        "fallback_seq_records": fallback_seq_count,
        "total_prefix_groups_found": total_prefix_groups,
        "total_fallback_prefix_groups_found": total_fallback_prefix_groups,
        "dropped_nearby_same_label": dropped_nearby_same_label,
        "dropped_by_max_samples_cap": dropped_by_cap,
        "min_prefix_gap_tokens": args.min_prefix_gap_tokens,
        "label_round_decimals": args.label_round_decimals,
        "max_samples_per_question": args.max_samples_per_question,
        "max_samples_strategy": args.max_samples_strategy,
        "cap_label_bins": args.cap_label_bins,
        "cap_length_bins": args.cap_length_bins,
        "cap_label_ratio": args.cap_label_ratio,
        "cap_length_ratio": args.cap_length_ratio,
        "max_generated_prefix_tokens": args.max_generated_prefix_tokens,
        "skipped_missing_question": skipped_missing_question,
        "skipped_bad_file": skipped_bad_file,
        "skipped_unscored_file": skipped_unscored_file,
        "score_mode": args.score_mode,
        "prefer_precomputed_scores": args.prefer_precomputed_scores,
        "skip_unscored_files": args.skip_unscored_files,
        "train_seq_records": train_seq_count,
        "test_seq_records": test_seq_count,
        "final_dataset_stats": {
            "summary_config": {
                "label_bin_edges": [float(x) for x in label_bin_edges.tolist()],
                "input_token_len_bin_edges": [int(x) for x in input_len_bin_edges.tolist()],
                "input_token_len_note": "approx_input_token_len = prompt_token_len + prefix_token_len",
            },
            "overall": overall_stats.to_dict(),
            "train": train_stats.to_dict(),
            "test": test_stats.to_dict(),
        },
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "wb") as f:
        if orjson is not None:
            f.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
        else:
            f.write(json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"))

    if args.save_hf_dataset:
        from datasets import load_dataset

        ds = load_dataset(
            "json",
            data_files={"train": str(train_seq_jsonl), "test": str(test_seq_jsonl)},
        )
        ds.save_to_disk(str(output_dir / "hf_dataset"))

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
