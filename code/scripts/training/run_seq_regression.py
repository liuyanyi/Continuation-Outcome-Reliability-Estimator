#!/usr/bin/env python3
"""Train probe_model with Binomial likelihood from (k, N).

This script is intentionally no longer backward-compatible with the old
classification/regression pipeline. It enforces:
- hidden-state probe head (`cognitive_probe.probe_model`)
- frozen backbone
- labels as `(k, N)` where target is `k / N`
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from cognitive_probe.probe_model.probe_model import SUPPORTED_MODEL_TYPE

check_min_version("4.30.0")
require_version("datasets>=2.14.0", "Please install: pip install -U datasets")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: str | None = field(default=None)
    tokenizer_name: str | None = field(default=None)
    cache_dir: str | None = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    token: str | None = field(default=None)
    trust_remote_code: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2")


@dataclass
class DataTrainingArguments:
    dataset_name: str | None = field(default=None)
    dataset_config_name: str | None = field(default=None)
    train_file: str | None = field(default=None)
    validation_file: str | None = field(default=None)
    test_file: str | None = field(default=None)
    text_column: str = field(default="text")
    k_column: str = field(default="correct_size")
    n_column: str = field(default="support_size")
    label_column: str = field(default="label")
    segment_start_column: str | None = field(default=None)
    question_id_column: str | None = field(default="question_id")
    max_seq_length: int = field(default=512)
    pad_to_max_length: bool = field(default=False)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: int | None = field(default=None)
    processed_dataset_cache_dir: str = field(default=".cache/seq_reg_processed")
    validation_split_percentage: int = field(default=5)
    max_train_samples: int | None = field(default=None)
    max_eval_samples: int | None = field(default=None)
    max_predict_samples: int | None = field(default=None)

    def __post_init__(self) -> None:
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or at least one data file.")
        for file_path in (self.train_file, self.validation_file, self.test_file):
            if file_path is None:
                continue
            ext = file_path.split(".")[-1].lower()
            if ext not in {"csv", "json", "jsonl"}:
                raise ValueError(
                    f"Unsupported file extension `{ext}` for `{file_path}`"
                )


@dataclass
class ProbeArguments:
    prob_num_layers: int = field(default=2)
    prob_hidden_size: int | None = field(default=None)
    min_n_value: float = field(default=1.0)
    loss_type: str = field(default="soft_bce")
    loss_weight_mode: str = field(default="sqrt_n")
    normalize_hidden_states_before_head: bool = field(default=True)
    use_hidden_state_feature: bool = field(default=True)
    use_logit_features: bool = field(default=False)
    use_logit_margin_feature: bool = field(default=False)
    use_logit_entropy_feature: bool = field(default=False)
    logit_feature_detach: bool = field(default=True)
    pooling_mode: str = field(default="last_token")
    segment_mean_window_tokens: int = field(default=100)
    unfreeze_last_layer_norm: bool = field(default=False)
    full_finetune_backbone: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    lora_modules_to_save: str = field(default="score")
    save_probe_head_only: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.prob_num_layers <= 0:
            raise ValueError("prob_num_layers must be >= 1")
        if self.prob_hidden_size is not None and self.prob_hidden_size <= 0:
            raise ValueError("prob_hidden_size must be > 0")
        if self.min_n_value <= 0.0:
            raise ValueError("min_n_value must be > 0")
        if self.loss_type not in {"soft_bce", "mse_prob"}:
            raise ValueError("loss_type must be one of: soft_bce, mse_prob")
        if self.loss_weight_mode not in {"none", "sqrt_n", "sqrt_n_normalized", "n"}:
            raise ValueError(
                "loss_weight_mode must be one of: none, sqrt_n, sqrt_n_normalized, n"
            )
        if self.pooling_mode not in {"last_token", "segment_mean"}:
            raise ValueError("pooling_mode must be one of: last_token, segment_mean")
        if self.segment_mean_window_tokens <= 0:
            raise ValueError("segment_mean_window_tokens must be > 0")
        if self.lora_r <= 0:
            raise ValueError("lora_r must be > 0")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be > 0")
        if not (0.0 <= self.lora_dropout <= 1.0):
            raise ValueError("lora_dropout must be in [0, 1]")
        if self.lora_bias not in {"none", "all", "lora_only"}:
            raise ValueError("lora_bias must be one of: none, all, lora_only")
        if self.full_finetune_backbone and self.use_lora:
            raise ValueError(
                "full_finetune_backbone and use_lora cannot both be enabled"
            )
        if self.full_finetune_backbone and self.unfreeze_last_layer_norm:
            raise ValueError(
                "full_finetune_backbone and unfreeze_last_layer_norm cannot both be enabled"
            )


_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


def _is_layer_norm_parameter_name(param_name: str) -> bool:
    name = param_name.lower()
    if "layernorm" in name or "layer_norm" in name:
        return True
    # Include common naming styles such as input_layernorm/post_attention_layernorm.
    return "norm" in name


def _unfreeze_last_transformer_layer_norm(model: torch.nn.Module) -> int:
    layer_indices: list[int] = []
    for name, _ in model.named_parameters():
        match = _LAYER_INDEX_RE.search(name)
        if match is not None:
            layer_indices.append(int(match.group(1)))

    if not layer_indices:
        logger.warning("No transformer layer indices found; skipped last-layer LayerNorm unfreeze.")
        return 0

    last_layer_idx = max(layer_indices)
    unfrozen = 0
    for name, p in model.named_parameters():
        match = _LAYER_INDEX_RE.search(name)
        if match is None or int(match.group(1)) != last_layer_idx:
            continue
        if _is_layer_norm_parameter_name(name):
            p.requires_grad = True
            unfrozen += p.numel()

    if unfrozen == 0:
        logger.warning(
            "Requested unfreeze of last-layer LayerNorm, but found no matching params at layers.%d.",
            last_layer_idx,
        )
    else:
        logger.info(
            "Unfroze last transformer layer LayerNorm params at layers.%d: %d params",
            last_layer_idx,
            unfrozen,
        )
    return unfrozen


def _split_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _freeze_backbone_and_keep_probe_head(
    model: torch.nn.Module,
    *,
    unfreeze_last_layer_norm: bool,
) -> None:
    trainable_prefix = "score"
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name == trainable_prefix or name.startswith(f"{trainable_prefix}."):
            p.requires_grad = True

    if unfreeze_last_layer_norm:
        _unfreeze_last_transformer_layer_norm(model)

    total, trainable = _count_parameters(model)
    if trainable == 0:
        raise ValueError("No trainable parameters left after freezing; expected `score` head.")
    logger.info(
        "Frozen backbone. trainable module=%s, trainable params=%d / %d (%.4f%%)",
        trainable_prefix,
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )


def _prepare_model_for_lora(
    model: torch.nn.Module,
    probe_args: ProbeArguments,
) -> torch.nn.Module:
    target_modules = _split_csv(probe_args.lora_target_modules)
    modules_to_save = _split_csv(probe_args.lora_modules_to_save)
    if not target_modules:
        raise ValueError("lora_target_modules must contain at least one module name")

    lora_bias = cast(Literal["none", "all", "lora_only"], probe_args.lora_bias)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=probe_args.lora_r,
        lora_alpha=probe_args.lora_alpha,
        lora_dropout=probe_args.lora_dropout,
        bias=lora_bias,
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
    )
    lora_model = get_peft_model(cast(Any, model), lora_config)

    total, trainable = _count_parameters(lora_model)
    if trainable == 0:
        raise ValueError("LoRA enabled but no trainable parameters were found")
    logger.info(
        "Enabled LoRA. target_modules=%s, modules_to_save=%s, trainable params=%d / %d (%.4f%%)",
        target_modules,
        modules_to_save,
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )
    return lora_model


def _enable_full_finetune(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True

    total, trainable = _count_parameters(model)
    if trainable == 0:
        raise ValueError("No trainable parameters found after enabling full finetune")
    logger.info(
        "Enabled full finetune. trainable params=%d / %d (%.4f%%)",
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _binomial_nll_from_logits(logits: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    x = logits.astype(np.float64)
    k = k.astype(np.float64)
    n = n.astype(np.float64)
    p = _sigmoid(x)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    one_minus_p = np.clip(np.subtract(1.0, p), 1e-6, 1.0)
    term = k * np.log(p) + (n - k) * np.log(one_minus_p)
    nll = np.multiply(term, -1.0)
    return float(np.mean(nll))


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64).reshape(-1)
    y = y.astype(np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC from scratch (average-rank, tie-aware).

    Returns 0.5 when only one class is present.
    """
    y_true = y_true.astype(np.int64).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    if y_true.size == 0 or y_score.size == 0 or y_true.size != y_score.size:
        return 0.5

    pos_mask = y_true == 1
    n_pos = int(np.sum(pos_mask))
    n_total = int(y_true.size)
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    sorted_ranks = np.empty(n_total, dtype=np.float64)

    i = 0
    while i < n_total:
        j = i + 1
        while j < n_total and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # Ranks are 1-based; average rank for tied block [i, j).
        avg_rank = 0.5 * ((i + 1) + j)
        sorted_ranks[i:j] = avg_rank
        i = j

    ranks = np.empty(n_total, dtype=np.float64)
    ranks[order] = sorted_ranks
    sum_pos_ranks = float(np.sum(ranks[pos_mask]))
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Average Precision (PR-AUC) from scores and binary labels."""
    y_true = y_true.astype(np.int64).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    if y_true.size == 0 or y_score.size == 0 or y_true.size != y_score.size:
        return 0.0

    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tp_cum = np.cumsum(sorted_true == 1)
    fp_cum = np.cumsum(sorted_true == 0)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    ap = float(np.sum(precision[sorted_true == 1]) / n_pos)
    return float(np.clip(ap, 0.0, 1.0))


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = y_true.astype(np.float64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)
    if y_true.size == 0 or y_prob.size == 0 or y_true.size != y_prob.size:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(y_prob.size)
    ece = 0.0
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= left) & (y_prob < right)
        else:
            mask = (y_prob >= left) & (y_prob <= right)
        count = int(np.sum(mask))
        if count == 0:
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (count / total) * abs(conf - acc)
    return float(ece)


def _question_level_topk_keep(
    question_ids: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    keep_ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
) -> list[dict[str, Any]]:
    if len(question_ids) != int(y_true.size) or len(question_ids) != int(y_score.size):
        raise ValueError("question_ids size must match y_true and y_score")

    group_to_indices: dict[str, list[int]] = {}
    for idx, question_id in enumerate(question_ids):
        group_to_indices.setdefault(str(question_id), []).append(idx)

    rows: list[dict[str, Any]] = []
    for frac in keep_ratios:
        per_question_scores: list[float] = []
        total_kept = 0
        for indices in group_to_indices.values():
            local_scores = y_score[indices]
            order = np.argsort(-local_scores, kind="mergesort")
            keep_count = max(1, int(round(float(len(indices)) * frac)))
            chosen = [indices[int(i)] for i in order[:keep_count]]
            total_kept += len(chosen)
            per_question_scores.append(float(np.mean(y_true[chosen])))

        question_count = len(per_question_scores)
        rows.append(
            {
                "keep_ratio": frac,
                "question_count": question_count,
                "total_kept_count": int(total_kept),
                "mean_keep_count_per_question": float(total_kept / max(question_count, 1)),
                "mean_true_success_rate": float(np.mean(per_question_scores)) if per_question_scores else 0.0,
            }
        )
    return rows


def _write_probe_metadata(output_dir: str, payload: dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "created_by": "scripts/training/run_seq_regression.py",
        "task_metadata": payload,
    }
    with open(os.path.join(output_dir, "probe_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, ProbeArguments)  # type: ignore[arg-type]
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, probe_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]),
            allow_extra_keys=True,
        )
    else:
        model_args, data_args, training_args, probe_args = parser.parse_args_into_dataclasses()

    model_args = cast(ModelArguments, model_args)
    data_args = cast(DataTrainingArguments, data_args)
    training_args = cast(TrainingArguments, training_args)
    probe_args = cast(ProbeArguments, probe_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)
    logger.info("Training/evaluation parameters %s", training_args)

    last_checkpoint = None
    assert training_args.output_dir is not None, "Output directory is required."
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and os.listdir(training_args.output_dir):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info("Checkpoint detected, resuming training at %s", last_checkpoint)

    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        data_files: dict[str, str] = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = next(iter(data_files.values())).split(".")[-1].lower()
        if extension == "jsonl":
            extension = "json"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    if "validation" not in raw_datasets:
        if "train" not in raw_datasets:
            raise ValueError("No train split found; cannot create validation split.")
        split = raw_datasets["train"].train_test_split(
            test_size=data_args.validation_split_percentage / 100.0,
            seed=training_args.seed,
        )
        raw_datasets = DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
                **({"test": raw_datasets["test"]} if "test" in raw_datasets else {}),
            }
        )

    sample_split = next(iter(raw_datasets.keys()))
    sample_columns = set(raw_datasets[sample_split].column_names)
    for required in (data_args.text_column, data_args.k_column, data_args.n_column):
        if required not in sample_columns:
            raise ValueError(
                f"Required column `{required}` not found. Available columns: {sorted(sample_columns)}"
            )
    if (
        data_args.segment_start_column is not None
        and data_args.segment_start_column not in sample_columns
    ):
        raise ValueError(
            f"segment_start_column `{data_args.segment_start_column}` not found. "
            f"Available columns: {sorted(sample_columns)}"
        )

    # Apply explicit sample caps before tokenization so smoke tests and capped eval
    # do not spend time preprocessing the full source files.
    if (
        data_args.max_train_samples is not None
        or data_args.max_eval_samples is not None
        or data_args.max_predict_samples is not None
    ):
        capped_datasets = DatasetDict(raw_datasets)
        if data_args.max_train_samples is not None and "train" in capped_datasets:
            capped_datasets["train"] = capped_datasets["train"].select(
                range(min(len(capped_datasets["train"]), data_args.max_train_samples))
            )
        if data_args.max_eval_samples is not None and "validation" in capped_datasets:
            capped_datasets["validation"] = capped_datasets["validation"].select(
                range(min(len(capped_datasets["validation"]), data_args.max_eval_samples))
            )
        if data_args.max_predict_samples is not None and "test" in capped_datasets:
            capped_datasets["test"] = capped_datasets["test"].select(
                range(min(len(capped_datasets["test"]), data_args.max_predict_samples))
            )
        raw_datasets = capped_datasets

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.num_labels = 1
    config.problem_type = "regression"
    task_params = dict(getattr(config, "task_specific_params", {}) or {})
    task_params["prob_num_layers"] = probe_args.prob_num_layers
    task_params["prob_hidden_size"] = (
        probe_args.prob_hidden_size
        if probe_args.prob_hidden_size is not None
        else int(config.hidden_size)
    )
    task_params["loss_type"] = probe_args.loss_type
    task_params["loss_weight_mode"] = probe_args.loss_weight_mode
    task_params["normalize_hidden_states_before_head"] = (
        probe_args.normalize_hidden_states_before_head
    )
    task_params["use_hidden_state_feature"] = probe_args.use_hidden_state_feature
    # Keep old combined switch for backward compatibility while enabling independent features.
    task_params["use_logit_features"] = probe_args.use_logit_features
    task_params["use_logit_margin_feature"] = probe_args.use_logit_margin_feature
    task_params["use_logit_entropy_feature"] = probe_args.use_logit_entropy_feature
    task_params["logit_feature_detach"] = probe_args.logit_feature_detach
    task_params["pooling_mode"] = probe_args.pooling_mode
    task_params["segment_mean_window_tokens"] = probe_args.segment_mean_window_tokens
    config.task_specific_params = task_params

    if config.model_type not in SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"Unsupported model_type={config.model_type}. Supported: {sorted(SUPPORTED_MODEL_TYPE.keys())}"
        )
    model_cls = SUPPORTED_MODEL_TYPE[config.model_type]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has no pad_token or eos_token; set a pad token before training."
            )
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer has no pad_token; using eos_token as pad_token.")
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
    )

    if probe_args.use_lora and probe_args.unfreeze_last_layer_norm:
        logger.warning(
            "`unfreeze_last_layer_norm` is ignored when `use_lora=true` in this script."
        )

    if probe_args.use_lora:
        model = _prepare_model_for_lora(model, probe_args)
    elif probe_args.full_finetune_backbone:
        _enable_full_finetune(model)
    else:
        _freeze_backbone_and_keep_probe_head(
            model,
            unfreeze_last_layer_norm=probe_args.unfreeze_last_layer_norm,
        )
    total_params, trainable_params = _count_parameters(model)

    if data_args.pad_to_max_length:
        padding: bool | str = "max_length"
    else:
        padding = False

    def preprocess_function(examples: dict[str, list[Any]]) -> dict[str, Any]:
        texts = examples[data_args.text_column]
        tokenized = tokenizer(
            texts,
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

        k = np.asarray(examples[data_args.k_column], dtype=np.float32)
        n = np.asarray(examples[data_args.n_column], dtype=np.float32)
        n = np.maximum(n, probe_args.min_n_value)
        k = np.clip(k, 0.0, n)
        tokenized["labels"] = k.tolist()
        tokenized["label_weights"] = n.tolist()
        if data_args.segment_start_column is not None:
            seg_start = np.asarray(examples[data_args.segment_start_column], dtype=np.int64)
            tokenized["segment_start_positions"] = np.maximum(seg_start, 0).tolist()
        return tokenized

    source_signature: dict[str, Any] = {
        "dataset_name": data_args.dataset_name,
        "dataset_config_name": data_args.dataset_config_name,
        "train_file": data_args.train_file,
        "validation_file": data_args.validation_file,
        "test_file": data_args.test_file,
    }
    for split_name, path in {
        "train_file": data_args.train_file,
        "validation_file": data_args.validation_file,
        "test_file": data_args.test_file,
    }.items():
        if path is not None and os.path.exists(path):
            stat = os.stat(path)
            source_signature[f"{split_name}_size"] = int(stat.st_size)
            source_signature[f"{split_name}_mtime_ns"] = int(stat.st_mtime_ns)

    preprocess_cache_key_payload = {
        "source": source_signature,
        "text_column": data_args.text_column,
        "k_column": data_args.k_column,
        "n_column": data_args.n_column,
        "segment_start_column": data_args.segment_start_column,
        "tokenizer": model_args.tokenizer_name or model_args.model_name_or_path,
        "tokenizer_revision": model_args.model_revision,
        "use_fast_tokenizer": model_args.use_fast_tokenizer,
        "max_seq_length": data_args.max_seq_length,
        "pad_to_max_length": data_args.pad_to_max_length,
        "min_n_value": probe_args.min_n_value,
        "max_train_samples": data_args.max_train_samples,
        "max_eval_samples": data_args.max_eval_samples,
        "max_predict_samples": data_args.max_predict_samples,
    }
    preprocess_cache_hash = hashlib.sha1(
        json.dumps(preprocess_cache_key_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_root = os.path.abspath(data_args.processed_dataset_cache_dir)
    os.makedirs(cache_root, exist_ok=True)
    cache_file_names = {
        split_name: os.path.join(cache_root, f"{split_name}-{preprocess_cache_hash}.arrow")
        for split_name in raw_datasets.keys()
    }
    logger.info("Preprocess cache key=%s cache_dir=%s", preprocess_cache_hash, cache_root)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets[sample_split].column_names,
            cache_file_names=cache_file_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = None
    eval_dataset = None
    raw_eval_dataset = None
    predict_dataset = None
    if training_args.do_train:
        train_dataset = processed_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(min(len(train_dataset), data_args.max_train_samples))
            )
    if training_args.do_eval:
        eval_dataset = processed_datasets["validation"]
        raw_eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            keep_count = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(keep_count))
            raw_eval_dataset = raw_eval_dataset.select(range(keep_count))
    if training_args.do_predict and "test" in processed_datasets:
        predict_dataset = processed_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(min(len(predict_dataset), data_args.max_predict_samples))
            )

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=8 if (training_args.fp16 or training_args.bf16) else None,
        )

    # Include N in labels returned by prediction/evaluation for metric calculation.
    training_args.label_names = ["labels", "label_weights"]

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        predictions, label_ids = eval_pred
        logits = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if not isinstance(label_ids, tuple) or len(label_ids) < 2:
            raise ValueError("Expected label_ids tuple(labels, label_weights)")
        k = np.asarray(label_ids[0], dtype=np.float64).reshape(-1)
        n = np.asarray(label_ids[1], dtype=np.float64).reshape(-1)
        n = np.clip(n, 1e-6, None)

        y = np.clip(k / n, 1e-6, 1.0 - 1e-6)
        p = np.clip(_sigmoid(logits), 1e-6, 1.0 - 1e-6)
        y_binary = (y > 0.5).astype(np.int64)
        p_binary = (p >= 0.5).astype(np.int64)

        binomial_nll = _binomial_nll_from_logits(logits=logits, k=k, n=n)
        pearson_corr = _pearson_corr(p, y)
        roc_auc = _binary_roc_auc(y_binary, p)
        pr_auc = _binary_pr_auc(y_binary, p)
        ece_10bin = _expected_calibration_error(y, p, n_bins=10)
        brier_score = float(np.mean((p - y) ** 2))

        tp = float(np.sum((p_binary == 1) & (y_binary == 1)))
        tn = float(np.sum((p_binary == 0) & (y_binary == 0)))
        fp = float(np.sum((p_binary == 1) & (y_binary == 0)))
        fn = float(np.sum((p_binary == 0) & (y_binary == 1)))
        total = max(tp + tn + fp + fn, 1.0)
        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0.0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )
        balanced_accuracy = 0.5 * (recall + specificity)

        return {
            "binomial_nll": binomial_nll,
            "pearson_corr": pearson_corr,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "ece_10bin": ece_10bin,
            "brier_score": brier_score,
            "mean_p": float(np.mean(p)),
            "std_p": float(np.std(p)),
            "mean_support": float(np.mean(n)),
            "pred_threshold": 0.5,
            "label_threshold": 0.5,
            "accuracy": float((tp + tn) / total),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "balanced_accuracy": float(balanced_accuracy),
            "pred_pos_rate": float(np.mean(p_binary)),
            "label_pos_rate": float(np.mean(y_binary)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    if probe_args.use_lora:
        train_mode = "lora+score"
        trainable_module = "lora + score"
        freeze_backbone = False
    elif probe_args.full_finetune_backbone:
        train_mode = "full_finetune"
        trainable_module = "full backbone + score"
        freeze_backbone = False
    elif probe_args.unfreeze_last_layer_norm:
        train_mode = "score+last_layer_norm"
        trainable_module = "score + last transformer layer norm"
        freeze_backbone = True
    else:
        train_mode = "score_only"
        trainable_module = "score"
        freeze_backbone = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
    )

    if training_args.process_index == 0:
        _write_probe_metadata(
            training_args.output_dir,
            {
                "task_type": "probe_binomial",
                "model_type": config.model_type,
                "input_text_column": data_args.text_column,
                "k_column": data_args.k_column,
                "n_column": data_args.n_column,
                "label_column": data_args.label_column,
                "segment_start_column": data_args.segment_start_column,
                "prob_num_layers": probe_args.prob_num_layers,
                "prob_hidden_size": task_params["prob_hidden_size"],
                "pooling_mode": probe_args.pooling_mode,
                "segment_mean_window_tokens": probe_args.segment_mean_window_tokens,
                "freeze_backbone": freeze_backbone,
                "train_mode": train_mode,
                "trainable_module": trainable_module,
                "use_lora": probe_args.use_lora,
                "full_finetune_backbone": probe_args.full_finetune_backbone,
                "lora_r": probe_args.lora_r,
                "lora_alpha": probe_args.lora_alpha,
                "lora_dropout": probe_args.lora_dropout,
                "lora_bias": probe_args.lora_bias,
                "lora_target_modules": _split_csv(probe_args.lora_target_modules),
                "lora_modules_to_save": _split_csv(probe_args.lora_modules_to_save),
                "min_n_value": probe_args.min_n_value,
                "loss_type": probe_args.loss_type,
                "loss_weight_mode": probe_args.loss_weight_mode,
                "normalize_hidden_states_before_head": probe_args.normalize_hidden_states_before_head,
                "use_hidden_state_feature": probe_args.use_hidden_state_feature,
                "use_logit_features": probe_args.use_logit_features,
                "use_logit_margin_feature": probe_args.use_logit_margin_feature,
                "use_logit_entropy_feature": probe_args.use_logit_entropy_feature,
                "logit_feature_detach": probe_args.logit_feature_detach,
                "unfreeze_last_layer_norm": probe_args.unfreeze_last_layer_norm,
                "save_probe_head_only": probe_args.save_probe_head_only,
                "trainable_param_count": int(trainable_params),
                "total_param_count": int(total_params),
                "trainable_ratio": float(trainable_params / max(total_params, 1)),
            },
        )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if probe_args.save_probe_head_only:
            if trainer.is_world_process_zero():
                model_to_save = trainer.model
                if hasattr(model_to_save, "module"):
                    model_to_save = model_to_save.module
                if not hasattr(model_to_save, "export_score_head"):
                    raise ValueError(
                        "save_probe_head_only=true but model does not implement export_score_head"
                    )
                model_to_save.export_score_head(training_args.output_dir)
                tokenizer.save_pretrained(training_args.output_dir)
        else:
            trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples or len(train_dataset)  # type: ignore[arg-type]
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))  # type: ignore[arg-type]
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if eval_dataset is None:
            raise ValueError("do_eval=true but eval_dataset is None")
        eval_output = trainer.predict(cast(Any, eval_dataset), metric_key_prefix="eval")
        metrics = cast(dict[str, float], eval_output.metrics or {})
        max_eval_samples = data_args.max_eval_samples or len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if training_args.process_index == 0:
            logits = np.asarray(eval_output.predictions, dtype=np.float64).reshape(-1)
            label_ids = eval_output.label_ids
            if not isinstance(label_ids, tuple) or len(label_ids) < 2:
                raise ValueError("Expected label_ids tuple(labels, label_weights) in eval output")

            k = np.asarray(label_ids[0], dtype=np.float64).reshape(-1)
            n = np.clip(np.asarray(label_ids[1], dtype=np.float64).reshape(-1), 1e-6, None)
            y = np.clip(k / n, 1e-6, 1.0 - 1e-6)
            p = np.clip(_sigmoid(logits), 1e-6, 1.0 - 1e-6)

            eval_question_ids: list[str] | None = None
            if (
                raw_eval_dataset is not None
                and data_args.question_id_column is not None
                and data_args.question_id_column in raw_eval_dataset.column_names
            ):
                eval_question_ids = [str(x) for x in raw_eval_dataset[data_args.question_id_column]]
                if len(eval_question_ids) != int(len(p)):
                    logger.warning(
                        "question_id count (%d) does not match eval predictions (%d); skip question-level top-k metric.",
                        len(eval_question_ids),
                        len(p),
                    )
                    eval_question_ids = None

            eval_pred_path = os.path.join(training_args.output_dir, "eval_predictions.jsonl")
            with open(eval_pred_path, "w", encoding="utf-8") as f:
                for idx, (logit_v, p_v, k_v, n_v, y_v) in enumerate(zip(logits, p, k, n, y)):
                    row = {
                        "index": int(idx),
                        "logit": float(logit_v),
                        "pred_score": float(p_v),
                        "correct_size": float(k_v),
                        "support_size": float(n_v),
                        "label": float(y_v),
                        "label_binary": int(y_v > 0.5),
                    }
                    if eval_question_ids is not None:
                        row["question_id"] = eval_question_ids[idx]
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("Saved eval predictions to %s", eval_pred_path)

            cm_keys = ("eval_tn", "eval_fp", "eval_fn", "eval_tp")
            if all(key in metrics for key in cm_keys):
                cm_payload = {
                    "labels": ["negative", "positive"],
                    "pred_threshold": float(metrics.get("eval_pred_threshold", 0.5)),
                    "label_threshold": float(metrics.get("eval_label_threshold", 0.5)),
                    "matrix": [
                        [float(metrics["eval_tn"]), float(metrics["eval_fp"])],
                        [float(metrics["eval_fn"]), float(metrics["eval_tp"])],
                    ],
                }
                cm_path = os.path.join(training_args.output_dir, "eval_confusion_matrix.json")
                with open(cm_path, "w", encoding="utf-8") as f:
                    json.dump(cm_payload, f, ensure_ascii=False, indent=2)
                logger.info("Saved eval confusion matrix to %s", cm_path)

            y_binary = (y > 0.5).astype(np.int64)
            n_buckets = [
                ("N <= 5", n <= 5),
                ("5 < N <= 20", (n > 5) & (n <= 20)),
                ("20 < N <= 100", (n > 20) & (n <= 100)),
                ("N > 100", n > 100),
            ]
            bucket_rows: list[dict[str, Any]] = []
            for name, mask in n_buckets:
                count = int(np.sum(mask))
                if count == 0:
                    bucket_rows.append(
                        {
                            "bucket": name,
                            "count": 0,
                            "mean_p": None,
                            "mean_k_over_n": None,
                            "roc_auc": None,
                            "binomial_nll": None,
                        }
                    )
                    continue

                bucket_rows.append(
                    {
                        "bucket": name,
                        "count": count,
                        "mean_p": float(np.mean(p[mask])),
                        "mean_k_over_n": float(np.mean(y[mask])),
                        "roc_auc": float(_binary_roc_auc(y_binary[mask], p[mask])),
                        "binomial_nll": float(_binomial_nll_from_logits(logits[mask], k[mask], n[mask])),
                    }
                )

            topk_rows_global: list[dict[str, Any]] = []
            order = np.argsort(-p, kind="mergesort")
            for frac in (0.1, 0.2, 0.3, 0.5):
                k_keep = max(1, int(round(float(len(p)) * frac)))
                kept = order[:k_keep]
                topk_rows_global.append(
                    {
                        "keep_ratio": frac,
                        "keep_count": int(k_keep),
                        "mean_true_success_rate": float(np.mean(y[kept])),
                    }
                )

            topk_rows_question = (
                _question_level_topk_keep(eval_question_ids, y, p)
                if eval_question_ids is not None
                else None
            )

            extended_payload = {
                "global": {
                    "binomial_nll": float(metrics.get("eval_binomial_nll", _binomial_nll_from_logits(logits, k, n))),
                    "mean_p": float(metrics.get("eval_mean_p", np.mean(p))),
                    "std_p": float(metrics.get("eval_std_p", np.std(p))),
                    "pearson": float(metrics.get("eval_pearson_corr", _pearson_corr(p, y))),
                    "roc_auc": float(metrics.get("eval_roc_auc", _binary_roc_auc(y_binary, p))),
                    "pr_auc": float(metrics.get("eval_pr_auc", _binary_pr_auc(y_binary, p))),
                    "ece_10bin": float(metrics.get("eval_ece_10bin", _expected_calibration_error(y, p, n_bins=10))),
                    "brier_score": float(metrics.get("eval_brier_score", np.mean((p - y) ** 2))),
                    "accuracy_at_0_5": float(metrics.get("eval_accuracy", 0.0)),
                    "balanced_accuracy": float(metrics.get("eval_balanced_accuracy", 0.0)),
                    "pred_pos_rate": float(metrics.get("eval_pred_pos_rate", np.mean(p >= 0.5))),
                    "label_pos_rate": float(metrics.get("eval_label_pos_rate", np.mean(y_binary))),
                    "num_nodes": int(len(n)),
                    "mean_n": float(np.mean(n)),
                    "median_n": float(np.median(n)),
                    "max_n": float(np.max(n)),
                    "min_n": float(np.min(n)),
                },
                "n_bucket": bucket_rows,
                "topk_keep": topk_rows_global,
                "topk_keep_global": topk_rows_global,
                "topk_keep_question": topk_rows_question,
            }

            extended_path = os.path.join(training_args.output_dir, "eval_extended_metrics.json")
            with open(extended_path, "w", encoding="utf-8") as f:
                json.dump(extended_payload, f, ensure_ascii=False, indent=2)
            logger.info("Saved extended eval metrics to %s", extended_path)

    if training_args.do_predict and predict_dataset is not None:
        logger.info("*** Predict ***")
        predictions = trainer.predict(cast(Any, predict_dataset), metric_key_prefix="predict")
        metrics = cast(dict[str, float], predictions.metrics or {})
        max_predict_samples = data_args.max_predict_samples or len(predict_dataset)
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()
