#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from cognitive_probe.probe_model.probe_model import SUPPORTED_MODEL_TYPE


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        raise ValueError(f"JSON file must contain list[object], got {type(payload)}")
    raise ValueError(f"Unsupported file type: {path}")


def _load_probe_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = model_path / "probe_metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("task_metadata", raw)


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
    p = np.clip(_sigmoid(logits), 1e-6, 1.0 - 1e-6)
    return float(np.mean(-(k * np.log(p) + (n - k) * np.log(1.0 - p))))


class _InferDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        text_column: str,
        max_seq_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        text = str(row.get(self.text_column, ""))
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "row_idx": idx,
        }


def _collate(batch: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    input_ids = [x["input_ids"] for x in batch]
    attn = [x["attention_mask"] for x in batch]
    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attn},
        padding=True,
        return_tensors="pt",
    )
    padded["row_idx"] = torch.tensor([x["row_idx"] for x in batch], dtype=torch.long)
    return padded


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for probe_model binomial head")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--text-column", type=str, default="")
    parser.add_argument("--label-column", type=str, default="")
    parser.add_argument("--k-column", type=str, default="")
    parser.add_argument("--n-column", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    metadata = _load_probe_metadata(model_path)

    text_column = args.text_column or metadata.get("input_text_column", "text")
    label_column = args.label_column or metadata.get("label_column", "label")
    k_column = args.k_column or metadata.get("k_column", "correct_size")
    n_column = args.n_column or metadata.get("n_column", "support_size")

    records = _load_records(Path(args.input_file))
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No input records found")

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
    )
    if config.model_type not in SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"Unsupported model_type={config.model_type}. Supported: {sorted(SUPPORTED_MODEL_TYPE.keys())}"
        )
    model_cls = SUPPORTED_MODEL_TYPE[config.model_type]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
    )
    model = model_cls.from_pretrained(
        str(model_path),
        config=config,
        trust_remote_code=args.trust_remote_code,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = _InferDataset(
        records=records,
        tokenizer=tokenizer,
        text_column=text_column,
        max_seq_length=args.max_seq_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: _collate(x, tokenizer),
    )

    logits_all = np.zeros(len(records), dtype=np.float64)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer", unit="batch"):
            row_idx = batch.pop("row_idx").numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.squeeze(-1).detach().cpu().numpy().astype(np.float64)
            logits_all[row_idx] = logits

    pred_prob = _sigmoid(logits_all)
    out_rows: list[dict[str, Any]] = []
    has_k_n = all((k_column in r and n_column in r) for r in records)
    has_label = all((label_column in r) for r in records)

    for i, row in enumerate(records):
        out = {
            "index": i,
            "id": row.get("id"),
            "question_id": row.get("question_id"),
            "pred_logit": float(logits_all[i]),
            "pred_score": float(pred_prob[i]),
        }
        if label_column in row and row[label_column] is not None:
            out["label"] = float(row[label_column])
        if k_column in row and row[k_column] is not None:
            out["k"] = float(row[k_column])
        if n_column in row and row[n_column] is not None:
            out["N"] = float(row[n_column])
        out_rows.append(out)

    summary: dict[str, Any] = {
        "num_samples": int(len(records)),
        "model_path": str(model_path),
        "text_column": text_column,
        "label_column": label_column,
        "k_column": k_column,
        "n_column": n_column,
        "task_type": "probe_binomial",
    }

    if has_label:
        labels = np.asarray([float(r[label_column]) for r in records], dtype=np.float64)
        summary["eval_like_metrics"] = {
            "mae": float(np.mean(np.abs(pred_prob - labels))),
            "rmse": float(np.sqrt(np.mean((pred_prob - labels) ** 2))),
        }

    if has_k_n:
        k = np.asarray([float(r[k_column]) for r in records], dtype=np.float64)
        n = np.asarray([float(r[n_column]) for r in records], dtype=np.float64)
        n = np.clip(n, 1e-6, None)
        y = np.clip(k / n, 1e-6, 1.0 - 1e-6)
        summary.setdefault("eval_like_metrics", {})
        summary["eval_like_metrics"].update(
            {
                "target_mae": float(np.mean(np.abs(pred_prob - y))),
                "target_rmse": float(np.sqrt(np.mean((pred_prob - y) ** 2))),
                "weighted_target_mae": float(np.average(np.abs(pred_prob - y), weights=n)),
                "binomial_nll": _binomial_nll_from_logits(logits_all, k, n),
                "mean_support": float(np.mean(n)),
            }
        )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions: {out_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
