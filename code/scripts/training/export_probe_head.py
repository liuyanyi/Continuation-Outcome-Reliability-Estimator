#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoConfig

from cognitive_probe.probe_model.probe_model import SUPPORTED_MODEL_TYPE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained probe model into a standalone SingleProbeHead directory."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained probe model directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the exported head-only directory.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forwarded to AutoConfig/from_pretrained.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
    )
    model_type = str(config.model_type)
    if model_type not in SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"Unsupported model_type={model_type}. "
            f"Supported: {sorted(SUPPORTED_MODEL_TYPE.keys())}"
        )

    model_cls = SUPPORTED_MODEL_TYPE[model_type]
    model = model_cls.from_pretrained(
        str(model_path),
        config=config,
        trust_remote_code=args.trust_remote_code,
    )
    model.export_score_head(str(output_dir))
    print(f"Exported probe head to: {output_dir}")


if __name__ == "__main__":
    main()
