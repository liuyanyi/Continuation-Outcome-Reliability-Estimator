# Data Format

Full training questions are packaged as `data/dapo/train.jsonl` and
`data/dapo/test.jsonl`; see [split details](../data/dapo/README.md).

## Training Question Input

Tree generation uses JSONL rows with:

```json
{
  "id": "train_0",
  "messages": [
    {"role": "user", "content": "Question text"}
  ],
  "answer": "ground truth answer"
}
```

## Tree Generation Output

Each question produces one JSON file:

```json
{
  "id": "train_0",
  "branches": [
    {
      "branch_id": "train_0:0",
      "text": "generated reasoning",
      "finish_reason": "stop",
      "is_finished": true,
      "output_token_count": 1234,
      "score": 1.0
    }
  ],
  "answer": "ground truth answer"
}
```

The `score` field is added by branch scoring.

## Seq-Reg Dataset Row

```json
{
  "id": "train_0::9:hash",
  "question_id": "train_0",
  "text": "prompt plus generated prefix",
  "label": 0.25,
  "target": 0.25,
  "support_size": 4,
  "correct_size": 1,
  "prefix_token_len": 128,
  "approx_input_token_len": 512
}
```

Meaning:

- `support_size = N`: number of branches sharing the prefix.
- `correct_size = k`: number of correct branches under the prefix.
- `label = k / N`: target success probability.

Packaged benchmark files and evaluator selection are listed in [EVALUATION.md](EVALUATION.md).

## Benchmark Input

Benchmark generation uses JSONL rows with:

```json
{
  "id": "sample_0",
  "prompt": [
    {"role": "user", "content": "Question text"}
  ],
  "solution": "ground truth answer",
  "judge_type": "math",
  "judge_payload": {},
  "metadata": {}
}
```

## Generation Result Row

```json
{
  "example_id": "sample_0",
  "prompt": [{"role": "user", "content": "Question text"}],
  "input_text": "rendered chat template",
  "input_ids": [1, 2, 3],
  "answer": "ground truth answer",
  "seed": 123,
  "response": "model output",
  "output_ids": [4, 5, 6],
  "response_len_tokens": 3,
  "probs": [],
  "finish_reason": "stop",
  "judge_type": "math",
  "judge_payload": {},
  "metadata": {}
}
```

Probe, DeepConf, and REFRAIN scoring write token-aligned scores into `probs`.

## V2 Evaluation Artifacts

A run directory should contain:

- `args.json`
- `generation_results.jsonl`
- `run_overview.json`
- `generation_segments.jsonl`
- `decision_log.jsonl`
- `final_outputs.jsonl`
- `source_targets.jsonl`
- `eval_report_v2_{k}.json`

