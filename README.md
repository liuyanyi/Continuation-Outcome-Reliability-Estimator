# Continuation-Outcome-Reliability-Estimator(CORE)

**Knowing When to Continue: Step-Boundary Continuation Estimation for LLM Quality Control**

**Paper status: Under Review**

## Overview

Long reasoning traces can continue consuming compute after intermediate errors, making it difficult to decide whether further generation is worthwhile. CORE (**Continuation Outcome Reliability Estimator**) estimates, at each reasoning boundary, the probability that continuing from the current prefix will eventually produce a correct final answer.

CORE targets verifiable reasoning tasks, including mathematical reasoning, multiple-choice question answering, and code generation with executable tests. Task-specific verifiers provide offline supervision; at inference time, the learned estimator supplies a continuation-quality signal without additional rollout sampling or online verification.

## Method and Contributions

- **Rollout-tree distillation.** Prefix-shared rollout trees aggregate terminal correctness into boundary-level Binomial count supervision, turning expensive offline continuation outcomes into reusable training targets.
- **Lightweight Internal-State Estimator.** A small MLP combines the host model's last-layer hidden state, top-2 logit margin, and normalized entropy. A rollout-support-weighted soft-label objective trains it to predict continuation success.
- **Inference-time control.** EMA-Below detects persistently low continuation scores. Armed n-out-of-m (ANoM) adds an arming gate to reduce false pruning under distribution shift. The same score interface supports pruning and routing to a higher-cost model.

## Evaluation Highlights

The paper evaluates CORE with **Qwen3-4B-Instruct-2507** and **Nemotron-Nano-4B** on **AIME 2024**, **AIME 2025**, **GPQA**, and **LiveCodeBench v6**.

In the primary Qwen setting, at operating points selected on held-out data:

| Benchmark | Token reduction vs. unpruned baseline | Baseline Maj@32 | CORE Maj@32 |
| --- | ---: | ---: | ---: |
| AIME 2024 | Approximately 31% | 76.67% | 80.00% |
| AIME 2025 | Approximately 26% | 66.67% | 70.00% |

Maj@32 denotes majority-vote accuracy over 32 sampled trajectories. The paper reports the broader quality-efficiency trade-offs across both host models and all four benchmarks.

## Code & Data

The implementation is located in [`code/`](code/). Installation, configuration, and usage instructions are provided in [`code/README.md`](code/README.md).
