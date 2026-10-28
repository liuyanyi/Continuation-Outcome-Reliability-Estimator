"""测试入口。"""

import asyncio
import time

import orjson
import transformers
from loguru import logger
from tqdm.auto import tqdm

from cognitive_probe.tree_generate.core import (
    LLMGenTask,
    MessageType,
    SamplingParams,
    TreeGenBranchForOutput,
    TreeGenConfig,
    TreeGenResultForOutput,
)
from cognitive_probe.tree_generate.generator import AsyncTreeGenerator
from cognitive_probe.utils.log_utils import init_loguru_logger

init_loguru_logger()


async def main():
    """asyncio 测试入口"""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507"
    )

    tree_gen = AsyncTreeGenerator(
        tokenizer=tokenizer,
        base_url="http://localhost:30000",
    )

    llm_task = LLMGenTask(
        id="test_task_1",
        messages=[
            MessageType(
                role="user",
                content="In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
            ),
        ],
        sampling_params=SamplingParams(
            temperature=0.7, top_p=0.9, max_new_tokens=32768
        ),
        track_token_ids=[],
        track_characters=["\n\n", "\n"],
    )

    tree_config = TreeGenConfig(
        initial=8,
        expand_prob=0.005,
        max_total_branches=512,
        split_every_k_tokens=64,
        depth_norm_tokens=1,
        depth_prob_power=1,
        enable_refill=True,
        refill_ratio=0.2,
        refill_full_remaining_ratio=0.05,
    )

    results = []
    progress_total = (
        tree_config.max_total_branches if tree_config.max_total_branches > 0 else None
    )
    pbar = tqdm(
        total=progress_total,
        desc="tree_generate",
        unit="branch",
        mininterval=1.0,
    )
    last_finished = 0
    last_postfix_update = 0.0

    def on_progress(stats: dict):
        nonlocal last_finished, last_postfix_update
        finished = int(stats.get("finished", 0))
        created = int(stats.get("created", 0))
        running = int(stats.get("running", 0))
        waiting = int(stats.get("waiting", 0))

        if pbar.total is None:
            target = int(stats.get("target", 0) or 0)
            if target > 0:
                pbar.total = target

        inc = finished - last_finished
        if inc > 0:
            pbar.update(inc)
            last_finished = finished

        now = time.monotonic()
        if now - last_postfix_update >= 1.0 or inc > 0:
            pbar.set_postfix(
                created=created,
                running=running,
                waiting=waiting,
                refresh=False,
            )
            last_postfix_update = now

    try:
        results = await tree_gen.tree_generate(
            llm_task,
            tree_config,
            progress_callback=on_progress,
        )
    except KeyboardInterrupt:
        logger.info("用户中断，正在取消进行中的请求…")
    except Exception as exc:
        logger.exception(f"tree_generate 异常: {exc}")
    finally:
        pbar.close()
        await tree_gen.stop(cancel_running=True)

    output_result = TreeGenResultForOutput(
        id=llm_task.id,
        branches=[
            TreeGenBranchForOutput(
                branch_id=r.id,
                text=r.output,
                finish_reason=r.finish_reason,
                is_finished=r.is_finished,
                output_token_count=len(r.output_token_ids),
            )
            for r in results
        ],
    )

    with open("tree_gen_results_async0212.json", "wb") as f:
        f.write(
            orjson.dumps(
                {
                    "id": output_result.id,
                    "branches": [
                        {
                            "branch_id": b.branch_id,
                            "text": b.text,
                            "finish_reason": b.finish_reason,
                            "is_finished": b.is_finished,
                            "output_token_count": b.output_token_count,
                        }
                        for b in output_result.branches
                    ],
                },
                option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
