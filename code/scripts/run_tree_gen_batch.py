"""批量树生成脚本，支持断点续传。"""

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson
import transformers
from loguru import logger
from tqdm.auto import tqdm

from cognitive_probe.tree_generate.core import (
    LLMGenTask,
    MessageType,
    SamplingParams,
    SchedulerConfig,
    TimeoutConfig,
    TreeGenBranchForOutput,
    TreeGenConfig,
    TreeGenResultForOutput,
)
from cognitive_probe.tree_generate.generator import AsyncTreeGenerator
from cognitive_probe.utils.log_utils import init_loguru_logger

init_loguru_logger("INFO")


def get_existing_ids(output_dir: Path) -> set[str]:
    """获取输出目录中已生成的 id"""
    existing_ids = set()
    for json_file in output_dir.glob("*.json"):
        try:
            with open(json_file, "rb") as f:
                data = orjson.loads(f.read())
                if "id" in data:
                    existing_ids.add(data["id"])
        except Exception as e:
            logger.warning(f"读取文件 {json_file} 失败: {e}")
    return existing_ids

async def process_single_item(
    item: dict,
    output_dir: Path,
    tree_config: TreeGenConfig,
    sampling_params: SamplingParams,
    tree_gen: AsyncTreeGenerator,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> bool:
    """处理单条记录

    Args:
        item: 单条记录数据
        output_dir: 输出目录
        tree_config: 树生成配置
        sampling_params: 采样参数
        tree_gen: 复用的树生成器实例

    Returns:
        bool: 是否成功处理
    """
    item_id = item["id"]
    messages = item["messages"]
    answer = item.get("answer", "")

    try:
        # 构建 LLMGenTask
        llm_task = LLMGenTask(
            id=item_id,
            messages=[
                MessageType(role=msg["role"], content=msg["content"]) for msg in messages
            ],
            sampling_params=sampling_params,
            track_token_ids=[],
            track_characters=["\n\n"],
        )

        # 执行树生成
        results = await tree_gen.tree_generate(
            llm_task,
            tree_config,
            progress_callback=progress_callback,
        )

        # 构建输出结果
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

        # 添加原始 answer 信息到输出
        output_dict = {
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
            "answer": answer,
        }

        # 写入文件（使用 orjson 以获得更好的性能）
        output_file = output_dir / f"{item_id}.json"
        with open(output_file, "wb") as f:
            f.write(
                orjson.dumps(
                    output_dict,
                    option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS,
                )
            )

        logger.info(f"成功处理: {item_id}")
        return True

    except Exception as e:
        logger.error(f"处理 {item_id} 时出错: {e}")
        return False


async def main_async(
    input_file: str = "data/examples/train_questions_tiny.jsonl",
    output_dir: str = "outputs/tree_gen_tiny",
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507",
    base_url: str = "http://localhost:30000",
    initial: int = 2,
    expand_prob: float = 0.01,
    max_total_branches: int = 32,
    batch_concurrent_tasks: int = 10,
    scheduler_max_concurrent_tasks: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 15000,
    split_every_k_tokens: int = 64,
    depth_norm_tokens: int = 64,
    depth_prob_power: float = 1.5,
    enable_refill: bool = True,
    refill_ratio: float = 0.2,
    refill_full_remaining_ratio: float = 0.05,
    disable_timeout: bool = False,
):
    """批量处理入口（纯 asyncio 版本）"""
    # 全局共享 tokenizer（只加载一次）
    logger.info(f"正在加载 tokenizer: {model_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    logger.info("tokenizer 加载完成")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取已生成的 id
    existing_ids = get_existing_ids(output_path)
    logger.info(f"发现已生成的记录数: {len(existing_ids)}")

    # 读取输入文件
    logger.info(f"开始读取输入文件: {input_file}")
    data_list = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))

    total_count = len(data_list)
    logger.info(f"输入文件总记录数: {total_count}")

    # 过滤掉已生成的记录
    pending_items = [item for item in data_list if item["id"] not in existing_ids]
    pending_count = len(pending_items)
    skipped_count = total_count - pending_count

    logger.info(f"跳过已生成的记录数: {skipped_count}")
    logger.info(f"待处理的记录数: {pending_count}")

    if pending_count == 0:
        logger.info("所有记录均已生成，无需处理")
        return

    # 配置树生成参数
    tree_config = TreeGenConfig(
        initial=initial,
        expand_prob=expand_prob,
        max_total_branches=max_total_branches,
        split_every_k_tokens=split_every_k_tokens,
        depth_norm_tokens=depth_norm_tokens,
        depth_prob_power=depth_prob_power,
        enable_refill=enable_refill,
        refill_ratio=refill_ratio,
        refill_full_remaining_ratio=refill_full_remaining_ratio,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    # 配置超时（默认全禁用；可通过不开 --disable-timeout 启用一组保守超时）
    if disable_timeout:
        timeout_config = TimeoutConfig(
            queue_timeout=None,
            http_connect=None,
            http_read=None,
            http_write=None,
            http_pool=None,
        )
    else:
        timeout_config = TimeoutConfig(
            queue_timeout=30.0,
            http_connect=5.0,
            http_read=120.0,
            http_write=30.0,
            http_pool=30.0,
        )

    tree_gen = AsyncTreeGenerator(
        tokenizer=tokenizer,
        base_url=base_url,
        scheduler_config=SchedulerConfig(
            max_concurrent_tasks=scheduler_max_concurrent_tasks,
            process_check_interval=5,
            enqueue_delay_range=(1, 2),
        ),
        timeout_config=timeout_config,
    )

    success_count = 0
    fail_count = 0
    processed_count = 0

    # 双进度条：总任务 + 子任务（分支）
    total_task_pbar = tqdm(
        total=pending_count,
        desc="batch_tasks",
        unit="task",
        position=0,
        mininterval=1.0,
    )
    branch_total = pending_count * max_total_branches if max_total_branches > 0 else None
    branch_pbar = tqdm(
        total=branch_total,
        desc="branch_tasks",
        unit="branch",
        position=1,
        mininterval=1.0,
    )

    last_finished_by_tree: dict[str, int] = {}
    created_by_tree: dict[str, int] = {}
    running_by_tree: dict[str, int] = {}
    waiting_by_tree: dict[str, int] = {}
    last_branch_postfix_update = 0.0
    last_total_postfix_update = 0.0

    def on_tree_progress(stats: dict[str, Any]):
        nonlocal last_branch_postfix_update
        tree_id = str(stats.get("tree_id", ""))
        if not tree_id:
            return

        finished = int(stats.get("finished", 0))
        created = int(stats.get("created", 0))
        running = int(stats.get("running", 0))
        waiting = int(stats.get("waiting", 0))

        prev_finished = last_finished_by_tree.get(tree_id, 0)
        inc = finished - prev_finished
        if inc > 0:
            branch_pbar.update(inc)
            last_finished_by_tree[tree_id] = finished

        created_by_tree[tree_id] = created
        running_by_tree[tree_id] = running
        waiting_by_tree[tree_id] = waiting

        active_trees = sum(
            1
            for tid in running_by_tree
            if running_by_tree.get(tid, 0) > 0 or waiting_by_tree.get(tid, 0) > 0
        )
        now = time.monotonic()
        if now - last_branch_postfix_update >= 1.0 or inc > 0:
            branch_pbar.set_postfix(
                active_trees=active_trees,
                created=sum(created_by_tree.values()),
                running=sum(running_by_tree.values()),
                waiting=sum(waiting_by_tree.values()),
                refresh=False,
            )
            last_branch_postfix_update = now

    sem = asyncio.Semaphore(batch_concurrent_tasks)

    async def _run_one(item: dict, index: int) -> tuple[str, bool, int]:
        item_id = item["id"]
        async with sem:
            logger.info(f"正在处理 [{index}/{pending_count}]: {item_id}")
            ok = await process_single_item(
                item=item,
                output_dir=output_path,
                tree_config=tree_config,
                sampling_params=sampling_params,
                tree_gen=tree_gen,
                progress_callback=on_tree_progress,
            )
            return item_id, ok, index

    tasks = [
        asyncio.create_task(_run_one(item, idx))
        for idx, item in enumerate(pending_items, 1)
    ]

    logger.info(f"开始批量处理，并发数: {batch_concurrent_tasks}")

    try:
        for fut in asyncio.as_completed(tasks):
            item_id, ok, index = await fut
            processed_count += 1
            if ok:
                success_count += 1
            else:
                fail_count += 1
            total_task_pbar.update(1)
            now = time.monotonic()
            if now - last_total_postfix_update >= 1.0:
                total_task_pbar.set_postfix(
                    success=success_count,
                    fail=fail_count,
                    refresh=False,
                )
                last_total_postfix_update = now
            status = "成功" if ok else "失败"
            logger.info(
                f"完成 [{index}/{pending_count}]: {item_id} - {status} [{processed_count}/{pending_count}]"
            )
    except KeyboardInterrupt:
        logger.info("用户中断，正在取消未完成任务…")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await tree_gen.stop(cancel_running=True)
        total_task_pbar.close()
        branch_pbar.close()

    # 统计结果
    logger.info("=" * 50)
    logger.info("批量处理完成！")
    logger.info(f"总记录数: {total_count}")
    logger.info(f"跳过记录数: {skipped_count}")
    logger.info(f"本次处理数: {pending_count}")
    logger.info(f"成功数: {success_count}")
    logger.info(f"失败数: {fail_count}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 50)
    if fail_count:
        raise RuntimeError(f"Tree generation failed for {fail_count} question(s)")


def main(
    input_file: str = "data/examples/train_questions_tiny.jsonl",
    output_dir: str = "outputs/tree_gen_tiny",
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507",
    base_url: str = "http://localhost:30000",
    initial: int = 2,
    expand_prob: float = 0.01,
    max_total_branches: int = 32,
    batch_concurrent_tasks: int = 10,
    scheduler_max_concurrent_tasks: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 15000,
    split_every_k_tokens: int = 64,
    depth_norm_tokens: int = 64,
    depth_prob_power: float = 1.5,
    enable_refill: bool = True,
    refill_ratio: float = 0.2,
    refill_full_remaining_ratio: float = 0.05,
    disable_timeout: bool = False,
):
    asyncio.run(
        main_async(
            input_file=input_file,
            output_dir=output_dir,
            model_path=model_path,
            base_url=base_url,
            initial=initial,
            expand_prob=expand_prob,
            max_total_branches=max_total_branches,
            batch_concurrent_tasks=batch_concurrent_tasks,
            scheduler_max_concurrent_tasks=scheduler_max_concurrent_tasks,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            split_every_k_tokens=split_every_k_tokens,
            depth_norm_tokens=depth_norm_tokens,
            depth_prob_power=depth_prob_power,
            enable_refill=enable_refill,
            refill_ratio=refill_ratio,
            refill_full_remaining_ratio=refill_full_remaining_ratio,
            disable_timeout=disable_timeout,
        )
    )


def parse_args(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description="批量树生成脚本", allow_abbrev=False)
    parser.add_argument("--config", type=Path, help="JSON config; CLI overrides config, paths are relative to the working directory")
    parser.add_argument("--print-config", action="store_true", help="Print resolved parameters and exit without loading models")
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/examples/train_questions_tiny.jsonl",
        help="输入 jsonl 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tree_gen_tiny",
        help="输出目录路径",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="模型路径",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000",
        help="sglang 服务地址",
    )
    parser.add_argument(
        "--initial",
        type=int,
        default=2,
        help="树生成的初始分支数",
    )
    parser.add_argument(
        "--expand-prob",
        type=float,
        default=0.005,
        help="树生成的扩展概率",
    )
    parser.add_argument(
        "--max-total-branches",
        type=int,
        default=512,
        help="树生成的最大总分支数",
    )
    parser.add_argument(
        "--batch-concurrent-tasks",
        type=int,
        default=1,
        help="批量并发处理的记录数（同时处理多少条记录）",
    )
    parser.add_argument(
        "--scheduler-max-concurrent-tasks",
        type=int,
        default=64,
        help="调度器的最大并发任务数（单条记录内的并发）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="采样 top_p",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16000,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--split-every-k-tokens",
        type=int,
        default=64,
        help="同一分支上相邻两次分叉位置至少间隔 K 个 token",
    )
    parser.add_argument(
        "--depth-norm-tokens",
        type=int,
        default=64,
        help="深度概率归一化长度（token）",
    )
    parser.add_argument(
        "--depth-prob-power",
        type=float,
        default=1.5,
        help="深度概率幂次，>1 更抑制浅层分叉",
    )
    parser.add_argument(
        "--disable-refill",
        action="store_true",
        help="关闭自然扩展结束后的回填补齐",
    )
    parser.add_argument(
        "--refill-ratio",
        type=float,
        default=0.2,
        help="每轮回填比例（0~1]",
    )
    parser.add_argument(
        "--refill-full-remaining-ratio",
        type=float,
        default=0.05,
        help="剩余缺口占比不高于该值时，单轮全量回填",
    )
    parser.add_argument(
        "--disable-timeout",
        action="store_true",
        help="禁用所有超时限制",
    )

    parser.add_argument("--enable-refill", action="store_false", dest="disable_refill", default=argparse.SUPPRESS,
                        help="Override disable_refill=true in a config")
    parser.add_argument("--enable-timeout", action="store_false", dest="disable_timeout", default=argparse.SUPPRESS,
                        help="Override disable_timeout=true in a config")
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--config", type=Path)
    known, _ = bootstrap.parse_known_args(argv)
    if known.config:
        try:
            config = json.loads(known.config.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read config: {exc}")
        if not isinstance(config, dict):
            parser.error("Config must be a JSON object")
        actions = {a.dest: a for a in parser._actions if a.dest not in {"help", "config", "print_config"}}
        for key, value in config.items():
            if key not in actions:
                parser.error(f"Unknown config key: {key}")
            action = actions[key]
            expected = action.type or bool
            valid = (type(value) is bool if expected is bool else
                     type(value) is int if expected is int else
                     type(value) in (int, float) if expected is float else
                     isinstance(value, expected))
            if not valid:
                parser.error(f"Invalid type for config key {key}: expected {expected.__name__}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    import math
    for key in ("initial", "batch_concurrent_tasks", "scheduler_max_concurrent_tasks",
                "max_new_tokens", "split_every_k_tokens", "depth_norm_tokens"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    bounds = {
        "expand_prob": (0, 1, False), "top_p": (0, 1, True),
        "refill_ratio": (0, 1, True), "refill_full_remaining_ratio": (0, 1, False),
        "temperature": (0, math.inf, False), "depth_prob_power": (0, math.inf, False),
    }
    for key, (low, high, exclude_low) in bounds.items():
        value = getattr(args, key)
        if not math.isfinite(value) or value < low or value > high or (exclude_low and value == low):
            parser.error(f"Invalid value for {key}: {value}")
    return args


if __name__ == "__main__":
    import json

    args = parse_args()
    if args.print_config:
        print(json.dumps({k: v for k, v in vars(args).items() if k not in {"config", "print_config"}}, indent=2))
        raise SystemExit(0)

    main(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_path=args.model_path,
        base_url=args.base_url,
        initial=args.initial,
        expand_prob=args.expand_prob,
        max_total_branches=args.max_total_branches,
        batch_concurrent_tasks=args.batch_concurrent_tasks,
        scheduler_max_concurrent_tasks=args.scheduler_max_concurrent_tasks,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        split_every_k_tokens=args.split_every_k_tokens,
        depth_norm_tokens=args.depth_norm_tokens,
        depth_prob_power=args.depth_prob_power,
        enable_refill=not args.disable_refill,
        refill_ratio=args.refill_ratio,
        refill_full_remaining_ratio=args.refill_full_remaining_ratio,
        disable_timeout=args.disable_timeout,
    )
