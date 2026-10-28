"""异步树生成器实现。

核心设计：
- 流协程（_stream_task）：只负责 I/O 和状态更新
- 处理协程（_process_task）：定期检查，只处理 delta
- 完全解耦：两个协程独立运行
- 高并发：可轻松处理 1000+ 并发任务
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
import orjson
from loguru import logger
from transformers import PreTrainedTokenizer

from cognitive_probe.tree_generate.core import (
    LLMGenTask,
    LLMGenTaskResult,
    MessageType,
    SamplingParams,
    SchedulerConfig,
    TimeoutConfig,
    TrackedResult,
    TreeGenConfig,
)
from cognitive_probe.utils.tokenize_utils import is_character_in_token, token_id_to_str


@dataclass(slots=True)
class _SchedulerState:
    waiting_tasks: deque[LLMGenTask] = field(default_factory=deque)
    running_tasks: set[str] = field(default_factory=set)
    pending_next_hits: dict[str, deque[int]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    results: dict[str, LLMGenTaskResult] = field(default_factory=dict)
    last_processed_len: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_update_time: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    scheduler_event: asyncio.Event = field(default_factory=asyncio.Event)


class AsyncTreeGenerator:
    """与 OpenAI 兼容客户端通信的树状生成器 - Asyncio 简化版。

    核心设计：
    - 流协程（_stream_task）：只负责 I/O 和状态更新
    - 处理协程（_process_task）：定期检查，只处理 delta
    - 完全解耦：两个协程独立运行
    - 高并发：可轻松处理 1000+ 并发任务
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        base_url: str,
        http_client: httpx.AsyncClient | None = None,
        stream_read_timeout: float = 180.0,
        scheduler_config: SchedulerConfig | None = None,  # 调度器配置
        timeout_config: TimeoutConfig | None = None,  # 超时配置
        default_max_new_tokens: int = 8192,  # 默认最大生成 token 数
        top_logprobs_num: int = 10,  # 返回的 top-k logprobs 候选数
    ):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.base_url = base_url.rstrip("/")
        self.stream_read_timeout = stream_read_timeout
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.timeout_config = timeout_config or TimeoutConfig()
        self.default_max_new_tokens = default_max_new_tokens
        self.top_logprobs_num = top_logprobs_num
        self.counter: dict[str, int] = defaultdict(int)

        # 全局停止信号（对所有 tree_generate 生效）
        self.stop_event = asyncio.Event()

        # 仅用于 stop() 里等待：不依赖锁（单线程事件循环内原子更新）
        self._running_task_count = 0
        self._running_zero_event = asyncio.Event()
        self._running_zero_event.set()

        # HTTP 客户端
        if http_client is None:
            # 默认禁用所有超时；如需超时请显式在 TimeoutConfig 中设置
            timeout = httpx.Timeout(
                connect=self.timeout_config.http_connect,
                read=self.timeout_config.http_read,
                write=self.timeout_config.http_write,
                pool=self.timeout_config.http_pool,
            )
            self.http_client = httpx.AsyncClient(timeout=timeout, trust_env=False, limits=httpx.Limits(max_connections=None, max_keepalive_connections=None))
            self._own_http_client = True
        else:
            self.http_client = http_client
            self._own_http_client = False

    async def stop(self, *, cancel_running: bool = False, wait_timeout: float = 5.0):
        """请求安全停止。

        说明：
        - `tree_generate()` 的调度/队列/结果均为本地 state；stop() 不再尝试“清空等待队列”。
        - stop_event 会让所有流协程/处理协程尽快退出。
        """
        if self.stop_event.is_set():
            return

        self.stop_event.set()

        # cancel_running 语义：尽快停止所有正在进行的请求（通过 stop_event 触发）。
        # 不做强制 cancel，避免 httpx 流读取处于异常状态。
        _ = cancel_running

        try:
            await asyncio.wait_for(
                self._running_zero_event.wait(), timeout=wait_timeout
            )
        except TimeoutError:
            logger.warning(
                f"stop() 超时，仍有 {self._running_task_count} 个生成任务未退出"
            )

        if self._own_http_client:
            await self.http_client.aclose()

    def _create_queue(self) -> asyncio.Queue:
        return asyncio.Queue()

    async def _enqueue_tasks(self, state: _SchedulerState, tasks: list[LLMGenTask]):
        """加入随机延迟以优化 prefix caching。

        策略：
        - 每个任务入队前随机延迟（默认 0-50ms）
        - 让前面的请求先建立 prefix cache
        - 后续请求可以复用缓存，加速处理
        """
        delay_min, delay_max = self.scheduler_config.enqueue_delay_range
        for idx, task in enumerate(tasks):
            if idx > 0 and delay_max > delay_min:
                delay = random.uniform(delay_min, delay_max)
                logger.debug(f"任务 {task.id} 延迟 {delay * 1000:.1f}ms 后入队")
                await asyncio.sleep(delay)

            state.waiting_tasks.append(task)

        # 通知调度器有新任务
        state.scheduler_event.set()

    def _get_input_ids_from_messages(self, messages: list[MessageType]) -> list[int]:
        """将 messages 转换为模型输入的 token ids 列表。"""
        _messages = [asdict(message) for message in messages]
        _chat_template_input = {
            "conversation": _messages,
            "tokenize": True,
        }

        if _messages[-1]["role"] != "assistant":
            _chat_template_input["add_generation_prompt"] = True

        return self.tokenizer.apply_chat_template(**_chat_template_input)  # type: ignore

    async def _stream_task(
        self,
        state: _SchedulerState,
        task_id: str,
        payload: dict[str, Any],
        input_ids: list[int],
        stream_done: asyncio.Event,
    ):
        """流协程：只负责 I/O，持续更新 task_result（全量）。

        职责：
        - 从服务端流式读取
        - 解析 JSON
        - 更新 state.results[task_id]（全量状态）
        """
        try:
            async with self.http_client.stream(
                "POST", f"{self.base_url}/generate", json=payload
            ) as response:
                response.raise_for_status()

                # 使用 aiter_lines 但要检查 stop_event
                # 注意：aiter_lines() 是一个阻塞的异步迭代器，我们需要在每次迭代后检查 stop_event
                async for raw_line in response.aiter_lines():
                    if self.stop_event.is_set():
                        result = state.results.get(task_id)
                        if result:
                            result.is_finished = True
                            result.finish_reason = "cancelled"
                        break

                    if not raw_line:
                        continue

                    line = raw_line.strip()
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        result = state.results.get(task_id)
                        if result:
                            result.is_finished = True
                            result.finish_reason = "stop"
                        break

                    # JSON 解析
                    try:
                        obj = orjson.loads(line)
                    except Exception:
                        continue

                    meta_info = obj.get("meta_info", {}) or {}
                    full_text = obj.get("text", "") or ""
                    output_ids = obj.get("output_ids") or []
                    output_token_logprobs = meta_info.get("output_token_logprobs") or []
                    output_top_logprobs = meta_info.get("output_top_logprobs") or []

                    # 获取当前全量长度
                    new_output_len = (
                        len(output_token_logprobs)
                        if output_token_logprobs
                        else len(output_ids)
                    )

                    # 重建 task_result（全量）并更新共享状态
                    if output_ids:
                        token_ids = [int(tid) for tid in output_ids[:new_output_len]]
                    elif output_token_logprobs:
                        token_ids = [
                            int(entry[1])
                            for entry in output_token_logprobs[:new_output_len]
                        ]
                    else:
                        token_ids = []

                    task_result = LLMGenTaskResult(
                        id=task_id,
                        input_token_ids=input_ids,
                        output=full_text,
                        output_token_ids=token_ids,
                        output_top_logprobs=output_top_logprobs,
                    )

                    # 更新共享状态（处理协程会读取）
                    state.results[task_id] = task_result
                    state.last_update_time[task_id] = time.time()

                    # 检查是否完成
                    finish_reason = None
                    finish_meta = meta_info.get("finish_reason") or {}
                    finish_reason = finish_meta.get("type") or finish_meta or None
                    if finish_reason:
                        task_result.is_finished = True
                        task_result.finish_reason = str(finish_reason)
                        state.results[task_id] = task_result
                        logger.debug(f"任务 {task_id} 完成: {finish_reason}")
                        break

        except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
            # HTTP 客户端关闭时的正常情况
            if self.stop_event.is_set():
                logger.debug(f"[流协程] 任务 {task_id} 因停止信号而中断")
                result = state.results.get(task_id)
                if result and not result.is_finished:
                    result.is_finished = True
                    result.finish_reason = "cancelled"
                    state.results[task_id] = result
            else:
                logger.exception(f"[流协程] 任务 {task_id} HTTP 异常: {exc}")
                result = state.results.get(
                    task_id, LLMGenTaskResult(id=task_id, input_token_ids=input_ids)
                )
                result.is_finished = True
                result.finish_reason = f"http_error:{type(exc).__name__}"
                state.results[task_id] = result
        except Exception as exc:
            logger.exception(f"[流协程] 任务 {task_id} 异常: {exc}")
            result = state.results.get(
                task_id, LLMGenTaskResult(id=task_id, input_token_ids=input_ids)
            )
            result.is_finished = True
            result.finish_reason = f"error:{type(exc).__name__}"
            state.results[task_id] = result

        finally:
            stream_done.set()

    async def _process_task(
        self,
        state: _SchedulerState,
        task_id: str,
        task: LLMGenTask,
        stream_done: asyncio.Event,
    ):
        """处理协程：定期检查，只处理新增的 tokens（delta）。

        职责：
        - 定期检查 state.results[task_id]
        - 只处理新增的 tokens（delta）
        - 发送 tracked_hits 到 output_queue

        与流协程完全解耦
        """
        processing_start = time.time()
        processed_tokens = 0

        try:
            while not stream_done.is_set() and not self.stop_event.is_set():
                # 定期检查是否有更新
                try:
                    await asyncio.wait_for(
                        self.stop_event.wait(),
                        timeout=self.scheduler_config.process_check_interval,
                    )
                    # 如果等到了 stop_event，说明停止了，跳出循环
                    break
                except TimeoutError:
                    # 超时，说明只是到了检查间隔，继续执行检查逻辑
                    pass

                task_result = state.results.get(task_id)
                if not task_result:
                    continue

                # 优先检查是否已生成完成
                if task_result.is_finished:
                    current_len = len(task_result.output_token_ids)
                    last_processed = state.last_processed_len.get(task_id, 0)
                    if current_len <= last_processed:
                        break
                else:
                    # 检查是否有新 tokens
                    current_len = len(task_result.output_token_ids)
                    last_processed = state.last_processed_len.get(task_id, 0)

                    if current_len <= last_processed:
                        continue

                delta_count = current_len - last_processed
                processed_tokens += delta_count

                # 读取待处理的 triggers
                pending_triggers = []
                while state.pending_next_hits[task_id]:
                    pending_triggers.append(
                        state.pending_next_hits[task_id].popleft()
                    )

                # 处理新增的 tokens
                tracked_hits = []
                tokens_to_track = []
                for idx in range(delta_count):
                    global_idx = last_processed + idx
                    if global_idx >= current_len:
                        break

                    token_id = task_result.output_token_ids[global_idx]
                    trigger_token_id = (
                        pending_triggers[idx] if idx < len(pending_triggers) else None
                    )

                    if trigger_token_id is not None:
                        # 从 task_result 直接获取 top_logprobs
                        top_logprobs = task_result.output_top_logprobs
                        top_logprobs_entry = (
                            top_logprobs[global_idx]
                            if global_idx < len(top_logprobs)
                            else None
                        )

                        logprob_map: dict[str, float] = {}
                        if top_logprobs_entry:
                            for top_lp in top_logprobs_entry:
                                lp, tok_id, _tok_text = top_lp
                                if lp is None or tok_id is None:
                                    continue
                                logprob_map[str(tok_id)] = float(lp)

                        trigger_token_str = token_id_to_str(
                            trigger_token_id, self.tokenizer
                        )
                        next_token_str = token_id_to_str(token_id, self.tokenizer)

                        tracked_hits.append(
                            TrackedResult(
                                trigger_token_id=trigger_token_id,
                                trigger_token=trigger_token_str,
                                next_token_id=token_id,
                                next_token=next_token_str,
                                next_token_position=global_idx + 1,
                                logprobs=logprob_map,
                            )
                        )

                    if self._is_track_hit(
                        token_id, task.track_token_ids, task.track_characters
                    ):
                        tokens_to_track.append(token_id)

                # 写入新的待跟踪 tokens
                for tok in tokens_to_track:
                    state.pending_next_hits[task_id].append(tok)

                # 发送 tracked_hits 到 output_queue（触发分支扩展）
                # 即使没有 hits，如果任务完成了也必须发送消息，以便主循环清理 unfinished 状态
                should_notify = (tracked_hits is not None and len(tracked_hits) > 0) or task_result.is_finished

                if task.output_queue and should_notify:
                    await task.output_queue.put((task_id, task_result, tracked_hits))

                # 更新已处理位置
                state.last_processed_len[task_id] = current_len

                # 如果任务完成，退出循环
                if task_result.is_finished:
                    break

        except Exception as exc:
            logger.exception(f"[处理协程] 任务 {task_id} 异常: {exc}")

        finally:
            elapsed = time.time() - processing_start
            logger.debug(
                f"任务 {task_id} 处理完成，耗时 {elapsed:.2f}s，处理 {processed_tokens} tokens"
            )

    async def _process_sg_task_with_state(self, state: _SchedulerState, task: LLMGenTask):
        """包装：启动流协程和处理协程（state 隔离版本）"""
        task_id = task.id
        logger.debug(f"任务 {task_id} 开始处理")

        # 全局计数：用于 stop() 等待
        if self._running_task_count == 0:
            self._running_zero_event.clear()
        self._running_task_count += 1

        # 初始化共享状态
        state.last_processed_len[task_id] = 0
        state.last_update_time[task_id] = time.time()

        try:
            await self._process_sg_task_impl(state, task)
            logger.debug(f"任务 {task_id} 处理完成")
        except Exception as exc:
            logger.exception(f"任务 {task_id} 异常终止: {exc}")
            err_result = state.results.get(task_id, LLMGenTaskResult(id=task_id))
            err_result.is_finished = True
            err_result.finish_reason = f"error:{type(exc).__name__}"
            state.results[task_id] = err_result
            if task.output_queue:
                await task.output_queue.put((task_id, err_result, []))
        finally:
            state.running_tasks.discard(task_id)
            state.last_processed_len.pop(task_id, None)
            state.last_update_time.pop(task_id, None)
            state.pending_next_hits.pop(task_id, None)

            # 任务完成，通知调度器可以调度新任务
            state.scheduler_event.set()

            self._running_task_count -= 1
            if self._running_task_count == 0:
                self._running_zero_event.set()

    async def _process_sg_task_impl(self, state: _SchedulerState, task: LLMGenTask):
        """核心实现：启动两个解耦的协程"""
        task_id = task.id

        # 准备 input_ids
        if task.input_ids is not None:
            input_ids = task.input_ids
        elif task.messages is not None:
            input_ids = self._get_input_ids_from_messages(task.messages)
        else:
            raise ValueError(f"任务 {task_id} 缺少输入")

        logger.debug(f"任务 {task_id} 输入 token 数：{len(input_ids)}")

        # 准备 sampling_params
        sampling_params = {}
        if task.sampling_params:
            sampling_params = asdict(task.sampling_params)
            sampling_params = {
                k: v for k, v in sampling_params.items() if v is not None
            }
        if "max_new_tokens" not in sampling_params:
            sampling_params["max_new_tokens"] = self.default_max_new_tokens

        payload = {
            "rid": task_id,
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "top_logprobs_num": self.top_logprobs_num,
            "stream": True,
        }

        # 初始化 task_result
        state.results[task_id] = LLMGenTaskResult(
            id=task_id, input_token_ids=input_ids
        )

        # 启动两个解耦的协程
        stream_done = asyncio.Event()

        # 流协程：只负责 I/O 和状态更新
        stream_task = asyncio.create_task(
            self._stream_task(state, task_id, payload, input_ids, stream_done)
        )

        # 处理协程：定期检查，只处理 delta
        process_task = asyncio.create_task(
            self._process_task(state, task_id, task, stream_done)
        )

        # 等待两个协程完成
        await asyncio.gather(stream_task, process_task, return_exceptions=True)

    def _is_track_hit(
        self, token_id: int, track_token_ids: list[int], track_characters: list[str]
    ) -> bool:
        """判断当前 token_id 是否命中跟踪列表中的 token id 或包含指定字符。"""
        if track_token_ids and token_id in track_token_ids:
            return True
        for char in track_characters:
            if is_character_in_token(char, token_id, self.tokenizer):
                return True
        return False

    async def tree_generate(
        self,
        llm_task: LLMGenTask,
        tree_config: TreeGenConfig,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[LLMGenTaskResult]:
        """asyncio 树生成：完全异步，高并发

        Args:
            llm_task: 基础 LLM 生成任务
            tree_config: 树生成配置
        """
        tree_gen_start = time.time()
        tree_id = llm_task.id
        logger.info(
            f"树生成任务开始: tree_id={tree_id}, "
            f"initial_branches={tree_config.initial}, "
            f"max_branches={tree_config.max_total_branches}"
        )
        initial_tasks = []
        output_queue: asyncio.Queue = self._create_queue()

        def _emit_progress(event: str):
            if progress_callback is None:
                return
            with contextlib.suppress(Exception):
                progress_callback(
                    {
                        "event": event,
                        "tree_id": tree_id,
                        "created": len(all_task_ids),
                        "finished": len(all_task_ids) - len(all_unfinished_task_ids),
                        "unfinished": len(all_unfinished_task_ids),
                        "running": len(state.running_tasks),
                        "waiting": len(state.waiting_tasks),
                        "target": tree_config.max_total_branches,
                    }
                )

        # 本次 tree_generate 的调度/结果完全隔离（允许多个 tree_generate 并发调用）
        state = _SchedulerState()

        # 为本次树生成创建独立的停止事件
        tree_stop_event = asyncio.Event()

        # 预先计算共享输入前缀（避免在每次扩展时重复调用 tokenizer）
        if llm_task.input_ids is not None:
            share_input_ids = llm_task.input_ids
        elif llm_task.messages is not None:
            share_input_ids = self._get_input_ids_from_messages(llm_task.messages)
        else:
            logger.error("基础任务缺少 input_ids 和 messages")
            return []

        for i in range(tree_config.initial):
            task_id = self._get_sub_task_id(tree_id)
            new_task = LLMGenTask(
                id=task_id,
                messages=llm_task.messages,
                input_ids=llm_task.input_ids,
                sampling_params=llm_task.sampling_params,
                track_token_ids=llm_task.track_token_ids,
                track_characters=llm_task.track_characters,
                output_queue=output_queue,
            )
            initial_tasks.append(new_task)

        await self._enqueue_tasks(state, initial_tasks)

        all_task_ids = {t.id for t in initial_tasks}
        all_unfinished_task_ids = set(all_task_ids)
        non_expanded_tasks: list[LLMGenTask] = []
        last_expand_pos_by_task: dict[str, int] = defaultdict(int)
        _emit_progress("enqueued_initial")

        async def _progress_heartbeat(interval: float = 1.0):
            while not tree_stop_event.is_set() and not self.stop_event.is_set():
                _emit_progress("heartbeat")
                await asyncio.sleep(interval)

        # 为本次树生成创建独立的调度器，传入局部的停止事件
        scheduler_task = asyncio.create_task(
            self._scheduler_loop(state=state, stop_event=tree_stop_event)
        )
        progress_task = (
            asyncio.create_task(_progress_heartbeat())
            if progress_callback is not None
            else None
        )

        # 最小可用：只做“消息驱动的扩展”，不做 stuck 检测与回填
        queue_timeout = self.timeout_config.queue_timeout
        refill_ratio = tree_config.refill_ratio
        if refill_ratio <= 0:
            logger.warning("refill_ratio<=0，自动降级为 1.0 以避免回填死循环")
            refill_ratio = 1.0
        refill_ratio = min(1.0, refill_ratio)

        tail_full_ratio = tree_config.refill_full_remaining_ratio
        tail_full_ratio = min(1.0, max(0.0, tail_full_ratio))
        tail_full_threshold = (
            max(1, int(tree_config.max_total_branches * tail_full_ratio))
            if tree_config.max_total_branches > 0 and tail_full_ratio > 0
            else 0
        )

        while (
            not tree_stop_event.is_set()
            and not self.stop_event.is_set()
            and (
                all_unfinished_task_ids
                or (
                    tree_config.enable_refill
                    and tree_config.max_total_branches > 0
                    and len(all_task_ids) < tree_config.max_total_branches
                )
            )
        ):
            # 回填：当前没有未完成任务，但还没达到 max_total_branches
            if (
                tree_config.enable_refill
                and not all_unfinished_task_ids
                and tree_config.max_total_branches > 0
                and len(all_task_ids) < tree_config.max_total_branches
            ):
                deficit = tree_config.max_total_branches - len(all_task_ids)
                use_tail_full_refill = (
                    tail_full_threshold > 0 and deficit <= tail_full_threshold
                )

                refill_quota = (
                    deficit
                    if use_tail_full_refill or refill_ratio >= 1.0
                    else max(1, int(deficit * refill_ratio))
                )
                refill_budget = min(deficit, refill_quota)
                fill_tasks: list[LLMGenTask] = []

                if non_expanded_tasks and refill_budget > 0:
                    consume = min(refill_budget, len(non_expanded_tasks))
                    sampled, remaining = self._sample_tasks_uniform_by_position(
                        non_expanded_tasks,
                        sample_size=consume,
                        share_input_len=len(share_input_ids),
                    )
                    fill_tasks.extend(sampled)
                    non_expanded_tasks = remaining
                    refill_budget -= len(sampled)

                while refill_budget > 0:
                    new_task_id = self._get_sub_task_id(tree_id)
                    fill_tasks.append(
                        LLMGenTask(
                            id=new_task_id,
                            messages=llm_task.messages,
                            input_ids=llm_task.input_ids,
                            sampling_params=llm_task.sampling_params,
                            track_token_ids=llm_task.track_token_ids,
                            track_characters=llm_task.track_characters,
                            output_queue=output_queue,
                        )
                    )
                    refill_budget -= 1

                if fill_tasks:
                    await self._enqueue_tasks(state, fill_tasks)
                    for t in fill_tasks:
                        all_task_ids.add(t.id)
                        all_unfinished_task_ids.add(t.id)
                    _emit_progress("refill")
                continue

            get_task = asyncio.create_task(output_queue.get())
            stop_task = asyncio.create_task(self.stop_event.wait())

            try:
                if queue_timeout is None:
                    done, pending = await asyncio.wait(
                        [get_task, stop_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    done, pending = await asyncio.wait(
                        [get_task, stop_task],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=queue_timeout,
                    )

                if stop_task in done:
                    break

                if get_task not in done:
                    # timeout：继续等（不做额外检测）
                    continue

                task_id, task_result, hits = get_task.result()

                # finished 统一在这里收敛
                if task_result.is_finished:
                    all_unfinished_task_ids.discard(task_id)
                    _emit_progress("task_finished")

                if hits:
                    split_every_k = max(1, tree_config.split_every_k_tokens)
                    min_next_expand_pos = (
                        last_expand_pos_by_task.get(task_id, 0) + split_every_k
                    )

                    (
                        expand_tasks,
                        _non_expanded,
                        last_selected_expand_pos,
                    ) = await self._create_expand_tasks(
                        llm_task,
                        tree_config,
                        task_result,
                        hits,
                        share_input_ids,
                        output_queue,
                        min_next_expand_pos=min_next_expand_pos,
                    )

                    if last_selected_expand_pos is not None:
                        last_expand_pos_by_task[task_id] = last_selected_expand_pos

                    if _non_expanded:
                        non_expanded_tasks.extend(_non_expanded)

                    # 容量限制
                    if tree_config.max_total_branches > 0:
                        remaining_slots = tree_config.max_total_branches - len(all_task_ids)
                        if remaining_slots <= 0:
                            non_expanded_tasks.extend(expand_tasks)
                            expand_tasks = []
                        elif remaining_slots < len(expand_tasks):
                            non_expanded_tasks.extend(expand_tasks[remaining_slots:])
                            expand_tasks = expand_tasks[:remaining_slots]

                    if expand_tasks:
                        await self._enqueue_tasks(state, expand_tasks)
                        for t in expand_tasks:
                            all_task_ids.add(t.id)
                            all_unfinished_task_ids.add(t.id)
                        _emit_progress("expanded")
            finally:
                for p in (get_task, stop_task):
                    if not p.done():
                        p.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await p

        # 设置本次树生成的停止事件，通知调度器退出
        tree_stop_event.set()

        # 停止进度心跳
        if progress_task is not None:
            progress_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await progress_task

        # 取消并等待本次树生成的调度器
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            logger.debug("调度器任务已取消")
        except Exception as e:
            logger.warning(f"调度器任务结束时异常: {e}")

        all_raw_results = [state.results.get(_id, None) for _id in all_task_ids]

        final_results = []
        # 复用前面已计算的 share_input_ids

        for res in all_raw_results:
            if res is None:
                continue
            if not res.input_token_ids:
                logger.warning(f"跳过任务 {res.id}：缺少 input_ids")
                continue

            full_ids = res.input_token_ids + res.output_token_ids
            if full_ids[: len(share_input_ids)] != share_input_ids:
                logger.warning(f"跳过任务 {res.id}：输入前缀与共享前缀不一致")
                continue
            gen_ids = full_ids[len(share_input_ids) :]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            final_result = LLMGenTaskResult(
                id=res.id,
                output=gen_text,
                delta=res.delta,
                input_token_ids=share_input_ids,
                output_token_ids=gen_ids,
                is_finished=res.is_finished,
                finish_reason=res.finish_reason,
            )
            final_results.append(final_result)

        final_results.sort(key=lambda x: x.id)

        elapsed = time.time() - tree_gen_start
        _emit_progress("done")
        logger.info(
            f"树生成任务完成: tree_id={tree_id}, "
            f"总分支数={len(final_results)}, "
            f"总耗时={elapsed:.2f}s"
        )

        return final_results

    async def _scheduler_loop(
        self,
        *,
        state: _SchedulerState,
        stop_event: asyncio.Event | None = None,
    ):
        """asyncio 任务调度主循环（事件驱动）

        优化：使用事件驱动替代轮询，降低 CPU 占用
        - scheduler_event: 有新任务入队时触发
        - 任务完成时也会触发
        - 使用兜底超时防止事件丢失

        Args:
            stop_event: 可选的停止事件。如果为None，使用self.stop_event
        """
        local_stop = stop_event

        while True:
            if self.stop_event.is_set() or (local_stop is not None and local_stop.is_set()):
                return

            # 清除事件标志
            state.scheduler_event.clear()

            # 处理所有可调度的任务
            while (
                len(state.running_tasks) < self.scheduler_config.max_concurrent_tasks
                and state.waiting_tasks
                and not self.stop_event.is_set()
                and (local_stop is None or not local_stop.is_set())
            ):
                task = state.waiting_tasks.popleft()
                # 先占用并发槽位，避免 create_task 到协程实际运行之间的竞态导致超发
                state.running_tasks.add(task.id)
                try:
                    asyncio.create_task(self._process_sg_task_with_state(state, task))
                except Exception:
                    state.running_tasks.discard(task.id)
                    raise
                logger.debug(f"任务 {task.id} 已提交到调度器")

            # 如果没有可调度的任务，等待新任务、任务完成或停止信号
            if not state.waiting_tasks or len(state.running_tasks) >= self.scheduler_config.max_concurrent_tasks:
                waiters: list[asyncio.Task] = [asyncio.create_task(state.scheduler_event.wait())]
                if local_stop is not None:
                    waiters.append(asyncio.create_task(local_stop.wait()))
                waiters.append(asyncio.create_task(self.stop_event.wait()))

                try:
                    with contextlib.suppress(TimeoutError):
                        done, pending = await asyncio.wait_for(
                            asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED),
                            timeout=self.timeout_config.scheduler_loop_interval * 10,
                        )
                        _ = done
                        for p in pending:
                            p.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await p
                finally:
                    for w in waiters:
                        if not w.done():
                            w.cancel()

    def _get_sub_task_id(self, tree_id: str) -> str:
        """生成子任务 ID。"""
        new_task_id = f"{tree_id}:{self.counter[tree_id]}"
        self.counter[tree_id] += 1
        return new_task_id

    def _sample_tasks_uniform_by_position(
        self,
        tasks: list[LLMGenTask],
        *,
        sample_size: int,
        share_input_len: int,
    ) -> tuple[list[LLMGenTask], list[LLMGenTask]]:
        """按位置（深度）均匀采样任务。

        位置定义：`len(task.input_ids) - share_input_len`。

        在均匀覆盖前提下增加轻微随机性：
        - 先按位置排序
        - 切成 `sample_size` 个等宽区间
        - 每个区间随机选 1 个
        """
        if sample_size <= 0 or not tasks:
            return [], tasks

        if sample_size >= len(tasks):
            return tasks, []

        # 先按位置排序，再做等间隔抽样
        sorted_indices = sorted(
            range(len(tasks)),
            key=lambda idx: (len(tasks[idx].input_ids or []) - share_input_len, idx),
        )

        selected_indices: list[int] = []
        for i in range(sample_size):
            seg_start = int(i * len(sorted_indices) / sample_size)
            seg_end = int((i + 1) * len(sorted_indices) / sample_size)
            if seg_end <= seg_start:
                seg_end = min(len(sorted_indices), seg_start + 1)
            pos_in_sorted = random.randrange(seg_start, seg_end)
            selected_indices.append(sorted_indices[pos_in_sorted])

        selected_set = set(selected_indices)
        sampled = [tasks[idx] for idx in selected_indices]
        remaining = [task for idx, task in enumerate(tasks) if idx not in selected_set]
        return sampled, remaining

    async def _create_expand_tasks(
        self,
        llm_task: LLMGenTask,
        tree_config: TreeGenConfig,
        task_result: LLMGenTaskResult,
        hits: list[TrackedResult],
        share_input_ids: list[int],
        output_queue: asyncio.Queue | None = None,
        min_next_expand_pos: int = 1,
    ) -> tuple[list[LLMGenTask], list[LLMGenTask], int | None]:
        """根据 hits 创建扩展任务。

        重要：扩展任务的 max_new_tokens 需要减去已生成的 token 数，
        以确保总长度不超过原始限制。

        优化：使用预分配和切片减少内存分配
        """
        expand_tasks: list[LLMGenTask] = []
        non_expanded_tasks: list[LLMGenTask] = []
        last_selected_expand_pos: int | None = None

        # 获取原始的 max_new_tokens
        original_max_new_tokens = (
            llm_task.sampling_params.max_new_tokens
            if llm_task.sampling_params and llm_task.sampling_params.max_new_tokens
            else self.default_max_new_tokens
        )

        # 使用传入的 share_input_ids（在 tree_generate 中预先计算一次，避免重复调用 tokenizer）
        share_input_len = len(share_input_ids)

        split_every_k = max(1, tree_config.split_every_k_tokens)
        depth_norm = max(1, tree_config.depth_norm_tokens)

        for hit in hits:
            hit_pos = hit.next_token_position or 0
            if hit_pos <= 0:
                continue

            # 门控：相邻两次实际分叉位置至少间隔 K
            if hit_pos < min_next_expand_pos:
                continue

            # 深度概率缩放：越靠后扩展概率越高（上限为原始 expand_prob）
            depth_ratio = min(1.0, hit_pos / depth_norm)
            effective_expand_prob = min(
                1.0,
                tree_config.expand_prob * (depth_ratio ** tree_config.depth_prob_power),
            )

            # 构建新任务的 input_ids（不包含最后一个 output token）
            # 最后一个 token 会被 next_token 替换
            base_input_ids = list(task_result.input_token_ids)
            # hit_pos 为 1-based，base 取到 next_token 之前
            base_input_ids.extend(task_result.output_token_ids[: hit_pos - 1])

            # 计算新任务的实际生成长度
            # base_input_ids + next_token_id 就是从 share_input_ids 后已经"生成"的所有 token
            # 新任务可以继续生成的 token 数 = original_max_new_tokens - 已生成的数量
            new_task_base_len = len(base_input_ids) + 1  # +1 是即将添加的 next_token_id
            generated_count = new_task_base_len - share_input_len

            remaining_max_new_tokens = max(
                1, original_max_new_tokens - generated_count
            )

            available_next_tokens = [
                _tk_id
                for _tk_id, _tk_value in hit.logprobs.items()
                if int(_tk_id) != hit.next_token_id
                and _tk_value >= tree_config.log_prob_thr
            ]

            if not available_next_tokens:
                continue

            _candidate_new_tasks = []
            for next_token_id_str in available_next_tokens:
                for _ in range(tree_config.expand_n):
                    next_token_id = int(next_token_id_str)

                    # 使用列表复制+追加，而不是拼接
                    candidate_input_ids = base_input_ids.copy()
                    candidate_input_ids.append(next_token_id)

                    new_task_id = self._get_sub_task_id(llm_task.id)

                    # 创建新的 sampling_params，调整 max_new_tokens
                    expand_sampling_params = SamplingParams(
                        temperature=llm_task.sampling_params.temperature if llm_task.sampling_params else 0.7,
                        top_p=llm_task.sampling_params.top_p if llm_task.sampling_params else 0.9,
                        top_k=llm_task.sampling_params.top_k if llm_task.sampling_params else 50,
                        max_new_tokens=remaining_max_new_tokens,
                    )

                    new_task = LLMGenTask(
                        id=new_task_id,
                        input_ids=candidate_input_ids,
                        sampling_params=expand_sampling_params,
                        track_token_ids=llm_task.track_token_ids,
                        track_characters=llm_task.track_characters,
                        output_queue=output_queue,
                    )
                    _candidate_new_tasks.append(new_task)

            # 基于 expand_prob 进行采样
            _candidate_new_tasks = [
                t
                for t in _candidate_new_tasks
                if random.random() < effective_expand_prob
            ]
            random.shuffle(_candidate_new_tasks)
            selected_tasks = _candidate_new_tasks[: tree_config.max_expand]
            expand_tasks.extend(selected_tasks)
            non_expanded_tasks.extend(_candidate_new_tasks[tree_config.max_expand :])

            # 仅在该位置实际生成了分叉任务时，推进下一次允许分叉的最小位置
            if selected_tasks:
                last_selected_expand_pos = hit_pos
                min_next_expand_pos = hit_pos + split_every_k

        return expand_tasks, non_expanded_tasks, last_selected_expand_pos
