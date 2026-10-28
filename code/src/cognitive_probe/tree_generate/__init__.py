"""树状增量生成（Tree Generation）包。

本模块基于 OpenAI 兼容的流式 Chat Completions 接口，实现：
- asyncio 模式的任务执行模型（无锁，单线程并发）
- 有限并发的任务调度（最大并发由 `max_concurrent_tasks` 控制）
- 通过 `tracked_queue` 持续产出增量结果（delta），便于上层进行树状/分枝扩展策略

核心设计：
- 流协程（_stream_task）：只负责 I/O，持续更新 task_result（全量）
- 处理协程（_process_task）：定期检查，只处理新增 tokens（delta）
- 无队列通信：通过共享状态（self.all_results）通信
- 完全解耦：两个协程独立运行
"""

from cognitive_probe.tree_generate.core import (
    LLMGenTask,
    LLMGenTaskResult,
    MessageType,
    SamplingParams,
    SchedulerConfig,
    TimeoutConfig,
    TrackedResult,
    TreeGenBranchForOutput,
    TreeGenConfig,
    TreeGenResultForOutput,
)
from cognitive_probe.tree_generate.generator import AsyncTreeGenerator

__all__ = [
    "MessageType",
    "SamplingParams",
    "TimeoutConfig",
    "SchedulerConfig",
    "TreeGenConfig",
    "LLMGenTask",
    "LLMGenTaskResult",
    "TrackedResult",
    "TreeGenBranchForOutput",
    "TreeGenResultForOutput",
    "AsyncTreeGenerator",
]
