"""核心数据类定义。

包含树生成模块的所有数据类和配置类。
"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MessageType:
    """单条对话消息。

    - role: 角色（system/user/assistant）
    - content: 文本内容
    """

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class SamplingParams:
    """采样参数设置。

    - temperature: 温度系数（采样随机性）
    - top_p: nucleus sampling 截断阈值
    - top_k: 仅保留概率前 k 个候选
    - max_new_tokens: 最多生成 token 数（None 表示不限制，由服务端默认）
    """

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int | None = None


@dataclass
class TimeoutConfig:
    """超时配置。

    - queue_timeout: output_queue.get() 超时时间（秒），None 表示禁用超时
    - scheduler_loop_interval: 调度器循环间隔（秒）
    - http_connect: HTTP 连接超时（秒），None 表示禁用超时
    - http_read: HTTP 读取超时（秒），None 表示使用 stream_read_timeout * 2 或禁用
    - http_write: HTTP 写入超时（秒），None 表示禁用超时
    - http_pool: HTTP 连接池超时（秒），None 表示禁用超时
    """

    queue_timeout: float | None = None
    scheduler_loop_interval: float = 1
    http_connect: float | None = None
    http_read: float | None = None
    http_write: float | None = None
    http_pool: float | None = None


@dataclass
class SchedulerConfig:
    """调度器配置。

    - max_concurrent_tasks: 最大并发任务数
    - process_check_interval: 处理协程检查间隔（秒）
    - enqueue_delay_range: 入队延迟范围（秒），用于优化 prefix caching
    """

    max_concurrent_tasks: int = 32
    process_check_interval: float = 0.1
    enqueue_delay_range: tuple[float, float] = (0.0, 0.05)


@dataclass
class TreeGenConfig:
    """树生成/扩展配置。

    - initial: 初始根节点需要展开的分支数
    - expand_n: 每个候选分支扩展次数
    - max_expand: 每次扩展时允许的最大子分支数
    - log_prob_thr: log概率阈值，用于过滤扩展候选
    - expand_prob: 扩展概率，用于随机采样扩展任务
    - max_total_branches: 最大总分支数限制
    - enable_refill: 是否在自然扩展结束后补齐到 max_total_branches
    - refill_ratio: 每轮回填比例（0~1]；1 表示一次性补满缺口
    - refill_full_remaining_ratio: 当剩余缺口占总量比例不高于该值时，改为单轮全量回填
    - split_every_k_tokens: 同一分支上相邻两次分叉位置至少间隔 K 个 token（1 表示不限制间隔）
    - depth_norm_tokens: 深度概率归一化长度（token）
    - depth_prob_power: 深度概率幂次，>1 更强抑制早期扩展
    """

    initial: int = 1
    expand_n: int = 2
    max_expand: int = 2
    log_prob_thr: float = -10
    expand_prob: float = 0.25
    max_total_branches: int = 16
    enable_refill: bool = True
    refill_ratio: float = 1.0
    refill_full_remaining_ratio: float = 0.1
    split_every_k_tokens: int = 1
    depth_norm_tokens: int = 1
    depth_prob_power: float = 1.0


@dataclass
class LLMGenTask:
    """一次生成任务。

    - id: 任务唯一标识
    - messages: 对话上下文（OpenAI Chat 格式）
    - input_ids: 直接指定的输入 token ids（可选，优先于 messages）
    - sampling_params: 本次任务专用的采样参数（可覆盖全局/默认）
    - track_token_ids: 需要重点追踪的 token ids（可用于上层策略统计/定位）
    - output_queue: 本次生成的增量输出队列（由调用方传入，独立消费）
    """

    id: str
    messages: list[MessageType] | None = None
    input_ids: list[int] | None = None
    sampling_params: SamplingParams | None = None
    track_token_ids: list[int] = field(default_factory=list)
    track_characters: list[str] = field(default_factory=list)
    output_queue: "asyncio.Queue | None" = None


@dataclass
class LLMGenTaskResult:
    """生成任务的流式结果容器。

    - id: 对应任务 id
    - output: 累积的完整输出（含所有增量）
    - delta: 最近一次流式增量（仅当前步新增）
    - input_token_ids: 输入的 token id 序列
    - output_token_ids: 输出对应的 token id 序列（按流式顺序扩展）
    - output_top_logprobs: 每个 token 的 top-k logprobs（处理协程需要）
    - is_finished: 服务端是否已给出结束标记
    - finish_reason: 结束原因（stop、length 等）
    """

    id: str
    output: str = ""
    delta: str = ""
    input_token_ids: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    output_top_logprobs: list = field(default_factory=list)
    is_finished: bool = False
    finish_reason: str | None = None


@dataclass
class TrackedResult:
    """跟踪结果。

    - trigger_token_id: 被跟踪的 token（触发记录下一个 token 的分布）
    - trigger_token: 被跟踪的 token 文本
    - next_token_id: 真正生成的下一个 token id，用于结合 logprobs 解析
    - next_token: 下一个 token 的文本
    - next_token_position: next_token 在 output_token_ids 中的 1-based 位置
    - logprobs: logprob 映射表
    """

    trigger_token_id: int
    trigger_token: str | None = None
    next_token_id: int | None = None
    next_token: str | None = None
    next_token_position: int | None = None
    logprobs: dict[str, float] = field(default_factory=OrderedDict)


@dataclass
class TreeGenBranchForOutput:
    """单个分支的输出结构。"""

    branch_id: str
    text: str
    finish_reason: str | None = None
    is_finished: bool = False
    output_token_count: int = 0


@dataclass
class TreeGenResultForOutput:
    """树生成结果，便于输出。

    - id: 任务 id
    - branches: 结构化分支结果
    """

    id: str
    branches: list[TreeGenBranchForOutput] = field(default_factory=list)
