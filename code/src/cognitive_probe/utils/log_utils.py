import sys

from loguru import logger


def _is_rank_zero() -> bool:
    """
    检查当前进程是否为主进程（rank 0）。
    适用于分布式训练环境。
    """
    import torch.distributed as dist

    if not dist.is_available():
        return True
    return not dist.is_initialized() or dist.get_rank() == 0  # ty:ignore


def init_loguru_logger(level: str = "INFO"):
    logger.remove()
    logger.add(
        sys.stdout,
        # format="<level>{level}</level>: "  # 等级
        # "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "  # 颜色>时间
        # # "{process.name} | "  # 进程名
        # # "{thread.name} | "  # 进程名
        # "<cyan>{file}:{line}</cyan> | "  # 文件名:行号
        # # "<cyan>{function}</cyan> | "  # 方法名
        # "<level>{message}</level>",  # 日志内容
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "  # 时间
        "<blue>LOG</blue> "  # 日志类型
        "<level>{level.icon}</level> | "  # 等级
        "<cyan>{file: >10}:{line: <4}</cyan> | "  # 文件名:行号
        "<level>{message}</level>",  # 日志内容
        # backtrace=True,
        diagnose=False,
    )
    # 调整logger level的默认icon
    # 确保可以在控制台显示并具有相同的宽度
    logger.level("TRACE", icon="[T]")
    logger.level("DEBUG", icon="[D]")
    logger.level("INFO", icon="[I]")
    logger.level("SUCCESS", icon="[S]")
    logger.level("WARNING", icon="[W]")
    logger.level("ERROR", icon="[E]")
    logger.level("CRITICAL", icon="[C]")

    # 重新设置 handler 的级别：移除已添加的 sink 并以指定级别重新添加
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<blue>LOG</blue> "
        "<level>{level.icon}</level> | "
        "<cyan>{file: >10}:{line: <4}</cyan> | "
        "<level>{message}</level>",
        diagnose=False,
        level=level,
    )


def init_dist_loguru_logger(level: str = "INFO"):
    """
    初始化 Loguru 日志记录器，仅在主进程（rank 0）中启用日志记录。
    适用于分布式训练环境。
    """
    if _is_rank_zero():
        init_loguru_logger(level)
    else:
        logger.remove()  # 移除所有处理程序，禁用日志记录
