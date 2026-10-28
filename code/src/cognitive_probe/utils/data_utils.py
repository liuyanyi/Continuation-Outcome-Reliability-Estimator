from typing import TypeVar

import orjson
from loguru import logger
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

T = TypeVar("T", bound=BaseModel)


def load_jsonl_with_model[T: BaseModel](path: str, model_cls: type[T]) -> tuple[list[T], list[dict]]:
    """加载JSONL文件并使用模型校验数据

    Returns:
        tuple[list[T], list[dict]]: (验证成功的数据模型列表, 原始数据字典列表)
    """
    valid_data: list[T] = []
    raw_data: list[dict] = []
    logger.info(f"正在加载文件: {path}")

    # 先计算总行数用于进度条
    with open(path, "rb") as f:
        total_lines = sum(1 for _ in f)

    with open(path, "rb") as f:
        for line in tqdm(f, total=total_lines, desc="加载数据"):
            raw_dict = orjson.loads(line)
            raw_data.append(raw_dict)
            try:
                obj = model_cls.model_validate(raw_dict)
                valid_data.append(obj)
            except ValidationError as e:
                logger.error(f"数据校验失败: {e}")
    return valid_data, raw_data
