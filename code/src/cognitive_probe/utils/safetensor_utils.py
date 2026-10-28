from pathlib import Path

from loguru import logger
from safetensors import safe_open


def read_safetensors_metadata(file_path: str) -> dict[str, Any] | None:
    """
    读取本地safetensors文件的metadata信息。

    Args:
        file_path: safetensors文件的路径

    Returns:
        包含metadata信息的字典，如果读取失败返回None

    Example:
        >>> metadata = read_safetensors_metadata("model.safetensors")
        >>> print(metadata)
    """
    try:
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            logger.error(f"文件不存在: {file_path}")
            return None

        if file_path_obj.suffix != ".safetensors":
            logger.warning(f"文件扩展名不是.safetensors: {file_path}")

        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            return metadata

    except Exception as e:
        logger.error(f"读取metadata失败: {e}")
        return None
