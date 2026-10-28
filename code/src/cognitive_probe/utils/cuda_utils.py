import torch


def get_cuda_compute_capability() -> str:
    """获取当前CUDA设备的计算能力字符串，例如 "8.0"。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法获取计算能力。")
    device_index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"{major}.{minor}"


def is_ampere_or_newer() -> bool:
    """检查当前CUDA设备是否为Ampere架构或更新的架构（计算能力>=8.0）。"""
    capability_str = get_cuda_compute_capability()
    major, minor = map(int, capability_str.split("."))
    return (major > 8) or (major == 8 and minor >= 0)


def is_volta() -> bool:
    """检查当前CUDA设备是否为Volta架构（计算能力7.0）。"""
    capability_str = get_cuda_compute_capability()
    return capability_str == "7.0"
