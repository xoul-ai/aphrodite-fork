import os
from functools import lru_cache

import torch
from loguru import logger

from .interface import DeviceCapability, Platform, PlatformEnum

if os.environ.get("APHRODITE_WORKER_MULTIPROC_METHOD", None) in ["fork", None]:
    logger.warning("`fork` method is not supported by ROCm. "
                   "APHRODITE_WORKER_MULTIPROC_METHOD is overridden to"
                   " `spawn` instead.")
    os.environ["APHRODITE_WORKER_MULTIPROC_METHOD"] = "spawn"


class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory
