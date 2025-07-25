"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from functools import wraps
from typing import (TYPE_CHECKING, Callable, List, Optional, Tuple, TypeVar,
                    Union)

import torch
from loguru import logger
from typing_extensions import ParamSpec

# import custom ops, trigger op registration
import aphrodite._C  # noqa
import aphrodite.common.envs as envs
from aphrodite.common.logger import log_once
from aphrodite.common.utils import import_pynvml

from .interface import DeviceCapability, Platform, PlatformEnum, _Backend

try:
    from aphrodite.distributed.parallel_state import (
        get_tensor_model_parallel_rank)
except Exception:
    get_tensor_model_parallel_rank = lambda: 0

if TYPE_CHECKING:
    from aphrodite.common.config import AphroditeConfig, ModelConfig


_P = ParamSpec("_P")
_R = TypeVar("_R")

pynvml = import_pynvml()

# pytorch 2.5 uses cudnn sdpa by default, which will cause crash on some models
# see https://github.com/huggingface/diffusers/issues/9704 for details
torch.backends.cuda.enable_cudnn_sdp(False)


def device_id_to_physical_device_id(device_id: int) -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if device_ids == [""]:
            msg = (
                "CUDA_VISIBLE_DEVICES is set to empty string, which means"
                " GPU support is disabled. If you are using ray, please unset"
                " the environment variable `CUDA_VISIBLE_DEVICES` inside the"
                " worker/actor.")
            raise RuntimeError(msg)
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:

    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


class CudaPlatformBase(Platform):
    _enum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @classmethod
    def get_device_capability(cls,
                              device_id: int = 0
                              ) -> Optional[DeviceCapability]:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used")
            return False
        return True

    @classmethod
    def is_fully_connected(cls, device_ids: List[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def log_warnings(cls):
        pass

    @classmethod
    def check_and_update_config(cls,
                                aphrodite_config: "AphroditeConfig") -> None:
        parallel_config = aphrodite_config.parallel_config
        scheduler_config = aphrodite_config.scheduler_config
        compilation_config = aphrodite_config.compilation_config
        model_config = aphrodite_config.model_config

        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                if envs.APHRODITE_USE_V1:
                    raise NotImplementedError(
                        "Multi-step scheduling is not supported (and not "
                        "needed) on Aphrodite V1. Please launch without "
                        "--num-scheduler-steps.")
                else:
                    parallel_config.worker_cls = \
                        "aphrodite.worker.multi_step_worker.MultiStepWorker"
            elif aphrodite_config.speculative_config:
                if envs.APHRODITE_USE_V1:
                    parallel_config.worker_cls = \
                            "aphrodite.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = \
                        "aphrodite.spec_decode.spec_decode_worker.create_spec_worker"
                    parallel_config.sd_worker_cls = \
                        "aphrodite.worker.worker.Worker"
            else:
                if envs.APHRODITE_USE_V1:
                    parallel_config.worker_cls = \
                            "aphrodite.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = "aphrodite.worker.worker.Worker"  # noqa

        cache_config = aphrodite_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        if model_config is not None and model_config.use_mla:
            # if `APHRODITE_ATTENTION_BACKEND` is not set and we are using MLA,
            # then we default to FlashMLA backend, so we need to force the
            # blocksize here
            use_flashmla = (envs.APHRODITE_ATTENTION_BACKEND is None \
                or envs.APHRODITE_ATTENTION_BACKEND == "FLASHMLA")
            from aphrodite.attention.ops.flashmla import is_flashmla_supported
            if use_flashmla and is_flashmla_supported()[0] \
                and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLA backend.")

        if (parallel_config.data_parallel_size > 1
                and compilation_config.use_cudagraph):
            logger.info(
                "Data Parallel: Forcing enforce eager to be True since DP is "
                "currently not supported with CUDA Graphs.")
            aphrodite_config.model_config.enforce_eager = True
            compilation_config.use_cudagraph = False

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1,
                             use_mla) -> str:
        if use_mla:
            # TODO(lucas): refactor to  be more concise
            #  we should probably consider factoring out V1 here
            if selected_backend == _Backend.TRITON_MLA or block_size != 64:
                if use_v1:
                    log_once("INFO", "Using Triton MLA backend on V1 engine.")
                    return ("aphrodite.v1.attention.backends.mla."
                            "triton_mla.TritonMLABackend")
                else:
                    logger.info("Using Triton MLA backend.")
                    return ("aphrodite.attention.backends."
                            "triton_mla.TritonMLABackend")
            else:
                from aphrodite.attention.backends.flashmla import (
                    is_flashmla_supported)
                if not is_flashmla_supported()[0]:
                    logger.warning(
                        "FlashMLA backend is not supported due to {}",
                        is_flashmla_supported()[1])
                elif block_size != 64:
                    logger.warning(
                        "FlashMLA backend is not supported for block size {}"
                        " (currently only supports block size 64).",
                        block_size)
                else:
                    if use_v1:
                        log_once("INFO",
                            "Using FlashMLA backend on V1 engine.")
                        return ("aphrodite.v1.attention.backends.mla."
                                "flashmla.FlashMLABackend")
                    else:
                        logger.info("Using FlashMLA backend.")
                        return ("aphrodite.attention.backends."
                                "flashmla.FlashMLABackend")
        if use_v1:
            if selected_backend == _Backend.FLASHINFER:
                log_once("INFO", "Using FlashInfer backend on V1 engine.")
                return ("aphrodite.v1.attention.backends."
                        "flashinfer.FlashInferBackend")
            if selected_backend == _Backend.TRITON_ATTN_APHRODITE_V1:
                log_once("INFO", "Using Triton backend on V1 engine.")
                return ("aphrodite.v1.attention.backends."
                        "triton_attn.TritonAttentionBackend")
            if cls.has_device_capability(80):
                if get_tensor_model_parallel_rank() == 0:
                    log_once(
                        "INFO", "Using Flash Attention backend on V1 engine.")
                return ("aphrodite.v1.attention.backends."
                        "flash_attn.FlashAttentionBackend")
        if selected_backend == _Backend.FLASHINFER:
            logger.info("Using FlashInfer backend.")
            return "aphrodite.attention.backends.flashinfer.FlashInferBackend"
        elif selected_backend == _Backend.XFORMERS:
            logger.info("Using XFormers backend.")
            return "aphrodite.attention.backends.xformers.XFormersBackend"
        elif selected_backend == _Backend.FLASH_ATTN:
            pass
        elif selected_backend:
            raise ValueError(
                f"Invalid attention backend for {cls.device_name}, "
                f"with use_v1: {use_v1} use_mla: {use_mla}")

        target_backend = _Backend.FLASH_ATTN
        if not cls.has_device_capability(80):
            # Volta and Turing NVIDIA GPUs.
            logger.info(
                "Cannot use FlashAttention-2 backend for Volta and Turing "
                "GPUs.")
            target_backend = _Backend.XFORMERS
        elif dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention-2 backend for dtype other than "
                "torch.float16 or torch.bfloat16.")
            target_backend = _Backend.XFORMERS
        elif block_size % 16 != 0:
            logger.info(
                "Cannot use FlashAttention-2 backend for block size not "
                "divisible by 16.")
            target_backend = _Backend.XFORMERS

        # FlashAttn is valid for the model, checking if the package is
        # installed.
        if target_backend == _Backend.FLASH_ATTN:
            try:
                import aphrodite.aphrodite_flash_attn  # noqa: F401
                from aphrodite.attention.backends.flash_attn import (  # noqa: F401
                    FlashAttentionBackend, flash_attn_supports_fp8)

                supported_sizes = \
                    FlashAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    logger.info(
                        "Cannot use FlashAttention-2 backend for head size {}.",
                        head_size)
                    target_backend = _Backend.XFORMERS
                fp8_kv_cache = (kv_cache_dtype is not None
                                and kv_cache_dtype.startswith("fp8"))
                if (fp8_kv_cache and not flash_attn_supports_fp8()):
                    logger.info(
                        "Cannot use FlashAttention backend for FP8 KV cache.")
                    logger.warning(
                        "Please use FlashInfer backend with FP8 KV Cache for "
                        "better performance by setting environment variable "
                        "APHRODITE_ATTENTION_BACKEND=FLASHINFER")
                    target_backend = _Backend.XFORMERS
            except ImportError:
                logger.info(
                    "Cannot use FlashAttention-2 backend because the "
                    "aphrodite.aphrodite_flash_attn package is not found. "
                    "Make sure that aphrodite_flash_attn was built and "
                    "installed (on by default).")
                target_backend = _Backend.XFORMERS

        if target_backend == _Backend.XFORMERS:
            logger.info("Using XFormers backend.")
            return "aphrodite.attention.backends.xformers.XFormersBackend"

        logger.info("Using Flash Attention backend.")
        return "aphrodite.attention.backends.flash_attn.FlashAttentionBackend"

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "aphrodite.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "aphrodite.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa

    @classmethod
    def supports_fp8(cls) -> bool:
        return cls.has_device_capability(89)

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        return True

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return True


# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA
class NvmlCudaPlatform(CudaPlatformBase):

    @classmethod
    @with_nvml_context
    def get_device_capability(cls,
                              device_id: int = 0
                              ) -> Optional[DeviceCapability]:
        try:
            physical_device_id = device_id_to_physical_device_id(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @with_nvml_context
    def has_device_capability(
        cls,
        capability: Union[Tuple[int, int], int],
        device_id: int = 0,
    ) -> bool:
        try:
            return super().has_device_capability(capability, device_id)
        except RuntimeError:
            return False

    @classmethod
    @with_nvml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @with_nvml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return pynvml.nvmlDeviceGetUUID(handle)

    @classmethod
    @with_nvml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_nvml_context
    def is_fully_connected(cls, physical_device_ids: List[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids
        ]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception(
                            "NVLink detection failed. This is normal if"
                            " your machine has no NVLink equipped.")
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetName(handle)

    @classmethod
    @with_nvml_context
    def log_warnings(cls):
        device_ids: int = pynvml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [
                cls._get_physical_device_name(i) for i in range(device_ids)
            ]
            if (len(set(device_names)) > 1
                    and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"):
                logger.warning(
                    "Detected different devices in the system: {}. Please"
                    " make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )


class NonNvmlCudaPlatform(CudaPlatformBase):

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_fully_connected(cls, physical_device_ids: List[int]) -> bool:
        logger.exception(
            "NVLink detection not possible, as context support was"
            " not found. Assuming no NVLink available.")
        return False


# Autodetect either NVML-enabled or non-NVML platform
# based on whether NVML is available.
nvml_available = False
try:
    try:
        pynvml.nvmlInit()
        nvml_available = True
    except Exception:
        # On Jetson, NVML is not supported.
        nvml_available = False
finally:
    if nvml_available:
        pynvml.nvmlShutdown()

CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform

try:
    from sphinx.ext.autodoc.mock import _MockModule

    if not isinstance(pynvml, _MockModule):
        CudaPlatform.log_warnings()
except ModuleNotFoundError:
    CudaPlatform.log_warnings()
