import os
import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING, Optional

import psutil
import torch
from loguru import logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from aphrodite.common.config import AphroditeConfig
else:
    AphroditeConfig = None



class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "cpu"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        if selected_backend and selected_backend != _Backend.TORCH_SDPA:
            logger.info("Cannot use {} backend on CPU.", selected_backend)
        if use_mla:
            logger.info("Using CPU MLA backend.")
            return "aphrodite.attention.backends.cpu_mla.CPUMLABackend"
        logger.info("Using Torch SDPA backend.")
        return "aphrodite.attention.backends.torch_sdpa.TorchSDPABackend"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, aphrodite_config: AphroditeConfig) -> None:
        import aphrodite.common.envs as envs
        from aphrodite.common.utils import GiB_bytes
        model_config = aphrodite_config.model_config
        # Reminder: Please update docs/source/features/compatibility_matrix.md
        # If the feature combo become valid
        if not model_config.enforce_eager:
            model_config.enforce_eager = True

        cache_config = aphrodite_config.cache_config

        ipex_available = find_spec("intel_extension_for_pytorch") is not None

        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 128 if ipex_available else 16

        if not ipex_available and cache_config.block_size != 16:
            raise RuntimeError(
                f"--block-size={cache_config.block_size} requires"
                " intel_extension_for_pytorch")

        scheduler_config = aphrodite_config.scheduler_config
        if ((scheduler_config.chunked_prefill_enabled
             or cache_config.enable_prefix_caching)
                and cache_config.cache_dtype != "auto"):
            raise RuntimeError("Chunked-prefill and prefix-cache on the CPU "
                               "backend is not compatible with FP8 KV cache.")

        if cache_config.cache_dtype == "fp8_e4m3":
            cache_config.cache_dtype = "fp8_e5m2"
            logger.warning(
                "CPU backend doesn't support fp8_e4m3 KV cache type, "
                "cast to fp8_e5m2.")

        if (cache_config.cache_dtype != "auto"
                and model_config.dtype == torch.half):
            logger.warning("FP8 KV cache on the CPU backend only does not"
                           " support fp16 for now, cast to bf16.")
            model_config.dtype = torch.bfloat16

        kv_cache_space = envs.APHRODITE_CPU_KVCACHE_SPACE

        if kv_cache_space >= 0:
            if kv_cache_space == 0:
                cache_config.cpu_kvcache_space_bytes = 4 * GiB_bytes  # type: ignore
                logger.warning(
                    "Environment variable APHRODITE_CPU_KVCACHE_SPACE (GiB) "
                    "for CPU backend is not set, using 4 by default.")
            else:
                cache_config.cpu_kvcache_space_bytes = kv_cache_space * GiB_bytes  # type: ignore # noqa
        else:
            raise RuntimeError(
                "Invalid environment variable APHRODITE_CPU_KVCACHE_SPACE"
                f" {kv_cache_space}, expect a positive integer value.")

        parallel_config = aphrodite_config.parallel_config
        if (parallel_config.distributed_executor_backend is not None
                and parallel_config.distributed_executor_backend != "mp"):
            logger.warning(("{} is not supported on CPU, fallback to mp "
                            "distributed executor backend."),
                           parallel_config.distributed_executor_backend)
            parallel_config.distributed_executor_backend = "mp"
        if parallel_config.worker_cls == "auto":
            if aphrodite_config.speculative_config:
                parallel_config.worker_cls = \
                    "aphrodite.spec_decode.spec_decode_worker.create_spec_worker"
                parallel_config.sd_worker_cls = \
                    "aphrodite.worker.cpu_worker.CPUWorker"
            else:
                parallel_config.worker_cls = "aphrodite.worker.cpu_worker.CPUWorker"  # noqa

        assert aphrodite_config.device_config.device_type == "cpu"

        #
        # Environment variables for CPU executor
        #

        # Set default threads num for OpenMP parallel
        os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Intel OpenMP setting
        ld_prealod_str = os.getenv("LD_PRELOAD", "")
        if "libiomp5.so" in ld_prealod_str:
            # The time(milliseconds) that a thread should wait after
            # completing the execution of a parallel region, before sleeping.
            os.environ['KMP_BLOCKTIME'] = "1"
            # Prevents the CPU to run into low performance state
            os.environ['KMP_TPAUSE'] = "0"
            # Provides fine granularity parallelism
            os.environ['KMP_FORKJOIN_BARRIER_PATTERN'] = "dist,dist"
            os.environ['KMP_PLAIN_BARRIER_PATTERN'] = "dist,dist"
            os.environ['KMP_REDUCTION_BARRIER_PATTERN'] = "dist,dist"

        # To hint IPEX uses shared memory based AllReduce
        os.environ["LOCAL_WORLD_SIZE"] = str(
            aphrodite_config.parallel_config.tensor_parallel_size)
        if sys.platform == "darwin" and \
                envs.APHRODITE_WORKER_MULTIPROC_METHOD == "fork":
            if os.environ.get('APHRODITE_WORKER_MULTIPROC_METHOD',
                              None) is None:
                logger.warning(
                    "Default to spawn method on MacOS. If this is not desired,"
                    " set APHRODITE_WORKER_MULTIPROC_METHOD to fork explicitly"
                    ".")
                os.environ['APHRODITE_WORKER_MULTIPROC_METHOD'] = 'spawn'

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on CPU.")
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "aphrodite.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get device specific communicator class for distributed communication.
        """
        return "aphrodite.distributed.device_communicators.cpu_communicator.CpuCommunicator"  # noqa
