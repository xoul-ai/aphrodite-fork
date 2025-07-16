import hashlib
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    APHRODITE_HOST_IP: str = ""
    APHRODITE_PORT: Optional[int] = None
    APHRODITE_RPC_BASE_PATH: str = tempfile.gettempdir()
    APHRODITE_USE_MODELSCOPE: bool = False
    APHRODITE_RINGBUFFER_WARNING_INTERVAL: int = 60
    APHRODITE_INSTANCE_ID: Optional[str] = None
    APHRODITE_NCCL_SO_PATH: Optional[str] = None
    LD_LIBRARY_PATH: Optional[str] = None
    APHRODITE_USE_TRITON_FLASH_ATTN: bool = False
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    APHRODITE_ENGINE_ITERATION_TIMEOUT_S: int = 60
    APHRODITE_API_KEY: Optional[str] = None
    APHRODITE_ADMIN_KEY: Optional[str] = None
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None
    APHRODITE_CACHE_ROOT: str = os.path.expanduser("~/.cache/aphrodite")
    APHRODITE_CONFIG_ROOT: str = os.path.expanduser("~/.config/aphrodite")
    APHRODITE_CONFIGURE_LOGGING: int = 1
    APHRODITE_LOGGING_LEVEL: str = "INFO"
    APHRODITE_LOGGING_CONFIG_PATH: Optional[str] = None
    APHRODITE_LOGITS_PROCESSOR_THREADS: Optional[int] = None
    APHRODITE_FLASH_ATTN_VERSION: Optional[int] = None
    APHRODITE_TRACE_FUNCTION: int = 0
    APHRODITE_ATTENTION_BACKEND: Optional[str] = None
    APHRODITE_USE_SAMPLING_KERNELS: bool = False
    APHRODITE_PP_LAYER_PARTITION: Optional[str] = None
    APHRODITE_CPU_KVCACHE_SPACE: int = 0
    APHRODITE_CPU_OMP_THREADS_BIND: str = ""
    APHRODITE_OPENVINO_DEVICE: str = "CPU"
    APHRODITE_OPENVINO_KVCACHE_SPACE: int = 0
    APHRODITE_OPENVINO_CPU_KV_CACHE_PRECISION: Optional[str] = None
    APHRODITE_OPENVINO_ENABLE_QUANTIZED_WEIGHTS: bool = False
    APHRODITE_XLA_CACHE_PATH: str = os.path.join(APHRODITE_CACHE_ROOT, "xla_cache")  # noqa: E501
    APHRODITE_FUSED_MOE_CHUNK_SIZE: int = 64 * 1024
    APHRODITE_USE_RAY_SPMD_WORKER: bool = False
    APHRODITE_USE_RAY_COMPILED_DAG: bool = False
    APHRODITE_USE_RAY_COMPILED_DAG_NCCL_CHANNEL: bool = True
    APHRODITE_WORKER_MULTIPROC_METHOD: str = "fork"
    APHRODITE_ASSETS_CACHE: str = os.path.join(APHRODITE_CACHE_ROOT, "assets")
    APHRODITE_IMAGE_FETCH_TIMEOUT: int = 5
    APHRODITE_AUDIO_FETCH_TIMEOUT: int = 5
    APHRODITE_MM_INPUT_CACHE_GIB: int = 8
    APHRODITE_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: Optional[str] = None
    NVCC_THREADS: Optional[str] = None
    APHRODITE_USE_PRECOMPILED: bool = False
    APHRODITE_NO_DEPRECATION_WARNING: bool = False
    APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
    CMAKE_BUILD_TYPE: Optional[str] = None
    VERBOSE: bool = False
    APHRODITE_DYNAMIC_ROPE_SCALING: bool = False
    APHRODITE_TEST_FORCE_FP8_MARLIN: bool = False
    APHRODITE_PLUGINS: Optional[list[str]] = None
    APHRODITE_TORCH_PROFILER_DIR: Optional[str] = None
    APHRODITE_RPC_TIMEOUT: int = 20000
    APHRODITE_FORCE_SINGLE_USER_PREFIX_CACHE: bool = False
    APHRODITE_TEST_DYNAMO_GRAPH_CAPTURE: int = 0
    APHRODITE_TEST_DYNAMO_FULLGRAPH_CAPTURE: int = 0
    APHRODITE_USE_TRITON_AWQ: bool = False
    APHRODITE_DYNAMO_USE_CUSTOM_DISPATCHER: bool = False
    APHRODITE_USE_TRITON_BACKEND: bool = False
    APHRODITE_FORCE_P2P: bool = False
    APHRODITE_TEST_ENABLE_ARTIFICIAL_PREEMPT: bool = False
    APHRODITE_REQUEST_LEVEL_METRICS: bool = False
    APHRODITE_TORCH_COMPILE_LEVEL: int = 0
    APHRODITE_CUSTOM_OPS: list[str] = []
    APHRODITE_DISABLED_KERNELS: list[str] = []
    APHRODITE_SKIP_P2P_CHECK: bool = False
    APHRODITE_FLASHINFER_FORCE_TENSOR_CORES: bool = False
    APHRODITE_USE_V1: bool = False
    APHRODITE_ROCM_USE_AITER: bool = False
    APHRODITE_ROCM_USE_AITER_PAGED_ATTN: bool = False
    APHRODITE_ROCM_USE_AITER_LINEAR: bool = True
    APHRODITE_ROCM_USE_AITER_MOE: bool = True
    APHRODITE_ROCM_USE_AITER_RMSNORM: bool = True
    APHRODITE_ROCM_USE_AITER_MLA: bool = True
    APHRODITE_ROCM_USE_SKINNY_GEMM: bool = True
    APHRODITE_ROCM_FP8_PADDING: bool = True
    APHRODITE_ROCM_MOE_PADDING: bool = True
    APHRODITE_ROCM_CUSTOM_PAGED_ATTN: bool = True
    APHRODITE_ENABLE_V1_MULTIPROCESSING: bool = True
    APHRODITE_LOG_BATCHSIZE_INTERVAL: float = -1
    APHRODITE_DISABLE_COMPILE_CACHE: bool = False
    Q_SCALE_CONSTANT: int = 200
    K_SCALE_CONSTANT: int = 200
    V_SCALE_CONSTANT: int = 100
    APHRODITE_SERVER_DEV_MODE: bool = False
    APHRODITE_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
    APHRODITE_MLA_DISABLE: bool = False
    APHRODITE_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON: bool = False
    APHRODITE_RAY_PER_WORKER_GPUS: float = 1.0
    APHRODITE_RAY_BUNDLE_INDICES: str = ""
    APHRODITE_CUDART_SO_PATH: Optional[str] = None
    APHRODITE_USE_HPU_CONTIGUOUS_CACHE_FETCH: bool = True
    APHRODITE_HPU_USE_DELAYED_SAMPLING: bool = False
    APHRODITE_DP_RANK: int = 0
    APHRODITE_DP_RANK_LOCAL: int = -1
    APHRODITE_DP_SIZE: int = 1
    APHRODITE_DP_MASTER_IP: str = ""
    APHRODITE_DP_MASTER_PORT: int = 0
    APHRODITE_MARLIN_USE_ATOMIC_ADD: bool = False
    APHRODITE_V0_USE_OUTLINES_CACHE: bool = False
    APHRODITE_TPU_BUCKET_PADDING_GAP: int = 0
    APHRODITE_USE_DEEP_GEMM: bool = False
    APHRODITE_XGRAMMAR_CACHE_MB: int = 0
    APHRODITE_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
    APHRODITE_USAGE_STATS_SERVER: str = ""
    APHRODITE_NO_USAGE_STATS: bool = True
    APHRODITE_DO_NOT_TRACK: bool = True
    APHRODITE_USAGE_SOURCE: str = ""
    APHRODITE_ALLOW_RUNTIME_LORA_UPDATING: bool = False


def get_default_cache_root():
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root():
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

environment_variables: dict[str, Callable[[], Any]] = {

    # ================== Installation Time Env Vars ==================

    # Target device of Aphrodite, supporting [cuda (by default),
    # rocm, neuron, cpu, openvino]
    "APHRODITE_TARGET_DEVICE":
    lambda: os.getenv("APHRODITE_TARGET_DEVICE", "cuda"),

    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),

    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS":
    lambda: os.getenv("NVCC_THREADS", None),

    # If set, Aphrodite will use precompiled binaries (*.so)
    "APHRODITE_USE_PRECOMPILED":
    lambda: bool(os.environ.get("APHRODITE_USE_PRECOMPILED")),

    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),

    # If set, Aphrodite will print verbose logs during installation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),

    # Root directory for APHRODITE configuration files
    # Defaults to `~/.config/aphrodite` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how aphrodite finds its configuration
    # files during runtime, but also affects how aphrodite installs its
    # configuration files during **installation**.
    "APHRODITE_CONFIG_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "APHRODITE_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "aphrodite"),
        )),

    # ================== Runtime Env Vars ==================

    # Root directory for APHRODITE cache files
    # Defaults to `~/.cache/aphrodite` unless `XDG_CACHE_HOME` is set
    "APHRODITE_CACHE_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "APHRODITE_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "aphrodite"),
        )),

    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    'APHRODITE_HOST_IP':
    lambda: os.getenv('APHRODITE_HOST_IP', "") or os.getenv("HOST_IP", ""),

    # used in distributed environment to manually set the communication port
    # Note: if APHRODITE_PORT is set, and some code asks for multiple ports, the
    # APHRODITE_PORT will be used as the first port, and the rest will be
    # generated by incrementing the APHRODITE_PORT value.
    # '0' is used to make mypy happy
    'APHRODITE_PORT':
    lambda: int(os.getenv('APHRODITE_PORT', '0'))
    if 'APHRODITE_PORT' in os.environ else None,

    # path used for ipc when the frontend api server is running in
    # multi-processing mode to communicate with the backend engine process.
    'APHRODITE_RPC_BASE_PATH':
    lambda: os.getenv('APHRODITE_RPC_BASE_PATH', tempfile.gettempdir()),

    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    "APHRODITE_USE_MODELSCOPE":
    lambda: os.environ.get(
        "APHRODITE_USE_MODELSCOPE", "False").lower() == "true",

    # Instance id represents an instance of the APHRODITE. All processes in the
    # same instance should have the same instance id.
    "APHRODITE_INSTANCE_ID":
    lambda: os.environ.get("APHRODITE_INSTANCE_ID", None),

    # Interval in seconds to log a warning message when the ring buffer is full
    "APHRODITE_RINGBUFFER_WARNING_INTERVAL":
    lambda: int(os.environ.get("APHRODITE_RINGBUFFER_WARNING_INTERVAL", "60")),

    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME":
    lambda: os.environ.get("CUDA_HOME", None),

    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "APHRODITE_NCCL_SO_PATH":
    lambda: os.environ.get("APHRODITE_NCCL_SO_PATH", None),

    # when `APHRODITE_NCCL_SO_PATH` is not set, aphrodite will try to find the
    # nccl library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),

    # flag to control if aphrodite should use triton flash attention
    "APHRODITE_USE_TRITON_FLASH_ATTN":
    lambda: (os.environ.get(
        "APHRODITE_USE_TRITON_FLASH_ATTN", "True").lower() in ("true", "1")),

    # Force aphrodite to use a specific flash-attention version (2 or 3), only
    # valid when using the flash-attention backend.
    "APHRODITE_FLASH_ATTN_VERSION":
    lambda: maybe_convert_int(os.environ.get("APHRODITE_FLASH_ATTN_VERSION",
                                             None)),

    # Internal flag to enable Dynamo fullgraph capture
    "APHRODITE_TEST_DYNAMO_FULLGRAPH_CAPTURE":
    lambda: bool(
        os.environ.get("APHRODITE_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") != "0"),
    "APHRODITE_TORCH_COMPILE_LEVEL":
    lambda: int(os.environ.get("APHRODITE_TORCH_COMPILE_LEVEL", "0")),

    # Fine-grained control over which custom ops to enable/disable.
    # Use 'all' to enable all, 'none' to disable all.
    # Also specify a list of custom op names to enable (prefixed with a '+'),
    # or disable (prefixed with a '-').
    # Examples:
    # - 'all,-op1' to enable all except op1
    # - 'none,+op1,+op2' to enable only op1 and op2
    # By default, all custom ops are enabled when running without Inductor
    # and disabled when running with Inductor (compile_level >= Inductor).
    "APHRODITE_CUSTOM_OPS":
    lambda: os.environ.get("APHRODITE_CUSTOM_OPS",
                           "").replace(" ", "").split(","),

    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK":
    lambda: int(os.environ.get("LOCAL_RANK", "0")),

    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES":
    lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),

    # timeout for each iteration in the engine
    "APHRODITE_ENGINE_ITERATION_TIMEOUT_S":
    lambda: int(os.environ.get("APHRODITE_ENGINE_ITERATION_TIMEOUT_S", "60")),

    # API key for APHRODITE API server
    "APHRODITE_API_KEY":
    lambda: os.environ.get("APHRODITE_API_KEY", None),

    # Admin API key for APHRODITE API server
    "APHRODITE_ADMIN_KEY":
    lambda: os.environ.get("APHRODITE_ADMIN_KEY", None),

    # S3 access information, used for tensorizer to load model from S3
    "S3_ACCESS_KEY_ID":
    lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
    "S3_SECRET_ACCESS_KEY":
    lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL":
    lambda: os.environ.get("S3_ENDPOINT_URL", None),

    # Logging configuration
    # If set to 0, aphrodite will not configure logging
    # If set to 1, aphrodite will configure logging using the default
    # configuration or the configuration file specified by
    # APHRODITE_LOGGING_CONFIG_PATH
    "APHRODITE_CONFIGURE_LOGGING":
    lambda: int(os.getenv("APHRODITE_CONFIGURE_LOGGING", "1")),
    "APHRODITE_LOGGING_CONFIG_PATH":
    lambda: os.getenv("APHRODITE_LOGGING_CONFIG_PATH"),

    # this is used for configuring the default logging level
    "APHRODITE_LOGGING_LEVEL":
    lambda: os.getenv("APHRODITE_LOGGING_LEVEL", "INFO"),

    # if set, aphrodite will call logits processors in a thread pool with this
    # many threads. This is useful when using custom logits processors that
    # either (a) launch additional CUDA kernels or (b) do significant
    # CPU-bound work while not holding the python GIL, or both.
    "APHRODITE_LOGITS_PROCESSOR_THREADS":
    lambda: int(os.getenv("APHRODITE_LOGITS_PROCESSOR_THREADS", "0"))
    if "APHRODITE_LOGITS_PROCESSOR_THREADS" in os.environ else None,

    # Trace function calls
    # If set to 1, aphrodite will trace function calls
    # Useful for debugging
    "APHRODITE_TRACE_FUNCTION":
    lambda: int(os.getenv("APHRODITE_TRACE_FUNCTION", "0")),

    # Backend for attention computation
    # Available options:
    # - "TORCH_SDPA": use torch.nn.MultiheadAttention
    # - "FLASH_ATTN": use FlashAttention
    # - "XFORMERS": use XFormers
    # - "TRITON_FLASH": use TritonFlashAttention
    # - "FLASHINFER": use flashinfer
    "APHRODITE_ATTENTION_BACKEND":
    lambda: os.getenv("APHRODITE_ATTENTION_BACKEND", None),

    # If set, aphrodite will use custom sampling kernels
    "APHRODITE_USE_SAMPLING_KERNELS":
    lambda: bool(int(os.getenv("APHRODITE_USE_SAMPLING_KERNELS", "0"))),

    # Pipeline stage partition strategy
    "APHRODITE_PP_LAYER_PARTITION":
    lambda: os.getenv("APHRODITE_PP_LAYER_PARTITION", None),

    # (CPU backend only) CPU key-value cache space.
    # default is 4GB
    "APHRODITE_CPU_KVCACHE_SPACE":
    lambda: int(os.getenv("APHRODITE_CPU_KVCACHE_SPACE", "0")),

    # (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
    # "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
    "APHRODITE_CPU_OMP_THREADS_BIND":
    lambda: os.getenv("APHRODITE_CPU_OMP_THREADS_BIND", "all"),

    # OpenVINO device selection
    # default is CPU
    "APHRODITE_OPENVINO_DEVICE":
    lambda: os.getenv("APHRODITE_OPENVINO_DEVICE", "CPU").upper(),

    # OpenVINO key-value cache space
    # default is 4GB
    "APHRODITE_OPENVINO_KVCACHE_SPACE":
    lambda: int(os.getenv("APHRODITE_OPENVINO_KVCACHE_SPACE", "0")),

    # OpenVINO KV cache precision
    # default is bf16 if natively supported by platform, otherwise f16
    # To enable KV cache compression, please, explicitly specify u8
    "APHRODITE_OPENVINO_CPU_KV_CACHE_PRECISION":
    lambda: os.getenv("APHRODITE_OPENVINO_CPU_KV_CACHE_PRECISION", None),

    # Enables weights compression during model export via HF Optimum
    # default is False
    "APHRODITE_OPENVINO_ENABLE_QUANTIZED_WEIGHTS":
    lambda: bool(os.getenv(
        "APHRODITE_OPENVINO_ENABLE_QUANTIZED_WEIGHTS", False)),

    # If the env var is set, then all workers will execute as separate
    # processes from the engine, and we use the same mechanism to trigger
    # execution on all workers.
    # Run aphrodite with APHRODITE_USE_RAY_SPMD_WORKER=1 to enable it.
    "APHRODITE_USE_RAY_SPMD_WORKER":
    lambda: bool(int(os.getenv("APHRODITE_USE_RAY_SPMD_WORKER", "0"))),

    # If the env var is set, it uses the Ray's compiled DAG API
    # which optimizes the control plane overhead.
    # Run aphrodite with APHRODITE_USE_RAY_COMPILED_DAG=1 to enable it.
    "APHRODITE_USE_RAY_COMPILED_DAG":
    lambda: bool(int(os.getenv("APHRODITE_USE_RAY_COMPILED_DAG", "0"))),

    # If the env var is set, it uses NCCL for communication in
    # Ray's compiled DAG. This flag is ignored if
    # APHRODITE_USE_RAY_COMPILED_DAG is not set.
    "APHRODITE_USE_RAY_COMPILED_DAG_NCCL_CHANNEL":
    lambda: bool(int(
        os.getenv("APHRODITE_USE_RAY_COMPILED_DAG_NCCL_CHANNEL", "1"))),

    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "APHRODITE_WORKER_MULTIPROC_METHOD":
    lambda: os.getenv("APHRODITE_WORKER_MULTIPROC_METHOD", "fork"),

    # Path to the cache for storing downloaded assets
    "APHRODITE_ASSETS_CACHE":
    lambda: os.path.expanduser(
        os.getenv(
            "APHRODITE_ASSETS_CACHE",
            os.path.join(get_default_cache_root(), "aphrodite", "assets"),
        )),

    # Timeout for fetching images when serving multimodal models
    # Default is 5 seconds
    "APHRODITE_IMAGE_FETCH_TIMEOUT":
    lambda: int(os.getenv("APHRODITE_IMAGE_FETCH_TIMEOUT", "5")),

    # Timeout for fetching audio when serving multimodal models
    # Default is 5 seconds
    "APHRODITE_AUDIO_FETCH_TIMEOUT":
    lambda: int(os.getenv("APHRODITE_AUDIO_FETCH_TIMEOUT", "5")),


    # Cache size (in GiB) for multimodal input cache
    # Default is 4 GiB
    "APHRODITE_MM_INPUT_CACHE_GIB":
    lambda: int(os.getenv("APHRODITE_MM_INPUT_CACHE_GIB", "4")),

    # Path to the XLA persistent cache directory.
    # Only used for XLA devices such as TPUs.
    "APHRODITE_XLA_CACHE_PATH":
    lambda: os.path.expanduser(
        os.getenv(
            "APHRODITE_XLA_CACHE_PATH",
            os.path.join(get_default_cache_root(), "aphrodite", "xla_cache"),
        )),
    "APHRODITE_FUSED_MOE_CHUNK_SIZE":
    lambda: int(os.getenv("APHRODITE_FUSED_MOE_CHUNK_SIZE", "65536")),

    # If set, aphrodite will skip the deprecation warnings.
    "APHRODITE_NO_DEPRECATION_WARNING":
    lambda: bool(int(os.getenv("APHRODITE_NO_DEPRECATION_WARNING", "0"))),

    # If set, the OpenAI API server will stay alive even after the underlying
    # AsyncLLMEngine errors and stops serving requests
    "APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH":
    lambda: bool(os.getenv("APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH", 0)),

    # If the env var APHRODITE_DYNAMIC_ROPE_SCALING is set, it allows
    # the user to specify a max sequence length greater than
    # the max length derived from the model's config.json.
    # To enable this, set APHRODITE_DYNAMIC_ROPE_SCALING=1.
    "APHRODITE_DYNAMIC_ROPE_SCALING":
    lambda:
    (os.environ.get(
        "APHRODITE_DYNAMIC_ROPE_SCALING",
        "0").strip().lower() in ("1", "true")),

    # If set, forces FP8 Marlin to be used for FP8 quantization regardless
    # of the hardware support for FP8 compute.
    "APHRODITE_TEST_FORCE_FP8_MARLIN":
    lambda:
    (os.environ.get("APHRODITE_TEST_FORCE_FP8_MARLIN", "0").strip().lower() in
     ("1", "true")),

    # Time in ms for the zmq client to wait for a response from the backend
    # server for simple data operations
    "APHRODITE_RPC_TIMEOUT":
    lambda: int(os.getenv("APHRODITE_RPC_TIMEOUT", "20000")),

    # a list of plugin names to load, separated by commas.
    # if this is not set, it means all plugins will be loaded
    # if this is set to an empty string, no plugins will be loaded
    "APHRODITE_PLUGINS":
    lambda: None if "APHRODITE_PLUGINS" not in os.environ else os.environ[
        "APHRODITE_PLUGINS"].split(","),

    # Enables torch profiler if set. Path to the directory where torch profiler
    # traces are saved. Note that it must be an absolute path.
    "APHRODITE_TORCH_PROFILER_DIR":
    lambda: (None if os.getenv("APHRODITE_TORCH_PROFILER_DIR",
             None) is None else
             os.path.expanduser(
             os.getenv("APHRODITE_TORCH_PROFILER_DIR", "."))),

    # If set, forces prefix cache in single user mode
    "APHRODITE_FORCE_SINGLE_USER_PREFIX_CACHE":
    lambda: bool(int(os.getenv("APHRODITE_FORCE_SINGLE_USER_PREFIX_CACHE",
                               "0"))),

    # If set, Aphrodite will use Triton implementations of AWQ.
    "APHRODITE_USE_TRITON_AWQ":
    lambda: bool(int(os.getenv("APHRODITE_USE_TRITON_AWQ", "0"))),

    # If set, Aphrodite will use Triton implementations of layernorm.
    "APHRODITE_USE_TRITON_BACKEND":
    lambda: bool(int(os.getenv("APHRODITE_USE_TRITON_BACKEND", "0"))),

    # If set, Aphrodite will skip the P2P check and assume that P2P is
    # available. Used for custom all-reduce kernels.
    "APHRODITE_FORCE_P2P":
    lambda: bool(int(os.getenv("APHRODITE_FORCE_P2P", "0"))),

    # If set, Aphrodite will use artificial preemption.
    "APHRODITE_TEST_ENABLE_ARTIFICIAL_PREEMPT":
    lambda: bool(int(
        os.getenv("APHRODITE_TEST_ENABLE_ARTIFICIAL_PREEMPT", "0"))),

    # If set, Aphrodite will use request-level metrics instead of
    # interval-based metrics.
    "APHRODITE_REQUEST_LEVEL_METRICS":
    lambda: bool(int(os.getenv("APHRODITE_REQUEST_LEVEL_METRICS", "0"))),

    # list of quantization kernels that should be disabled, used for testing
    # and performance comparisons. Currently only affects MPLinearKernel
    # selection
    # (kernels: MacheteLinearKernel, MarlinLinearKernel, ExllamaLinearKernel)
    "APHRODITE_DISABLED_KERNELS":
    lambda: [
    ] if "APHRODITE_DISABLED_KERNELS" not in os.environ else os.environ[
        "APHRODITE_DISABLED_KERNELS"].split(","),

    # By default, Aphrodite will check the peer-to-peer capability itself,
    # in case of broken drivers.
    # If this env var is set to 1, Aphrodite will skip the peer-to-peer check,
    # and trust the driver's peer-to-peer capability report.
    "APHRODITE_SKIP_P2P_CHECK":
    lambda: os.getenv("APHRODITE_SKIP_P2P_CHECK", "0") == "1",

    # If set, Aphrodite will force flashinfer to use tensor cores;
    # otherwise will use heuristic based on model architecture.
    "APHRODITE_FLASHINFER_FORCE_TENSOR_CORES":
    lambda: bool(int(os.getenv("APHRODITE_FLASHINFER_FORCE_TENSOR_CORES",
                               "0"))),

    # If set, use the V1 code path.
    "APHRODITE_USE_V1":
    lambda: bool(int(os.getenv("APHRODITE_USE_V1", "0"))),

    # Disable aiter ops unless specifically enabled.
    # Acts as a parent switch to enable the rest of the other operations.
    "APHRODITE_ROCM_USE_AITER":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER", "False").lower() in
             ("true", "1")),

    # Whether to use aiter paged attention.
    # By default is disabled.
    "APHRODITE_ROCM_USE_AITER_PAGED_ATTN":
    lambda: (os.getenv(
             "APHRODITE_ROCM_USE_AITER_PAGED_ATTN", "False").lower() in
             ("true", "1")),

    # use aiter linear op if aiter ops are enabled
    # The following list of related ops
    # - scaled_mm (per-tensor / rowwise)
    "APHRODITE_ROCM_USE_AITER_LINEAR":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER_LINEAR", "True").lower() in
             ("true", "1")),

    # Whether to use aiter moe ops.
    # By default is enabled.
    "APHRODITE_ROCM_USE_AITER_MOE":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER_MOE", "True").lower() in
             ("true", "1")),

    # use aiter rms norm op if aiter ops are enabled.
    "APHRODITE_ROCM_USE_AITER_RMSNORM":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER_RMSNORM", "True").lower() in
             ("true", "1")),

    # Whether to use aiter mla ops.
    # By default is enabled.
    "APHRODITE_ROCM_USE_AITER_MLA":
    lambda: (os.getenv("APHRODITE_ROCM_USE_AITER_MLA", "True").lower() in
             ("true", "1")),
    # use rocm skinny gemms
    "APHRODITE_ROCM_USE_SKINNY_GEMM":
    lambda: (os.getenv("APHRODITE_ROCM_USE_SKINNY_GEMM", "True").lower() in
             ("true", "1")),

    # Pad the fp8 weights to 256 bytes for ROCm
    "APHRODITE_ROCM_FP8_PADDING":
    lambda: bool(int(os.getenv("APHRODITE_ROCM_FP8_PADDING", "1"))),

    # Pad the weights for the moe kernel
    "APHRODITE_ROCM_MOE_PADDING":
    lambda: bool(int(os.getenv("APHRODITE_ROCM_MOE_PADDING", "1"))),

    # custom paged attention kernel for MI3* cards
    "APHRODITE_ROCM_CUSTOM_PAGED_ATTN":
    lambda: (os.getenv("APHRODITE_ROCM_CUSTOM_PAGED_ATTN", "True").lower() in
             ("true", "1")),

    # Divisor for dynamic query scale factor calculation for FP8 KV Cache
    "Q_SCALE_CONSTANT":
    lambda: int(os.getenv("Q_SCALE_CONSTANT", "200")),
    # Divisor for dynamic key scale factor calculation for FP8 KV Cache
    "K_SCALE_CONSTANT":
    lambda: int(os.getenv("K_SCALE_CONSTANT", "200")),
    # Divisor for dynamic value scale factor calculation for FP8 KV Cache
    "V_SCALE_CONSTANT":
    lambda: int(os.getenv("V_SCALE_CONSTANT", "100")),

    # If set, enable multiprocessing in LLM for the V1 code path.
    "APHRODITE_ENABLE_V1_MULTIPROCESSING":
    lambda: bool(int(os.getenv("APHRODITE_ENABLE_V1_MULTIPROCESSING", "1"))),
    "APHRODITE_LOG_BATCHSIZE_INTERVAL":
    lambda: float(os.getenv("APHRODITE_LOG_BATCHSIZE_INTERVAL", "-1")),
    "APHRODITE_DISABLE_COMPILE_CACHE":
    lambda: bool(int(os.getenv("APHRODITE_DISABLE_COMPILE_CACHE", "0"))),

    # If set, Aphrodite will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "APHRODITE_SERVER_DEV_MODE":
    lambda: bool(int(os.getenv("APHRODITE_SERVER_DEV_MODE", "0"))),

    # Controls the maximum number of requests to handle in a
    # single asyncio task when processing per-token outputs in the
    # V1 AsyncLLM interface. It is applicable when handling a high
    # concurrency of streaming requests.
    # Setting this too high can result in a higher variance of
    # inter-message latencies. Setting it too low can negatively impact
    # TTFT and overall throughput.
    "APHRODITE_V1_OUTPUT_PROC_CHUNK_SIZE":
    lambda: int(os.getenv("APHRODITE_V1_OUTPUT_PROC_CHUNK_SIZE", "128")),

    # If set, Aphrodite will disable the MLA attention optimizations.
    "APHRODITE_MLA_DISABLE":
    lambda: bool(int(os.getenv("APHRODITE_MLA_DISABLE", "0"))),

    # If set, Aphrodite will use the Triton implementation of
    # moe_align_block_size, i.e. moe_align_block_size_triton in fused_moe.py.
    "APHRODITE_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON":
    lambda: bool(int(os.getenv("APHRODITE_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON",
                               "0"))),

    # Number of GPUs per worker in Ray, if it is set to be a fraction,
    # it allows ray to schedule multiple actors on a single GPU,
    # so that users can colocate other actors on the same GPUs as Aphrodite.
    "APHRODITE_RAY_PER_WORKER_GPUS":
    lambda: float(os.getenv("APHRODITE_RAY_PER_WORKER_GPUS", "1.0")),

    # Bundle indices for Ray, if it is set, it can control precisely
    # which indices are used for the Ray bundle, for every worker.
    # Format: comma-separated list of integers, e.g. "0,1,2,3"
    "APHRODITE_RAY_BUNDLE_INDICES":
    lambda: os.getenv("APHRODITE_RAY_BUNDLE_INDICES", ""),

    # In some system, find_loaded_library() may not work. So we allow users to
    # specify the path through environment variable APHRODITE_CUDART_SO_PATH.
    "APHRODITE_CUDART_SO_PATH":
    lambda: os.getenv("APHRODITE_CUDART_SO_PATH", None),

    # Contiguous cache fetching to avoid using costly gather operation on
    # Gaudi3. This is only applicable to HPU contiguous cache. If set to true,
    # contiguous cache fetch will be used.
    "APHRODITE_USE_HPU_CONTIGUOUS_CACHE_FETCH":
    lambda: os.environ.get("APHRODITE_CONTIGUOUS_PA", "true").lower() in
    ("1", "true"),

    # Use delayed sampling for HPU to reduce host cpu overhead
    # between each step.
    "APHRODITE_HPU_USE_DELAYED_SAMPLING":
    lambda: os.environ.get("APHRODITE_DELAYED_SAMPLING", "false").lower() in
    ("1", "true"),

    # Rank of the process in the data parallel setting
    "APHRODITE_DP_RANK":
    lambda: int(os.getenv("APHRODITE_DP_RANK", "0")),

    # Rank of the process in the data parallel setting.
    # Defaults to APHRODITE_DP_RANK when not set.
    "APHRODITE_DP_RANK_LOCAL":
    lambda: int(
        os.getenv("APHRODITE_DP_RANK_LOCAL",
        sys.modules[__name__].APHRODITE_DP_RANK)),

    # World size of the data parallel setting
    "APHRODITE_DP_SIZE":
    lambda: int(os.getenv("APHRODITE_DP_SIZE", "1")),

    # IP address of the master node in the data parallel setting
    "APHRODITE_DP_MASTER_IP":
    lambda: os.getenv("APHRODITE_DP_MASTER_IP", "127.0.0.1"),

    # Port of the master node in the data parallel setting
    "APHRODITE_DP_MASTER_PORT":
    lambda: int(os.getenv("APHRODITE_DP_MASTER_PORT", "0")),

    # Whether to use S3 path for model loading in CI via RunAI Streamer
    "APHRODITE_CI_USE_S3":
    lambda: os.environ.get("APHRODITE_CI_USE_S3", "0") == "1",

    # Use model_redirect to redirect the model name to a local folder.
    # `model_redirect` can be a json file mapping the model between
    # repo_id and local folder:
    # {"meta-llama/Llama-3.2-1B": "/tmp/Llama-3.2-1B"}
    # or a space separated values table file:
    # meta-llama/Llama-3.2-1B   /tmp/Llama-3.2-1B
    "APHRODITE_MODEL_REDIRECT_PATH":
    lambda: os.environ.get("APHRODITE_MODEL_REDIRECT_PATH", None),

    # Whether to use atomicAdd reduce in gptq/awq marlin kernel.
    "APHRODITE_MARLIN_USE_ATOMIC_ADD":
    lambda: os.environ.get("APHRODITE_MARLIN_USE_ATOMIC_ADD", "0") == "1",

    # Whether to turn on the outlines cache for V0
    # This cache is unbounded and on disk, so it's not safe to use in
    # an environment with potentially malicious users.
    "APHRODITE_V0_USE_OUTLINES_CACHE":
    lambda: os.environ.get("APHRODITE_V0_USE_OUTLINES_CACHE", "0") == "1",

    # Gap between padding buckets for the forward pass. So we have
    # 8, we will run forward pass with [16, 24, 32, ...].
    "APHRODITE_TPU_BUCKET_PADDING_GAP":
    lambda: int(os.environ["APHRODITE_TPU_BUCKET_PADDING_GAP"])
    if "APHRODITE_TPU_BUCKET_PADDING_GAP" in os.environ else 0,

    # Allow use of DeepGemm kernels for fused moe ops.
    "APHRODITE_USE_DEEP_GEMM":
    lambda: bool(int(os.getenv("APHRODITE_USE_DEEP_GEMM", "0"))),

    # Control the cache sized used by the xgrammar compiler. The default
    # of 512 MB should be enough for roughly 1000 JSON schemas.
    # It can be changed with this variable if needed for some reason.
    "APHRODITE_XGRAMMAR_CACHE_MB":
    lambda: int(os.getenv("APHRODITE_XGRAMMAR_CACHE_MB", "512")),

    # Control the threshold for msgspec to use 'zero copy' for
    # serialization/deserialization of tensors. Tensors below
    # this limit will be encoded into the msgpack buffer, and
    # tensors above will instead be sent via a separate message.
    # While the sending side still actually copies the tensor
    # in all cases, on the receiving side, tensors above this
    # limit will actually be zero-copy decoded.
    "APHRODITE_MSGPACK_ZERO_COPY_THRESHOLD":
    lambda: int(os.getenv("APHRODITE_MSGPACK_ZERO_COPY_THRESHOLD", "256")),

    # Usage stats collection
    "APHRODITE_USAGE_STATS_SERVER":
    lambda: os.environ.get("APHRODITE_USAGE_STATS_SERVER", ""),
    "APHRODITE_NO_USAGE_STATS": 
    lambda: os.environ.get("APHRODITE_NO_USAGE_STATS", "1") == "1",
    "APHRODITE_DO_NOT_TRACK":
    lambda: (os.environ.get("APHRODITE_DO_NOT_TRACK", "1") or os.environ.get(
        "DO_NOT_TRACK", "1") or "1") == "1",
    "APHRODITE_USAGE_SOURCE":
    lambda: os.environ.get("APHRODITE_USAGE_SOURCE", ""),

    # If set, allow loading or unloading lora adapters in runtime,
    "APHRODITE_ALLOW_RUNTIME_LORA_UPDATING":
    lambda:
    (os.environ.get("APHRODITE_ALLOW_RUNTIME_LORA_UPDATING",
                    "0").strip().lower() in
     ("1", "true")),

    # Whether to log responses from API Server for debugging
    "APHRODITE_DEBUG_LOG_API_SERVER_RESPONSE":
    lambda: os.environ.get("APHRODITE_DEBUG_LOG_API_SERVER_RESPONSE", "False").
    lower() == "true",

    # If set, enables the KoboldAI API routes in the API server
    "APHRODITE_KOBOLD_API":
    lambda: bool(int(os.getenv("APHRODITE_KOBOLD_API", "0"))),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())

def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def set_aphrodite_use_v1(use_v1: bool):
    if is_set("APHRODITE_USE_V1"):
        raise ValueError(
            "Should not call set_aphrodite_use_v1() if APHRODITE_USE_V1 is set "
            "explicitly by the user. Please raise this as a Github "
            "Issue and explicitly set APHRODITE_USE_V1=0 or 1.")
    os.environ["APHRODITE_USE_V1"] = "1" if use_v1 else "0"


def compute_hash() -> str:
    """
    WARNING: Whenever a new key is added to this environment
    variables, ensure that it is included in the factors list if
    it affects the computation graph. For example, different values
    of APHRODITE_PP_LAYER_PARTITION will generate different computation
    graphs, so it is included in the factors list. The env vars that
    affect the choice of different kernels or attention backends should
    also be included in the factors list.
    """
    factors: list[Any] = []

    # summarize environment variables
    def factorize(name: str):
        if __getattr__(name):
            factors.append(__getattr__(name))
        else:
            factors.append("None")

    # The values of envs may affects the computation graph.
    # TODO(DefTruth): hash all environment variables?
    # for key in environment_variables:
    #     factorize(key)
    environment_variables_to_hash = [
        "APHRODITE_PP_LAYER_PARTITION",
        "APHRODITE_MLA_DISABLE",
        "APHRODITE_USE_TRITON_FLASH_ATTN",
        "APHRODITE_USE_TRITON_AWQ",
        "APHRODITE_DP_RANK",
        "APHRODITE_DP_SIZE",
    ]
    for key in environment_variables_to_hash:
        if key in environment_variables:
            factorize(key)

    hash_str = hashlib.md5(str(factors).encode(),
                           usedforsecurity=False).hexdigest()

    return hash_str
