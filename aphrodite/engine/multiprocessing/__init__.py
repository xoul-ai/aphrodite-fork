import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Mapping, Optional, Union, overload

from typing_extensions import deprecated

from aphrodite import PoolingParams
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.utils import Device, deprecate_kwargs
from aphrodite.inputs import PromptType
from aphrodite.lora.request import LoRARequest
from aphrodite.prompt_adapter.request import PromptAdapterRequest

APHRODITE_RPC_SUCCESS_STR = "SUCCESS"

IPC_INPUT_EXT = "_input_socket"
IPC_OUTPUT_EXT = "_output_socket"
IPC_HEALTH_EXT = "_health_socket"
IPC_DATA_EXT = "_data_socket"


class MQEngineDeadError(RuntimeError):
    pass


@dataclass
class RPCProcessRequest:
    prompt: PromptType
    params: Union[SamplingParams, PoolingParams]
    request_id: str
    lora_request: Optional[LoRARequest] = None
    trace_headers: Optional[Mapping[str, str]] = None
    prompt_adapter_request: Optional[PromptAdapterRequest] = None
    priority: int = 0

    @overload
    def __init__(
        self,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        ...

    @overload
    @deprecated("'inputs' will be renamed to 'prompt")
    def __init__(
        self,
        *,
        inputs: PromptType,
        params: Union[SamplingParams, PoolingParams],
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        ...

    @deprecate_kwargs(
        "inputs",
        additional_message="Please use the 'prompt' parameter instead.",
    )
    def __init__(
            self,
            prompt: Optional[PromptType] = None,
            params: Optional[Union[SamplingParams, PoolingParams]] = None,
            request_id: Optional[str] = None,
            lora_request: Optional[LoRARequest] = None,
            trace_headers: Optional[Mapping[str, str]] = None,
            prompt_adapter_request: Optional[PromptAdapterRequest] = None,
            priority: int = 0,
            *,
            inputs: Optional[PromptType] = None,  # DEPRECATED
    ) -> None:
        if inputs is not None:
            prompt = inputs
        assert (prompt is not None and params is not None
                and request_id is not None)

        super().__init__()

        self.prompt = prompt
        self.params = params
        self.request_id = request_id
        self.lora_request = lora_request
        self.trace_headers = trace_headers
        self.prompt_adapter_request = prompt_adapter_request
        self.priority = priority


@dataclass
class RPCError:
    request_id: Optional[str]
    is_engine_errored: bool
    exception: BaseException


@dataclass
class RPCAbortRequest:
    request_id: str


class RPCStartupRequest(Enum):
    IS_SERVER_READY = 1


@dataclass
class RPCStartupResponse:
    tracing_enabled: bool


class RPCUProfileRequest(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


@dataclass
class RPCResetPrefixCacheRequest:
    device: Device


class RPCSleepRequest(Enum):
    SLEEP_LEVEL_1 = 1
    SLEEP_LEVEL_2 = 2


@dataclass
class RPCWakeUpRequest:
    tags: Optional[list[str]] = None


@dataclass
class RPCIsSleepingRequest:
    # Set the default value of request_id to a new UUID
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RPCIsSleepingResponse:
    request_id: str
    is_sleeping: bool

@dataclass
class RPCShutdownRequest:
    pass


@dataclass
class RPCLoadAdapterRequest:
    lora_request: LoRARequest
    # Set the default value of request_id to a new UUID
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RPCAdapterLoadedResponse:
    request_id: str


RPC_REQUEST_T = Union[RPCProcessRequest, RPCAbortRequest, RPCStartupRequest,
                      RPCUProfileRequest, RPCLoadAdapterRequest,
                      RPCResetPrefixCacheRequest, RPCSleepRequest,
                      RPCWakeUpRequest, RPCIsSleepingRequest, RPCShutdownRequest]

REQUEST_OUTPUTS_T = Union[List[RequestOutput], RPCAdapterLoadedResponse,
                          RPCIsSleepingResponse, RPCError]


def ENGINE_DEAD_ERROR(
        error: Optional[BaseException] = None) -> MQEngineDeadError:
    if error is None:
        return MQEngineDeadError(
            "Engine loop is not running. Inspect the stacktrace to "
            "find the original error")

    return MQEngineDeadError(
        "Engine loop is not running. Inspect the stacktrace to "
        f"find the original error: {repr(error)}.")
