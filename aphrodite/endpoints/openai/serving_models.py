import json
import pathlib
from asyncio import Lock
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional, Union

from loguru import logger

from aphrodite.common.config import ModelConfig
from aphrodite.common.utils import AtomicCounter
from aphrodite.endpoints.openai.protocol import (ErrorResponse,
                                                 LoadLoRAAdapterRequest,
                                                 ModelCard, ModelList,
                                                 ModelPermission,
                                                 UnloadLoRAAdapterRequest)
from aphrodite.engine.protocol import EngineClient
from aphrodite.lora.request import LoRARequest
from aphrodite.lora.resolver import LoRAResolver, LoRAResolverRegistry
from aphrodite.prompt_adapter.request import PromptAdapterRequest


@dataclass
class BaseModelPath:
    name: str
    model_path: str


@dataclass
class PromptAdapterPath:
    name: str
    local_path: str


@dataclass
class LoRAModulePath:
    name: str
    path: str
    base_model_name: Optional[str] = None


class OpenAIServingModels:
    """Shared instance to hold data about the loaded base model(s) and adapters.

    Handles the routes:
    - /v1/models
    - /v1/load_lora_adapter
    - /v1/unload_lora_adapter
    """

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        base_model_paths: list[BaseModelPath],
        *,
        lora_modules: Optional[list[LoRAModulePath]] = None,
        prompt_adapters: Optional[list[PromptAdapterPath]] = None,
    ):
        super().__init__()

        self.base_model_paths = base_model_paths
        self.max_model_len = model_config.max_model_len
        self.engine_client = engine_client
        self.model_config = model_config

        self.static_lora_modules = lora_modules
        self.lora_requests: list[LoRARequest] = []
        self.lora_id_counter = AtomicCounter(0)

        self.lora_resolvers: list[LoRAResolver] = []
        for lora_resolver_name in LoRAResolverRegistry.get_supported_resolvers(
        ):
            self.lora_resolvers.append(
                LoRAResolverRegistry.get_resolver(lora_resolver_name))
        self.lora_resolver_lock: dict[str, Lock] = defaultdict(Lock)

        self.prompt_adapter_requests = []
        if prompt_adapters is not None:
            for i, prompt_adapter in enumerate(prompt_adapters, start=1):
                with pathlib.Path(prompt_adapter.local_path,
                                  "adapter_config.json").open() as f:
                    adapter_config = json.load(f)
                    num_virtual_tokens = adapter_config["num_virtual_tokens"]
                self.prompt_adapter_requests.append(
                    PromptAdapterRequest(
                        prompt_adapter_name=prompt_adapter.name,
                        prompt_adapter_id=i,
                        prompt_adapter_local_path=prompt_adapter.local_path,
                        prompt_adapter_num_virtual_tokens=num_virtual_tokens))

    async def init_static_loras(self):
        """Loads all static LoRA modules.
        Raises if any fail to load"""
        if self.static_lora_modules is None:
            return
        for lora in self.static_lora_modules:
            load_request = LoadLoRAAdapterRequest(lora_path=lora.path,
                                                  lora_name=lora.name)
            load_result = await self.load_lora_adapter(
                request=load_request, base_model_name=lora.base_model_name)
            if isinstance(load_result, ErrorResponse):
                raise ValueError(load_result.message)

    def is_base_model(self, model_name) -> bool:
        return any(model.name == model_name for model in self.base_model_paths)

    def model_name(self, lora_request: Optional[LoRARequest] = None) -> str:
        """Returns the appropriate model name depending on the availability
        and support of the LoRA or base model.
        Parameters:
        - lora: LoRARequest that contain a base_model_name.
        Returns:
        - str: The name of the base model or the first available model path.
        """
        if lora_request is not None:
            return lora_request.lora_name
        return self.base_model_paths[0].name

    async def show_available_models(self) -> ModelList:
        """Show available models. This includes the base model and all 
        adapters"""
        model_cards = [
            ModelCard(id=base_model.name,
                      max_model_len=self.max_model_len,
                      root=base_model.model_path,
                      permission=[ModelPermission()])
            for base_model in self.base_model_paths
        ]
        lora_cards = [
            ModelCard(id=lora.lora_name,
                      root=lora.local_path,
                      parent=lora.base_model_name if lora.base_model_name else
                      self.base_model_paths[0].name,
                      permission=[ModelPermission()])
            for lora in self.lora_requests
        ]
        prompt_adapter_cards = [
            ModelCard(id=prompt_adapter.prompt_adapter_name,
                      root=self.base_model_paths[0].name,
                      permission=[ModelPermission()])
            for prompt_adapter in self.prompt_adapter_requests
        ]
        model_cards.extend(lora_cards)
        model_cards.extend(prompt_adapter_cards)
        return ModelList(data=model_cards)

    async def load_lora_adapter(
            self,
            request: LoadLoRAAdapterRequest,
            base_model_name: Optional[str] = None
    ) -> Union[ErrorResponse, str]:
        error_check_ret = await self._check_load_lora_adapter_request(request)
        if error_check_ret is not None:
            return error_check_ret

        lora_name, lora_path = request.lora_name, request.lora_path
        unique_id = self.lora_id_counter.inc(1)
        lora_request = LoRARequest(lora_name=lora_name,
                                   lora_int_id=unique_id,
                                   lora_path=lora_path)
        if base_model_name is not None and self.is_base_model(base_model_name):
            lora_request.base_model_name = base_model_name

        # Validate that the adapter can be loaded into the engine
        # This will also pre-load it for incoming requests
        try:
            await self.engine_client.add_lora(lora_request)
        except BaseException as e:
            error_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            if "No adapter found" in str(e):
                error_type = "NotFoundError"
                status_code = HTTPStatus.NOT_FOUND

            return create_error_response(message=str(e),
                                         err_type=error_type,
                                         status_code=status_code)

        self.lora_requests.append(lora_request)
        logger.info("Loaded new LoRA adapter: name '{}', path '{}'", lora_name,
                    lora_path)
        return f"Success: LoRA adapter '{lora_name}' added successfully."

    async def unload_lora_adapter(
            self,
            request: UnloadLoRAAdapterRequest) -> Union[ErrorResponse, str]:
        error_check_ret = await self._check_unload_lora_adapter_request(request
                                                                        )
        if error_check_ret is not None:
            return error_check_ret

        lora_name = request.lora_name
        self.lora_requests = [
            lora_request for lora_request in self.lora_requests
            if lora_request.lora_name != lora_name
        ]
        logger.info("Removed LoRA adapter: name '{}'", lora_name)
        return f"Success: LoRA adapter '{lora_name}' removed successfully."

    async def _check_load_lora_adapter_request(
            self, request: LoadLoRAAdapterRequest) -> Optional[ErrorResponse]:
        # Check if both 'lora_name' and 'lora_path' are provided
        if not request.lora_name or not request.lora_path:
            return create_error_response(
                message="Both 'lora_name' and 'lora_path' must be provided.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST)

        # Check if the lora adapter with the given name already exists
        if any(lora_request.lora_name == request.lora_name
               for lora_request in self.lora_requests):
            return create_error_response(
                message=
                f"The lora adapter '{request.lora_name}' has already been "
                "loaded.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST)

        return None

    async def _check_unload_lora_adapter_request(
            self,
            request: UnloadLoRAAdapterRequest) -> Optional[ErrorResponse]:
        # Check if either 'lora_name' or 'lora_int_id' is provided
        if not request.lora_name and not request.lora_int_id:
            return create_error_response(
                message=
                "either 'lora_name' and 'lora_int_id' needs to be provided.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST)

        # Check if the lora adapter with the given name exists
        if not any(lora_request.lora_name == request.lora_name
                   for lora_request in self.lora_requests):
            return create_error_response(
                message=
                f"The lora adapter '{request.lora_name}' cannot be found.",
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND)

        return None

    async def resolve_lora(
            self, lora_name: str) -> Union[LoRARequest, ErrorResponse]:
        """Attempt to resolve a LoRA adapter using available resolvers.

        Args:
            lora_name: Name/identifier of the LoRA adapter

        Returns:
            LoRARequest if found and loaded successfully.
            ErrorResponse (404) if no resolver finds the adapter.
            ErrorResponse (400) if adapter(s) are found but none load.
        """
        async with self.lora_resolver_lock[lora_name]:
            # First check if this LoRA is already loaded
            for existing in self.lora_requests:
                if existing.lora_name == lora_name:
                    return existing

            base_model_name = self.model_config.model
            unique_id = self.lora_id_counter.inc(1)
            found_adapter = False

            # Try to resolve using available resolvers
            for resolver in self.lora_resolvers:
                lora_request = await resolver.resolve_lora(
                    base_model_name, lora_name)

                if lora_request is not None:
                    found_adapter = True
                    lora_request.lora_int_id = unique_id

                    try:
                        await self.engine_client.add_lora(lora_request)
                        self.lora_requests.append(lora_request)
                        logger.info(
                            "Resolved and loaded LoRA adapter '{}' using {}",
                            lora_name, resolver.__class__.__name__)
                        return lora_request
                    except BaseException as e:
                        logger.warning(
                            "Failed to load LoRA '{}' resolved by {}: {}. "
                            "Trying next resolver.", lora_name,
                            resolver.__class__.__name__, e)
                        continue

            if found_adapter:
                # An adapter was found, but all attempts to load it failed.
                return create_error_response(
                    message=(f"LoRA adapter '{lora_name}' was found "
                             "but could not be loaded."),
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST)
            else:
                # No adapter was found
                return create_error_response(
                    message=f"LoRA adapter {lora_name} does not exist",
                    err_type="NotFoundError",
                    status_code=HTTPStatus.NOT_FOUND)


def create_error_response(
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
    return ErrorResponse(message=message,
                         type=err_type,
                         code=status_code.value)
