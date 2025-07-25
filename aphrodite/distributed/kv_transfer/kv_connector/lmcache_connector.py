"""
LMCache KV Cache Connector for Distributed Machine Learning Inference

The LMCacheConnector can (1) transfer KV caches between prefill Aphrodite worker
(KV cache producer) and decode Aphrodite worker (KV cache consumer) using
LMCache; (2) offload and share KV caches.
"""

from typing import TYPE_CHECKING, List, Tuple, Union

import torch
from loguru import logger

from aphrodite.common.config import AphroditeConfig
from aphrodite.common.sequence import IntermediateTensors
from aphrodite.distributed.kv_transfer.kv_connector.base import KVConnectorBase

if TYPE_CHECKING:
    from aphrodite.worker.model_runner import (
        ModelInputForGPUWithSamplingMetadata)



class LMCacheConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: AphroditeConfig,
    ):

        self.transfer_config = config.kv_transfer_config
        self.aphrodite_config = config

        from lmcache.experimental.cache_engine import LMCacheEngineBuilder

        from aphrodite.distributed.kv_transfer.kv_connector.v1.lmcache_integration.aphrodite_adapter import (  # noqa
            RetrieveStatus, StoreStatus, init_lmcache_engine,
            lmcache_retrieve_kv, lmcache_should_retrieve, lmcache_should_store,
            lmcache_store_kv)
        from aphrodite.distributed.kv_transfer.kv_connector.v1.lmcache_integration.utils import (  # noqa
            ENGINE_NAME)
        logger.info("Initializing LMCacheConfig under kv_transfer_config {}",
                    self.transfer_config)

        # TODO (Jiayi): Find model_config, parallel_config, and cache_config
        self.engine = init_lmcache_engine(config.model_config,
                                          config.parallel_config,
                                          config.cache_config)
        self.lmcache_engine_name = ENGINE_NAME
        self.lmcache_engine_builder = LMCacheEngineBuilder

        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config
        self.lmcache_retrieve_kv = lmcache_retrieve_kv
        self.lmcache_store_kv = lmcache_store_kv
        self.lmcache_should_retrieve = lmcache_should_retrieve
        self.lmcache_should_store = lmcache_should_store
        self.store_status = StoreStatus
        self.retrieve_status = RetrieveStatus

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        retrieve_status = self.lmcache_should_retrieve(model_input)
        model_input, bypass_model_exec, hidden_or_intermediate_states =\
            self.lmcache_retrieve_kv(
                model_executable, model_input, self.cache_config, kv_caches,
                retrieve_status)
        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        store_status = self.lmcache_should_store(model_input)
        self.lmcache_store_kv(
            self.model_config,
            self.parallel_config,
            self.cache_config,
            model_executable,
            model_input,
            kv_caches,
            store_status,
        )

    def close(self):
        self.lmcache_engine_builder.destroy(self.lmcache_engine_name)
