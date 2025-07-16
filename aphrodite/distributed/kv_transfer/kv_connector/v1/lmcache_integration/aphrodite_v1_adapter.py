# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import zmq
from lmcache.experimental.cache_engine import LMCacheEngine
from loguru import logger

import aphrodite.common.envs as envs
from aphrodite.common.config import AphroditeConfig
from aphrodite.common.utils import cdiv, make_zmq_socket
from aphrodite.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from aphrodite.v1.core.sched.output import SchedulerOutput
from aphrodite.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from .aphrodite_adapter import init_lmcache_engine

if TYPE_CHECKING:
    from aphrodite.attention.backends.abstract import AttentionMetadata
    from aphrodite.forward_context import ForwardContext
    from aphrodite.v1.core.kv_cache_manager import KVCacheManager
    from aphrodite.v1.core.sched.output import (CachedRequestData,
                                                NewRequestData)
    from aphrodite.v1.request import Request


def get_zmq_rpc_path_lmcache(
        role: KVConnectorRole,
        is_tp: bool = False,
        aphrodite_config: Optional["AphroditeConfig"] = None) -> str:
    base_url = envs.APHRODITE_RPC_BASE_PATH
    # Default to 0 if not configured
    rpc_port = 0
    if aphrodite_config is not None:
        rpc_port = aphrodite_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0)
    logger.debug("Base URL: {}, RPC Port: {}", base_url, rpc_port)
    return f"ipc://{base_url}/lmcache_rpc_port_{rpc_port}"


# TODO: move this to LMCache so that we can gracefully close it
class LMCacheLookupClient:

    def __init__(self, role: KVConnectorRole, is_tp: bool,
                 aphrodite_config: "AphroditeConfig"):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lmcache(role, is_tp, aphrodite_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False)

    def lookup(self, token_ids: torch.Tensor) -> int:
        request = self.encoder.encode(token_ids)
        self.socket.send_multipart(request, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):
        self.socket.close(linger=0)


class LMCacheLookupServer:

    def __init__(self, lmcache_engine: LMCacheEngine, role: KVConnectorRole,
                 is_tp: bool, aphrodite_config: "AphroditeConfig"):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lmcache(role, is_tp, aphrodite_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True)

        self.lmcache_engine = lmcache_engine
        self.running = True

        def process_request():
            while self.running:
                try:
                    #request = self.socket.recv()
                    frames = self.socket.recv_multipart(copy=False)
                    token_ids = self.decoder.decode(frames)
                    result = self.lmcache_engine.lookup(token_ids)
                    response = result.to_bytes(4, "big")
                    self.socket.send(response)
                except Exception as e:
                    logger.error("Error in LMCache lookup server: {}", e)
                    break
                #continue

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    aphrodite_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: list[int]

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will 
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.

        """
        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].
            copy(),
            allocated_block_ids=new_request.block_ids.copy(),
            num_saved_tokens=0,
        )

    def update(
        self,
        cached_request: "CachedRequestData",
    ) -> None:
        """Update the request tracker when a running request is 
        scheduled again
        """
        self.token_ids.extend(cached_request.new_token_ids)
        self.allocated_block_ids.extend(cached_request.new_block_ids)


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor
    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        skip_save: bool = False,
        discard_partial_chunks: bool = True,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            skip_save (bool): whether to skip the save operation.
            discard_partial_chunks (bool): whether to discard partial chunks.

        Returns:
            the request metadata if we need to perform load/save 
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = cdiv(tracker.num_saved_tokens, lmcache_chunk_size) * \
                lmcache_chunk_size
        skip_save = skip_save or (tracker.num_saved_tokens > 0 and \
                input_token_len < chunk_boundary)

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting
        num_tokens_to_save = (
            input_token_len // lmcache_chunk_size *
            lmcache_chunk_size) if discard_partial_chunks else input_token_len

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        # OPTIMIZATION: pre-allocate the buffer for token ids and block
        # ids
        token_ids = torch.tensor(input_token_ids)[:num_tokens_to_save]
        num_blocks = len(tracker.allocated_block_ids)
        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks."
                "Something might be wrong in scheduling logic!")
            logger.error("Num tokens: {}, num blocks: {}, block size: {}",
                         len(token_ids), num_blocks, block_size)

        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
                block_ids.reshape((num_blocks, 1)) * block_size

        slot_mapping = slot_mapping.flatten()[:len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug("Scheduled to load {} tokens for request {}",
                         load_spec.lmcache_cached_tokens, tracker.req_id)
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            save_spec=save_spec,
            load_spec=load_spec,
        )


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1Impl:

    def __init__(self, aphrodite_config: "AphroditeConfig",
                 role: KVConnectorRole, parent: KVConnectorBase_V1):
        self._parent = parent
        self.kv_role = aphrodite_config.kv_transfer_config.kv_role
        is_tp = aphrodite_config.parallel_config.tensor_parallel_size > 1
        if role == KVConnectorRole.SCHEDULER:
            self.lookup_client = LMCacheLookupClient(role, is_tp,
                                                     aphrodite_config)
        else:
            self.lmcache_engine = init_lmcache_engine(
                aphrodite_config.model_config,
                aphrodite_config.parallel_config,
                aphrodite_config.cache_config)
            # NOTE: Only create the KV lookup API server on worker rank 0
            # when there are multiple workers
            assert self.lmcache_engine is not None
            if aphrodite_config.parallel_config.rank == 0:
                self.lookup_server = LMCacheLookupServer(
                    self.lmcache_engine, role, is_tp, aphrodite_config)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = aphrodite_config.cache_config.block_size

        # request_id -> (aphrodite cached tokes, lmcache cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}

        self.kv_cache_manager: Optional[KVCacheManager] = None

        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}

        # Whether to discard partial chunks
        self._discard_partial_chunks = aphrodite_config\
                .kv_transfer_config.get_from_extra_config(
                    "discard_partial_chunks", False)

        # TODO: need to align this chunk size with lmcache
        self._lmcache_chunk_size = 256

        self.skip_last_n_tokens = \
            aphrodite_config.kv_transfer_config.get_from_extra_config(
                "skip_last_n_tokens", 0)

    def _init_kv_caches_from_forward_context(
            self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer {} does not have kv_cache, skip it",
                             layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[\
                        forward_context.virtual_engine]

    ####################
    # Worker side APIs
    ####################

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's 
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
        """
        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the attn_metadata is None")
            return

        # HACK: getting chunk size to correctly calculate retrieve mask
        assert self.lmcache_engine is not None
        lmcache_chunk_size = self.lmcache_engine.config.chunk_size

        for request in metadata.requests:
            if request.load_spec is None:
                continue

            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.cuda()
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones_like(tokens, dtype=torch.bool)
            masked_token_count = request.load_spec.aphrodite_cached_tokens // \
                lmcache_chunk_size * lmcache_chunk_size
            token_mask[:masked_token_count] = False

            if self.skip_last_n_tokens > 0:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:-self.skip_last_n_tokens],
                    token_mask[:-self.skip_last_n_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:-self.skip_last_n_tokens],
                )
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens,
                    token_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                )

            # Check the result
            num_retrieved_tokens = ret_token_mask.sum().item()
            num_expected_tokens = request.load_spec.lmcache_cached_tokens - \
                    request.load_spec.aphrodite_cached_tokens - \
                    self.skip_last_n_tokens
            if num_retrieved_tokens < num_expected_tokens:
                logger.error(
                    "The number of retrieved tokens is less than the "
                    "expected number of tokens! This should not happen!")
                logger.error(
                    "Num retrieved tokens: {}, num expected tokens: {}",
                    num_retrieved_tokens, num_expected_tokens)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer. 
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer 
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        if layer_name not in self.kv_caches:
            self.kv_caches[layer_name] = kv_layer

    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""
        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        # HACK: getting chunk size to correctly calculate store mask
        assert self.lmcache_engine is not None
        lmcache_chunk_size = self.lmcache_engine.config.chunk_size

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if save_spec is None or not save_spec.can_save:
                continue

            token_ids = request.token_ids
            assert isinstance(token_ids, torch.Tensor)
            assert token_ids.is_cpu

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.cuda()
            # NOTE: In PD setting, lmcache_engine.lookup() will always return
            # 0 if there is no local storage configured. In this case, we
            # should rely on the slip_leading_tokens in save_spec to avoid
            # transmit the already saved tokens again.
            skip_leading_tokens = max(self.lmcache_engine.lookup(token_ids),
                                      save_spec.skip_leading_tokens)
            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = skip_leading_tokens // \
                    lmcache_chunk_size * lmcache_chunk_size

            store_mask = torch.ones_like(token_ids, dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.info(
                "Storing KV cache for {} out of {} tokens for request {}",
                len(token_ids) - skip_leading_tokens, len(token_ids),
                request.req_id)
            self.lmcache_engine.store(token_ids,
                                      mask=store_mask,
                                      kvcaches=kvcaches,
                                      slot_mapping=slot_mapping,
                                      offset=skip_leading_tokens)

    ###################
    # Scheduler side APIs
    ####################

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        """
        Check for external KV cache hit.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the 
            external KV cache beyond what is already computed.
        """
        if self.kv_role == "kv_producer":
            return 0

        token_ids = torch.tensor(request.prompt_token_ids)
        if self.skip_last_n_tokens > 0:
            num_external_hit_tokens = self.lookup_client.lookup(
                token_ids[:-self.skip_last_n_tokens])
        else:
            num_external_hit_tokens = self.lookup_client.lookup(token_ids)

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1

        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.info(
            "Reqid: {}, Total tokens {}, LMCache hit tokens: {}, "
            "need to load: {}", request.request_id, request.num_tokens,
            num_external_hit_tokens, need_to_allocate)

        if need_to_allocate <= 0:
            return 0

        self.load_specs[request.request_id] = LoadSpec(
            aphrodite_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False)

        # TODO: Align to vLLM block size. Should test whether it can be removed
        #need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size
        return need_to_allocate

    def update_state_after_alloc(self, request: "Request",
                                 num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        assert num_external_tokens > 0 and num_external_tokens == \
            self.load_specs[request.request_id].lmcache_cached_tokens - \
            self.load_specs[request.request_id].aphrodite_cached_tokens, \
            f"Mismatch in number of tokens: {num_external_tokens} vs " \
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} - " \
            f"{self.load_specs[request.request_id].aphrodite_cached_tokens}" \
            f" for request {request.request_id}"

        self.load_specs[request.request_id].can_load = True

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output 
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer"

        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = request.num_computed_tokens + \
                    scheduler_output.num_scheduled_tokens[request.req_id]
            request_tracker = RequestTracker.from_new_request(
                request, num_tokens_to_compute)
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                skip_save=force_skip_save,
                discard_partial_chunks=self._discard_partial_chunks)
            if req_meta is not None:
                meta.add_request(req_meta)

        for request in scheduler_output.scheduled_cached_reqs:
            request_tracker = self._request_trackers[request.req_id]
            request_tracker.update(request)

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=None,
                skip_save=force_skip_save,
                discard_partial_chunks=self._discard_partial_chunks)
            if req_meta is not None:
                meta.add_request(req_meta)

        return meta
