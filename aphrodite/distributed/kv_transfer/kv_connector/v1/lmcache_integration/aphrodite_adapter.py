import dataclasses
from copy import deepcopy
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from aphrodite.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.cache_engine import (LMCacheEngine,
                                               LMCacheEngineBuilder)
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.gpu_connector import VLLMPagedMemGPUConnectorV2
from lmcache.utils import _lmcache_nvtx_annotate
from loguru import logger

from aphrodite.attention.backends.flash_attn import FlashAttentionMetadata
from aphrodite.common.config import CacheConfig, ModelConfig, ParallelConfig
from aphrodite.common.sequence import IntermediateTensors
from aphrodite.common.utils import get_kv_cache_torch_dtype

from .utils import ENGINE_NAME, lmcache_get_config

LMCACHE_CUDA_STREAM = torch.cuda.Stream()


class StoreStatus(Enum):
    PREFILL = 1
    CHUNK_PREFILL = 2
    DECODE = 3
    SUFFIX_PREFILL = 4
    NONE = 5


class RetrieveStatus(Enum):
    PREFILL = 1  # include (1) normal_prefill
    # (2) chunk_prefill_last
    # (3) prefix_prefill
    CHUNK_PREFILL = 2  # not last chunk
    NONE = 4


def need_gpu_interm_buffer(lmcache_config: LMCacheEngineConfig):
    if lmcache_config.local_cpu:
        return True
    else:
        return False


def init_lmcache_engine(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
) -> Optional[LMCacheEngine]:
    """Initialize the LMCache engine by the given model config and parallel
    config. This function will check the environment variable
    `LMCACHE_CONFIG_FILE` to load the configuration file. If that environment
    variable is not set, this function will return None.

    :param model_config: The model configuration in Aphrodite.
    :type model_config: ModelConfig
    :param parallel_config: The parallel configuration in Aphrodite.
    :type parallel_config: ParallelConfig
    :param cache_config: The KV cache configuration in Aphrodite.
    :type cache_config: CacheConfig

    :return: The initialized LMCache engine or None (if the environment variable
        `LMCACHE_CONFIG_FILE` is not set).
    :rtype: Optional[LMCacheEngine]
    """
    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        return None

    config = lmcache_get_config()
    assert isinstance(config, LMCacheEngineConfig), \
        "LMCache experimental configuration is should be passed."

    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype,
                                        model_config.dtype)

    # construct kv shape (for mem pool)
    num_layer = model_config.get_num_layers(parallel_config)
    chunk_size = config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)

    # Change current device.
    torch.cuda.device(parallel_config.rank)
    device = torch.device(f"cuda:{parallel_config.rank}")
    metadata = LMCacheEngineMetadata(model_config.model,
                                     parallel_config.world_size,
                                     parallel_config.rank, "aphrodite", kv_dtype,
                                     kv_shape)
    hidden_dim_size = num_kv_head * head_size
    use_gpu = need_gpu_interm_buffer(config)
    aphrodite_gpu_connector = VLLMPagedMemGPUConnectorV2(hidden_dim_size,
                                                    num_layer,
                                                    use_gpu=use_gpu,
                                                    chunk_size=chunk_size,
                                                    dtype=kv_dtype,
                                                    device=device)
    engine = LMCacheEngineBuilder.get_or_create(ENGINE_NAME, config, metadata,
                                                aphrodite_gpu_connector)

    return engine


def broadcast_seq_group_list(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    is_driver_worker: bool,
) -> "ModelInputForGPUWithSamplingMetadata":
    """Broadcast the `model_input` from driver worker to non-driver workers.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param is_driver_worker: Whether the code is executed in driver worker.
    :type is_driver_worker: bool

    : return: Original `model_input` if driver_worker.
              Broadcasted `model_input` otherwise.
    """

    # broadcast len of `seq_group_metadata_list`
    if is_driver_worker:
        assert model_input.sampling_metadata is not None
        assert model_input.sampling_metadata.seq_groups is not None
        seq_group_len_list = [len(model_input.sampling_metadata.seq_groups)]
    else:
        seq_group_len_list = [0]
    dist.broadcast_object_list(seq_group_len_list, src=0)
    seq_group_len = seq_group_len_list[0]

    # broadcast `seq_groups`
    if is_driver_worker:
        seq_groups = model_input.sampling_metadata.seq_groups  # type: ignore
    else:
        seq_groups = [None] * seq_group_len
    dist.broadcast_object_list(seq_groups, src=0)

    if is_driver_worker:
        return model_input
    else:
        sampling_metadata = model_input.sampling_metadata
        sampling_metadata.seq_groups = seq_groups  # type: ignore
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata)


def close_lmcache_engine() -> None:
    """Close the LMCache engine if it is initialized.
    """
    logger.debug("Closing LMCache Engine")
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


# This function is not used for now
def lmcache_should_retrieve(
    model_input: "ModelInputForGPUWithSamplingMetadata",
) -> List[RetrieveStatus]:
    """Check should we retrieve KV from LMCache for the current model_input.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory
    :type kv_caches: List[torch.Tensor]

    :return: RetrieveStatus.
    """

    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."

    # model_input doesn't have seq_lens in tp
    # but attn_metadata does
    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None
    num_seqs = len(seq_lens)
    retrieve_status = [RetrieveStatus.NONE] * num_seqs
    has_engine = LMCacheEngineBuilder.get(ENGINE_NAME) is not None
    if not has_engine:
        return retrieve_status

    attn_meta = model_input.attn_metadata

    prefill_exist = (attn_meta.num_prefills > 0)
    if not prefill_exist:
        return retrieve_status
    assert model_input.sampling_metadata is not None
    seq_group_list = model_input.sampling_metadata.seq_groups
    model_input = broadcast_seq_group_list(model_input, seq_group_list
                                           is not None)
    seq_group_list = model_input.sampling_metadata.seq_groups
    assert seq_group_list is not None

    seq_data_idx = 0
    #selected_token_indices_idx = 0
    for seq_group_idx, seq_group in enumerate(seq_group_list):
        num_seqs_in_seq_group = len(seq_group.seq_data)
        seq_data_idx_end = seq_data_idx + num_seqs_in_seq_group

        # DECODE
        if not seq_group.is_prompt:
            seq_data_idx = seq_data_idx_end
            continue

        # CHUNK_PREFILL
        if not seq_group.do_sample:
            retrieve_status[seq_data_idx:seq_data_idx_end] =\
                [RetrieveStatus.CHUNK_PREFILL] * num_seqs_in_seq_group
            seq_data_idx = seq_data_idx_end
        # LAST_CHUNK_PREFILL or NORMAL_PREFILL
        else:
            retrieve_status[seq_data_idx:seq_data_idx_end] =\
                [RetrieveStatus.PREFILL] * num_seqs_in_seq_group
            seq_data_idx = seq_data_idx_end

    return retrieve_status


def lmcache_should_store(
    model_input: "ModelInputForGPUWithSamplingMetadata",
) -> List[StoreStatus]:
    """Check should we store KV into LMCache for the current model_input.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata


    :return: A list of StoreStatus.
             StoreStatus.PREFILL/DECODE/CHUNK_PREFILL if
             we should store KV after PREFILL/DECODE.
             StoreStatus.NONE if no storing is required.
    """

    def is_blend_effective(attn_metadata):
        """Check if the blend is effective for the current request
        """
        blend_metadata = getattr(attn_metadata, "blend_metadata", None)
        if blend_metadata is None:
            return False

        return blend_metadata.processed_layer_count > 0

    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."

    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None
    store_status = [StoreStatus.NONE] * len(seq_lens)
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    has_engine = engine is not None
    if not has_engine:
        return store_status
    assert engine is not None

    attn_meta = model_input.attn_metadata

    # Don't store if this request is processed by cacheblend
    if is_blend_effective(attn_meta):
        return store_status

    assert model_input.sampling_metadata is not None
    seq_group_list = model_input.sampling_metadata.seq_groups
    # FIXME(Jiayi): Use `seq_group_list` to determine driver worker
    # Alternative 1, we can pass in a parameter `is_driver_worker`
    # Alternative 2, make the broadcast in outside, so the `broadcast`
    # doesn't need to be done twice in `lmcache_retrieve` and
    # `lmcache_store`
    # We use this dirty fix now as we don't want to modify the aphrodite
    # connector interface for now
    model_input = broadcast_seq_group_list(model_input, seq_group_list
                                           is not None)
    seq_group_list = model_input.sampling_metadata.seq_groups
    assert seq_group_list is not None

    selected_token_indices = \
        model_input.sampling_metadata.selected_token_indices

    seq_data_idx = 0
    selected_token_indices_idx = 0
    for seq_group_idx, seq_group in enumerate(seq_group_list):
        num_seqs_in_seq_group = len(seq_group.seq_data)
        seq_data_idx_end = seq_data_idx + num_seqs_in_seq_group

        # DECODE
        if not seq_group.is_prompt:
            # Determine whether to save decoded KV cache
            if not engine.config.save_decode_cache:
                for idx in range(seq_data_idx, seq_data_idx_end):
                    if seq_lens[idx] % engine.config.chunk_size == 0:
                        store_status[idx] = StoreStatus.DECODE
            seq_data_idx = seq_data_idx_end
            selected_token_indices_idx += num_seqs_in_seq_group
            continue

        # TODO(Jiayi): Maybe it's cleaner to handle all logic for
        # `lmcache_model_request` inside `cache_engine`
        # Check whether user has specified to not store the cache
        if hasattr(seq_group, "lmcache_model_request"):
            lmcache_model_request = seq_group.lmcache_model_request
            if lmcache_model_request is not None:
                user_should_store = lmcache_model_request.store_cache
                if not user_should_store:
                    logger.debug("User has specified not to store the cache")
                    seq_data_idx += len(seq_group.seq_data)
                    continue

        # CHUNK_PREFILL
        if not seq_group.do_sample:
            store_status[seq_data_idx:seq_data_idx_end] = \
                [StoreStatus.CHUNK_PREFILL] * num_seqs_in_seq_group
            seq_data_idx = seq_data_idx_end
            continue

        # LAST_CHUNK_PREFILL or NORMAL_PREFILL
        for seqid, seq_data in seq_group.seq_data.items():
            if seq_data.get_len(
            ) - 1 != selected_token_indices[selected_token_indices_idx]:
                # last chunk in chunk prefill
                # or prefix already hit in retrieve
                store_status[seq_data_idx] = StoreStatus.SUFFIX_PREFILL
            else:
                store_status[seq_data_idx] = StoreStatus.PREFILL
            seq_data_idx += 1
            selected_token_indices_idx += 1
    return store_status


@_lmcache_nvtx_annotate
def lmcache_store_kv(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor],
    store_status: List[StoreStatus],
) -> None:
    """Store the KV caches into LMCache for the current model_input.

    :param model_executable: The model executable for the current request.
    :type model_executable: torch.nn.Module

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory to get KV from
    :type kv_caches: List[torch.Tensor]

    :param store_status: Indicate whether and how KV cache of each req is stored
    :type store_status: List[StoreStatus]
    """
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine is not initialized."

    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."

    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None

    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
    assert slot_mapping is not None

    query_start_loc = model_input.attn_metadata.query_start_loc
    assert query_start_loc is not None

    block_tables = model_input.attn_metadata.block_tables

    # TODO (Jiayi): commenting the following out for now
    # as Turing architecture is not supported yet
    # For Turing GPU
    # num_heads = model_config.get_num_kv_heads(parallel_config)
    # head_size = model_config.get_head_size()
    # gpu_capability = torch.cuda.get_device_capability()

    seq_data_idx = 0
    assert model_input.sampling_metadata is not None

    seq_group_list = model_input.sampling_metadata.seq_groups

    assert seq_group_list is not None

    next_start_pos = 0
    for seq_group_idx, seq_group in enumerate(seq_group_list):
        for seqid, seq_data in seq_group.seq_data.items():
            status = store_status[seq_data_idx]
            # TODO (Jiayi): can chunk prefill and aphrodite prefix
            # caching use the same logic?
            if status in [StoreStatus.NONE]:
                continue
            elif status in [
                    StoreStatus.SUFFIX_PREFILL, StoreStatus.CHUNK_PREFILL
            ]:
                seq_len = seq_lens[seq_data_idx]
            else:
                seq_len = seq_data.get_len()
                if status == StoreStatus.DECODE:
                    if seq_len % engine.config.chunk_size != 0:
                        continue
            current_tokens = torch.tensor(seq_data.get_token_ids()[:seq_len],
                                          device="cpu")

            skip_leading_tokens = engine.lookup(current_tokens)
            assert skip_leading_tokens <= seq_len

            aphrodite_num_required_tokens = (query_start_loc[seq_data_idx + 1] -
                                        query_start_loc[seq_data_idx]).item()
            assert isinstance(aphrodite_num_required_tokens, int)

            start_pos = next_start_pos
            end_pos = start_pos + aphrodite_num_required_tokens
            next_start_pos = end_pos

            aphrodite_num_computed_tokens = seq_len - aphrodite_num_required_tokens
            if aphrodite_num_computed_tokens > 0:
                if skip_leading_tokens >= aphrodite_num_computed_tokens:
                    slot_mapping_req_full = torch.full(
                        (seq_len, ),
                        -1,
                        device=slot_mapping.device,
                        dtype=slot_mapping.dtype)
                    slot_mapping_req_full[aphrodite_num_computed_tokens:] = \
                        slot_mapping[start_pos:end_pos]
                else:
                    # NOTE(Jiayi): the cache is stored even if it's in aphrodite
                    # as long as it's not in lmc
                    assert block_tables is not None
                    block_table_full = block_tables[seq_group_idx]
                    aphrodite_block_size = cache_config.block_size

                    n_block = len(block_table_full)
                    indices = torch.arange(
                        aphrodite_block_size,
                        device=slot_mapping.device,
                        dtype=slot_mapping.dtype).repeat(n_block)
                    slot_mapping_req_full = aphrodite_block_size \
                        * block_table_full.repeat_interleave(aphrodite_block_size)\
                        + indices
                    slot_mapping_req_full = slot_mapping_req_full[:seq_len]

            else:
                slot_mapping_req_full = slot_mapping[start_pos:end_pos]

            if skip_leading_tokens < seq_len:
                assert skip_leading_tokens % engine.config.chunk_size == 0

                # TODO(Jiayi): Turing is not supported yet
                # need to write mem kernels for turing architecture

                # TODO(Jiayi): prefix caching and chunk prefill
                # might error here. `slot_mapping_seq` could be wrong

                stored_token_num = seq_len - skip_leading_tokens
                kv_tensors_mask = torch.ones_like(current_tokens,
                                                  dtype=torch.bool)
                kv_tensors_mask[:skip_leading_tokens] = False

                engine.store(current_tokens.cpu(),
                             kv_tensors_mask,
                             kvcaches=kv_caches,
                             slot_mapping=slot_mapping_req_full,
                             offset=skip_leading_tokens)
            else:
                stored_token_num = 0
                skip_leading_tokens = seq_len
            logger.debug(f"Store skips {skip_leading_tokens} tokens "\
                    f"and then stores {stored_token_num} tokens")
            seq_data_idx += 1


@_lmcache_nvtx_annotate
def lmcache_retrieve_kv(
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    cache_config: CacheConfig,
    kv_caches: List[torch.Tensor],
    retrieve_status: List[RetrieveStatus],
) -> Tuple["ModelInputForGPUWithSamplingMetadata", bool, Union[
        torch.Tensor, IntermediateTensors]]:
    """Retrieve the KV caches from LMCache for the current model_input. And
    rebuild the model_input to reflect the changes in KV if necessary.

    :param model_executable: The model executable for the current request.
    :type model_executable: torch.nn.Module

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory to put KV to
    :type kv_caches: List[torch.Tensor]

    :param retrieve_status: Indicate whether and how
                            KV cache of each req is retrieved
    :type retrieve_status: List[RetrieveStatus]

    :return: The rebuilt model_input to reflect the changes in KV.
    :return: The boolean value to indicate whether the
             entire execute_model should be skipped
    """

    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine is not initialized."

    if engine.config.enable_blending:
        return model_input, False, None

    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."

    query_start_loc = model_input.attn_metadata.query_start_loc
    assert query_start_loc is not None
    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
    assert slot_mapping is not None
    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None

    # The following metadata are needed to rebuilt the model input
    full_tokens_list = []
    num_computed_tokens_list = []
    lmc_num_computed_tokens_list = []

    start_pos_list = []
    is_prefill_list = []

    do_sample_list = []

    next_start_pos = 0
    num_request_not_found = 0

    # idx is on a sequence, not a sequence group.
    idx = 0

    assert model_input.sampling_metadata is not None
    seq_group_list = model_input.sampling_metadata.seq_groups

    assert seq_group_list is not None

    chunk_prefill_full_hit = True
    for seq_group in seq_group_list:
        seq_ids = seq_group.seq_ids
        for seq_id in seq_ids:
            seq_data = seq_group.seq_data[seq_id]
            is_prefill_list.append(seq_group.is_prompt)
            if retrieve_status[idx] == RetrieveStatus.CHUNK_PREFILL:
                total_seq_len = seq_lens[idx]
                do_sample_list.append(False)
            else:
                total_seq_len = seq_data.get_len()
                do_sample_list.append(True)

            full_token_tensor = torch.tensor(
                seq_data.get_token_ids()[:total_seq_len], device="cpu")
            full_tokens_list.append(full_token_tensor)

            aphrodite_num_required_tokens = (query_start_loc[idx + 1] -
                                        query_start_loc[idx]).item()
            assert isinstance(aphrodite_num_required_tokens, int)

            start_pos = next_start_pos
            end_pos = start_pos + aphrodite_num_required_tokens
            next_start_pos = end_pos
            start_pos_list.append(start_pos)

            # number of tokens already computed by aphrodite
            # (e.g., chunk prefill, prefix caching)
            aphrodite_num_computed_tokens = total_seq_len - aphrodite_num_required_tokens

            # NOTE: No need to retrieve from lmc if the current sequence is
            # in DECODE stage
            if retrieve_status[idx] == RetrieveStatus.NONE:
                assert aphrodite_num_required_tokens == 1
                total_seq_len = seq_lens[idx]
                num_computed_tokens_list.append(aphrodite_num_computed_tokens)
                lmc_num_computed_tokens_list.append(0)
                num_request_not_found += 1
                idx += 1
                logger.debug("Injected token number: 0. This is DECODE")
                continue

            # NOTE: No need to retrieve from lmc if the number of tokens
            # to be retrieved is small
            lmc_chunk_size = engine.config.chunk_size
            if aphrodite_num_required_tokens < lmc_chunk_size:
                num_computed_tokens_list.append(aphrodite_num_computed_tokens)
                lmc_num_computed_tokens_list.append(0)
                idx += 1
                num_request_not_found += 1
                continue

            # construct token mesk to indicate what tokens should be retrieved
            # from lmc. Tokens computed in aphrodite already should be skipped
            token_mask = torch.ones_like(full_token_tensor, dtype=torch.bool)
            aphrodite_num_computed_tokens_align = aphrodite_num_computed_tokens\
                // lmc_chunk_size * lmc_chunk_size
            token_mask[:aphrodite_num_computed_tokens_align] = False

            # TODO(Jiayi): Please get rid of this in the future
            # Please only pass the required slot_mapping to the engine
            if aphrodite_num_computed_tokens > 0:
                slot_mapping_req_full = torch.full((total_seq_len, ),
                                                   -1,
                                                   device=slot_mapping.device,
                                                   dtype=slot_mapping.dtype)
                slot_mapping_req_full[aphrodite_num_computed_tokens:] = \
                    slot_mapping[start_pos:end_pos]
            else:
                slot_mapping_req_full = slot_mapping[start_pos:end_pos]

            # call lmcache retrieve
            ret_token_mask = engine.retrieve(
                full_token_tensor,
                token_mask,
                kvcaches=kv_caches,
                slot_mapping=slot_mapping_req_full)
            lmc_num_computed_tokens = max(
                    torch.sum(ret_token_mask).item() - \
                    (aphrodite_num_computed_tokens - aphrodite_num_computed_tokens_align),
                    0
                )

            assert isinstance(lmc_num_computed_tokens, int)

            # total number of computed tokens (aphrodite + lmc)
            num_computed_tokens = aphrodite_num_computed_tokens + \
                lmc_num_computed_tokens

            # TODO(Jiayi): currently we do not skip anything if chunked prefill
            # is batched with any decode or other chunked prefills.
            if retrieve_status[idx] == RetrieveStatus.CHUNK_PREFILL:
                if num_computed_tokens != total_seq_len:
                    chunk_prefill_full_hit = False
                else:
                    lmc_num_computed_tokens -= 1
                    num_computed_tokens -= 1
            else:
                # Avoid error when prefix is exactly the same as the retrieved
                # However, the entire prefill should be skipped in chunk prefill
                if num_computed_tokens == total_seq_len:
                    lmc_num_computed_tokens -= 1
                    num_computed_tokens -= 1

            num_computed_tokens_list.append(num_computed_tokens)
            lmc_num_computed_tokens_list.append(lmc_num_computed_tokens)

            # No cache found, move on
            if lmc_num_computed_tokens == 0:
                num_request_not_found += 1

            # Inject the lmc retrieved kv cache
            logger.debug(f"Injected token number: {lmc_num_computed_tokens}")

            idx += 1

    seq_cnt = len(query_start_loc) - 1
    assert idx == seq_cnt
    assert len(lmc_num_computed_tokens_list) == seq_cnt
    assert len(num_computed_tokens_list) == seq_cnt

    is_all_chunk_prefill = all(
        [status == RetrieveStatus.CHUNK_PREFILL for status in retrieve_status])

    # NOTE: We can only skip model forward if all requests are chunk prefill

    if is_all_chunk_prefill and chunk_prefill_full_hit:
        num_tok = len(model_input.input_tokens)
        num_dim = model_executable.model.embed_tokens.embedding_dim
        dtype = model_executable.model.embed_tokens.weight.dtype
        device = model_input.input_tokens.device
        hidden_or_intermediate_states = torch.zeros(num_tok,
                                                    num_dim,
                                                    device=device,
                                                    dtype=dtype)
        logger.debug("Skip the entire model forward!")
        return model_input, True, hidden_or_intermediate_states

    if num_request_not_found < seq_cnt:
        rebuilt_model_input = build_partial_prefill_input(
            model_input,
            full_tokens_list,
            num_computed_tokens_list,
            start_pos_list,
            slot_mapping,
            lmc_num_computed_tokens_list,
            is_prefill_list,
            do_sample_list,
            kv_caches[0][0].device,
            cache_config,
        )
        logger.debug("Rebuilt the input!")
        return rebuilt_model_input, False, None

    logger.debug("Returning the original input!")
    return model_input, False, None


def build_partial_prefill_input(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    full_tokens_list: List[torch.Tensor],
    num_computed_tokens_list: List[int],
    start_pos_list: List[int],
    slot_mapping_flat: torch.Tensor,
    lmc_num_computed_tokens_list: List[int],
    is_prefill_list: List[bool],
    do_sample_list: List[bool],
    device: torch.device,
    cache_config: CacheConfig,
) -> "ModelInputForGPUWithSamplingMetadata":
    """Helper function to rebuild the model input for the current request.
    """
    assert model_input.attn_metadata is not None
    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
        "Only FlashAttention backend is supported for now."
    assert model_input.attn_metadata.context_lens_tensor is not None
    assert model_input.attn_metadata.block_tables is not None
    assert model_input.attn_metadata.query_start_loc is not None
    assert model_input.input_positions is not None

    rebuilt_input_tokens = []
    rebuilt_input_positions = []
    rebuilt_query_lens = []
    rebuilt_num_prefills = 0
    rebuilt_num_prefill_tokens = 0
    rebuilt_slot_mapping = []
    rebuilt_max_query_len = 0

    rebuilt_block_tables = []

    rebuilt_query_start_loc = [0]
    rebuilt_context_lens_tensor = []
    rebuilt_selected_token_indices = []

    last_query_start_loc = 0

    # recounting query and context lengths
    for idx in range(len(full_tokens_list)):
        token_tensor = full_tokens_list[idx]
        num_token = len(token_tensor)
        num_computed_token = num_computed_tokens_list[idx]
        start_pos = start_pos_list[idx]
        is_prefill = is_prefill_list[idx]
        lmc_num_computed_tokens = lmc_num_computed_tokens_list[idx]
        rebuilt_input_tokens.append(token_tensor[num_computed_token:])
        q_len = num_token - num_computed_token
        assert q_len > 0
        rebuilt_query_lens.append(q_len)
        start_input_pos_idx = start_pos + lmc_num_computed_tokens
        end_input_pos_idx = start_input_pos_idx + q_len
        rebuilt_input_positions.append(
            model_input.input_positions[start_input_pos_idx:end_input_pos_idx])
        # Attn metadata-related
        if is_prefill:
            rebuilt_num_prefills += 1
            rebuilt_num_prefill_tokens += q_len
        else:
            assert q_len == 1

        start_slot_idx = start_pos + lmc_num_computed_tokens
        end_slot_idx = start_slot_idx + q_len
        new_slot_mapping = slot_mapping_flat[start_slot_idx:end_slot_idx]
        rebuilt_slot_mapping.append(new_slot_mapping)
        rebuilt_max_query_len = max(q_len, rebuilt_max_query_len)

        last_query_start_loc += q_len
        rebuilt_query_start_loc.append(last_query_start_loc)  # start with 0
        rebuilt_context_lens_tensor.append(num_computed_token)

        # recover `block_table`
        if len(model_input.attn_metadata.block_tables[idx]) > 0:
            rebuilt_block_tables.append(
                model_input.attn_metadata.block_tables[idx])
        else:
            slot_mapping_req = slot_mapping_flat[start_pos:end_slot_idx]
            aphrodite_block_size = cache_config.block_size
            rebuilt_block_table = slot_mapping_req[::16].to(torch.int32) \
                // aphrodite_block_size
            rebuilt_block_tables.append(rebuilt_block_table)

        # Sampling metadata related
        # seq_groups (use rebuilt query lens)
        if do_sample_list[idx]:
            rebuilt_selected_token_indices.append(last_query_start_loc - 1)

    # rebuilt attn_metadata
    rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
    rebuilt_attn_metadata.num_prefills = rebuilt_num_prefills
    rebuilt_attn_metadata.num_prefill_tokens = rebuilt_num_prefill_tokens
    rebuilt_attn_metadata.slot_mapping = torch.cat(rebuilt_slot_mapping).to(
        device)
    rebuilt_attn_metadata.max_query_len = rebuilt_max_query_len

    rebuilt_attn_metadata.block_tables = pad_sequence(
        rebuilt_block_tables, batch_first=True).to(device)

    rebuilt_attn_metadata.query_start_loc = torch.tensor(
        rebuilt_query_start_loc,
        dtype=model_input.attn_metadata.query_start_loc.dtype).to(device)
    rebuilt_attn_metadata.context_lens_tensor = torch.tensor(
        rebuilt_context_lens_tensor,
        dtype=model_input.attn_metadata.context_lens_tensor.dtype,
    ).to(device)

    rebuilt_attn_metadata._cached_prefill_metadata = None
    rebuilt_sampling_metadata = None
    # rebuilt sampling_metadata
    if model_input.sampling_metadata is not None:
        rebuilt_sampling_metadata = deepcopy(model_input.sampling_metadata)
        for idx, q_len in enumerate(rebuilt_query_lens):
            if rebuilt_sampling_metadata.seq_groups is not None:
                rebuilt_sampling_metadata.seq_groups[idx].query_len = q_len

        rebuilt_sampling_metadata.selected_token_indices = torch.tensor(
            rebuilt_selected_token_indices,
            dtype=model_input.sampling_metadata.selected_token_indices.dtype,
        ).to(device)

    # import here to avoid circular import.
    from aphrodite.worker.model_runner import (
        ModelInputForGPUWithSamplingMetadata)
    rebuilt_model_input = ModelInputForGPUWithSamplingMetadata(
        input_tokens=torch.cat(rebuilt_input_tokens).to(device),
        input_positions=torch.cat(rebuilt_input_positions).to(device),
        seq_lens=model_input.seq_lens,
        query_lens=rebuilt_query_lens,
        lora_mapping=model_input.lora_mapping,
        lora_requests=model_input.lora_requests,
        attn_metadata=rebuilt_attn_metadata,
        prompt_adapter_mapping=model_input.prompt_adapter_mapping,
        prompt_adapter_requests=model_input.prompt_adapter_requests,
        multi_modal_kwargs=model_input.multi_modal_kwargs,
        request_ids_to_seq_ids=model_input.request_ids_to_seq_ids,
        finished_requests_ids=model_input.finished_requests_ids,
        virtual_engine=model_input.virtual_engine,
        sampling_metadata=rebuilt_sampling_metadata,
        is_prompt=model_input.is_prompt,
        async_callback=model_input.async_callback,
    )

    return rebuilt_model_input
