from typing import List, Optional, Tuple

import numpy
import torch

import aphrodite.common.envs as envs
from aphrodite import _custom_ops as ops
from aphrodite.modeling.layers.linear import LinearBase
from aphrodite.platforms import current_platform
from aphrodite.scalar_type import ScalarType, scalar_types

from .quant_utils import pack_cols, unpack_cols

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

# In case there is a performance issue with Marlin, the variable below can be
# changed to False, which allows Marlin to perform global reductions in fp16
# precision (instead of fp32), and therefore, save on some memory movements.
USE_FP32_REDUCE_DEFAULT = True


# For binary size and compile time, we don't support the same types for with and
#  without runtime zero-point. We support common cases, i.e. AWQ and GPTQ.
#  TODO: we may want to move this into the C++ so its closer to the actual impl
def query_marlin_supported_quant_types(has_zp: bool,
                                       device_capability: Optional[int] = None
                                       ):
    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        device_capability = (-1 if capability_tuple is None else
                             capability_tuple.to_int())

    if device_capability < 80:
        return []

    if has_zp:
        # AWQ style, unsigned + runtime zero-point
        return [scalar_types.uint4, scalar_types.uint8]
    else:
        # GPTQ style, unsigned + symmetric bias
        # TODO: once fp8_marlin is merged into "gptq_marlin" we should be able
        #  to add `scalar_types.float8_e4m3fn` here
        return [scalar_types.uint4b8, scalar_types.uint8b128]


def _check_marlin_supported(
        quant_type: ScalarType,
        group_size: Optional[int],
        has_zp: bool,
        device_capability: Optional[int] = None) -> Tuple[bool, Optional[str]]:

    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        device_capability = (-1 if capability_tuple is None else
                             capability_tuple.to_int())

    supported_types = query_marlin_supported_quant_types(
        has_zp, device_capability)

    if quant_type not in supported_types:
        return (False, f"Marlin does not support weight_bits = {quant_type}. "
                f"Only types = {supported_types} "
                f"are supported (for group_size = {group_size}, "
                f"device_capability = {device_capability}, zp = {has_zp}).")
    if (group_size is None or group_size not in MARLIN_SUPPORTED_GROUP_SIZES):
        return (False, f"Marlin does not support group_size = {group_size}. "
                f"Only group_sizes = {MARLIN_SUPPORTED_GROUP_SIZES} "
                "are supported.")

    return True, None


def check_marlin_supported(quant_type: ScalarType,
                           group_size: int,
                           has_zp: bool = False,
                           device_capability: Optional[int] = None) -> bool:
    cond, _ = _check_marlin_supported(quant_type, group_size, has_zp,
                                      device_capability)
    return cond


def verify_marlin_supported(quant_type: ScalarType,
                            group_size: int,
                            has_zp: bool = False) -> None:
    cond, err_msg = _check_marlin_supported(quant_type, group_size, has_zp)
    if not cond:
        assert err_msg is not None
        raise ValueError(err_msg)


def verify_marlin_supports_shape(output_size_per_partition: int,
                                 input_size_per_partition: int,
                                 input_size: int, group_size: int) -> None:

    # Validate output_size_per_partition
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(f"Weight output_size_per_partition = "
                         f"{output_size_per_partition} is not divisible by "
                         f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
                         "Consider reducing tensor_parallel_size or running "
                         "with --quantization gptq.")

    # Validate input_size_per_partition
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(f"Weight input_size_per_partition = "
                         f"{input_size_per_partition} is not divisible "
                         f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "
                         "Consider reducing tensor_parallel_size or running "
                         "with --quantization gptq.")

    if (group_size < input_size
            and input_size_per_partition % group_size != 0):
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition}"
            f" is not divisible by group_size = {group_size}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq.")


def check_marlin_supports_shape(output_size_per_partition: int,
                                input_size_per_partition: int,
                                input_size: int, group_size: int) \
                                    -> Tuple[bool, Optional[str]]:
    try:
        verify_marlin_supports_shape(output_size_per_partition,
                                     input_size_per_partition, input_size,
                                     group_size)
    except ValueError as e:
        return False, e.__str__()
    return True, None


def check_marlin_supports_layer(layer: LinearBase, group_size: int) \
                                    -> bool:
    output_size_per_partition = getattr(layer, "output_size_per_partition",
                                        None) or layer.output_size
    input_size_per_partition = getattr(layer, "input_size_per_partition",
                                       None) or layer.input_size

    return check_marlin_supports_shape(
        output_size_per_partition=output_size_per_partition,
        input_size_per_partition=input_size_per_partition,
        input_size=layer.input_size,
        group_size=group_size)[0]


def check_moe_marlin_supports_layer(layer: LinearBase, group_size: int) \
                                    -> bool:
    hidden_size = layer.hidden_size
    intermediate_size_per_partition = layer.intermediate_size_per_partition

    # gate-up: (n, k) = (intermediate_size_per_partition * 2, hidden_size)
    # down: (n, k) = (hidden_size, intermediate_size_per_partition)
    # moe marlin requires n % 128 == 0 and k % 64 == 0
    return hidden_size % 128 == 0 and \
        intermediate_size_per_partition % max(64, group_size) == 0 and \
        group_size in [-1, 32, 64, 128]


def marlin_make_workspace(output_size_per_partition: int,
                          device: torch.device) -> torch.Tensor:
    max_workspace_size = (output_size_per_partition //
                          GPTQ_MARLIN_MIN_THREAD_N) * GPTQ_MARLIN_MAX_PARALLEL

    return torch.zeros(max_workspace_size,
                       dtype=torch.int,
                       device=device,
                       requires_grad=False)


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(act_order: bool, group_size: int,
                                      is_row_parallel: bool) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)


def marlin_make_empty_zp(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)


def marlin_sort_g_idx(
        g_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def get_scale_perms():
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:

    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_moe_permute_scales(
    s: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
):
    num_experts = s.shape[0]
    output = torch.empty(
        (num_experts, s.shape[1], s.shape[2]),
        device=s.device,
        dtype=s.dtype,
    )

    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output


def marlin_zero_points(zp: torch.Tensor, size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    # Permute zero-points in a similar way to scales, but do not use the
    # "single" permutation, since zero-points are applied on every MMA
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp


def awq_to_marlin_zero_points(q_zp_packed: torch.Tensor, size_k: int,
                              size_n: int, num_bits: int) -> torch.Tensor:
    # AWQ zero-points are quantized and packed on the column dim.
    # In addition, the values are permuted based on dequantizer.
    # Here we undo both of these, and then apply marlin permutation
    # and pack it back.
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)

    # Undo interleaving (use argsort(..) to get inverse perm)
    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()

    marlin_zp = marlin_zero_points(q_zp, size_k, size_n, num_bits)
    return marlin_zp


def moe_awq_to_marlin_zero_points(q_zp_packed: torch.Tensor, size_k: int,
                                  size_n: int, num_bits: int):
    num_experts = q_zp_packed.shape[0]
    output = torch.empty(
        (num_experts, q_zp_packed.shape[1], q_zp_packed.shape[2]),
        device=q_zp_packed.device,
        dtype=q_zp_packed.dtype,
    )
    for e in range(num_experts):
        output[e] = awq_to_marlin_zero_points(q_zp_packed[e], size_k, size_n,
                                              num_bits)
    return output


def should_use_atomic_add_reduce(m: int, n: int, k: int, device: torch.device,
                                 dtype: torch.dtype) -> bool:
    # disable atomicAdd reduce by default,
    # one can enable it with APHRODITE_MARLIN_USE_ATOMIC_ADD=1
    if not envs.APHRODITE_MARLIN_USE_ATOMIC_ADD or device.type != "cuda":
        return False

    # sm8x doesn't support atomicAdd + bfloat16 natively
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        return False

    # the performance of atomicAdd is better than global reduce
    # only when m*n is small and k is large
    return n < 2048 and k >= 2048


def apply_gptq_marlin_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        wtype: ScalarType,
        output_size_per_partition: int,
        input_size_per_partition: int,
        has_zp: bool,
        is_k_full: bool,
        bias: Optional[torch.Tensor] = None,
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition, )

    use_atomic_add = should_use_atomic_add_reduce(m=reshaped_x.size(0),
                                                  n=output_size_per_partition,
                                                  k=reshaped_x.size(1),
                                                  device=input.device,
                                                  dtype=input.dtype)

    output = ops.gptq_marlin_gemm(reshaped_x,
                                  weight,
                                  weight_scale,
                                  weight_zp,
                                  g_idx,
                                  g_idx_sort_indices,
                                  workspace,
                                  wtype,
                                  size_m=reshaped_x.shape[0],
                                  size_n=output_size_per_partition,
                                  size_k=input_size_per_partition,
                                  is_k_full=is_k_full,
                                  use_atomic_add=use_atomic_add,
                                  has_zp=has_zp,
                                  use_fp32_reduce=use_fp32_reduce,
                                  is_zp_float=False)

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)


def apply_awq_marlin_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        quant_type: ScalarType,
        output_size_per_partition: int,
        input_size_per_partition: int,
        bias: Optional[torch.Tensor] = None,
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition, )

    use_atomic_add = should_use_atomic_add_reduce(m=reshaped_x.size(0),
                                                  n=output_size_per_partition,
                                                  k=reshaped_x.size(1),
                                                  device=input.device,
                                                  dtype=input.dtype)

    output = ops.gptq_marlin_gemm(reshaped_x,
                                  weight,
                                  weight_scale,
                                  weight_zp,
                                  g_idx,
                                  g_idx_sort_indices,
                                  workspace,
                                  quant_type,
                                  size_m=reshaped_x.shape[0],
                                  size_n=output_size_per_partition,
                                  size_k=input_size_per_partition,
                                  is_k_full=True,
                                  has_zp=True,
                                  use_atomic_add=use_atomic_add,
                                  use_fp32_reduce=use_fp32_reduce,
                                  is_zp_float=False)

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)
