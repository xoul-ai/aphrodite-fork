"""A basic correctness check for TPUs

Run `pytest tests/v1/tpu/test_basic.py`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from aphrodite.platforms import current_platform

if TYPE_CHECKING:
    from tests.conftest import AphroditeRunner

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    # TODO: Enable this models with v6e
    # "Qwen/Qwen2-7B-Instruct",
    # "meta-llama/Llama-3.1-8B",
]

TENSOR_PARALLEL_SIZES = [1]
MAX_NUM_REQS = [16, 1024]

# TODO: Enable when CI/CD will have a multi-tpu instance
# TENSOR_PARALLEL_SIZES = [1, 4]


@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This is a basic test for TPU only")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("tensor_parallel_size", TENSOR_PARALLEL_SIZES)
@pytest.mark.parametrize("max_num_seqs", MAX_NUM_REQS)
def test_basic(
    aphrodite_runner: type[AphroditeRunner],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    max_tokens: int,
    tensor_parallel_size: int,
    max_num_seqs: int,
) -> None:
    prompt = "The next numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with monkeypatch.context() as m:
        m.setenv("APHRODITE_USE_V1", "1")

        with aphrodite_runner(
                model,
                # Note: max_num_batched_tokens == 1024 is needed here to
                # actually test chunked prompt
                max_num_batched_tokens=1024,
                max_model_len=8192,
                gpu_memory_utilization=0.7,
                max_num_seqs=max_num_seqs,
                tensor_parallel_size=tensor_parallel_size) as aphrodite_model:
            aphrodite_outputs = aphrodite_model.generate_greedy(
                example_prompts, max_tokens)
        output = aphrodite_outputs[0][1]

        assert "1024" in output or "0, 1" in output
