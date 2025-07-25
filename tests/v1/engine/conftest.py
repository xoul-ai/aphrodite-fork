import pytest
import torch
from transformers import AutoTokenizer

from aphrodite.engine.args_tools import EngineArgs
from aphrodite.transformers_utils.tokenizer_group import (
    init_tokenizer_from_configs)
from tests.v1.engine.utils import (NUM_PROMPT_LOGPROBS_UNDER_TEST,
                                   NUM_SAMPLE_LOGPROBS_UNDER_TEST, PROMPT_LEN,
                                   TOKENIZER_NAME,
                                   DummyOutputProcessorTestVectors,
                                   generate_dummy_prompt_logprobs_tensors,
                                   generate_dummy_sample_logprobs)

from ...distributed.conftest import publisher_config, random_port  # noqa: F401

from tests.v1.engine.utils import FULL_STRINGS  # isort: skip

EngineCoreSampleLogprobsType = list[tuple[torch.Tensor, torch.Tensor]]
EngineCorePromptLogprobsType = tuple[torch.Tensor, torch.Tensor]


def _build_test_vectors_no_logprobs() -> DummyOutputProcessorTestVectors:
    """Generate output processor dummy test vectors, without logprobs
    
    Returns:
      DummyOutputProcessorTestVectors instance with no logprobs
    """

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    aphrodite_config = EngineArgs(model=TOKENIZER_NAME).create_engine_config()
    # Tokenize prompts under test & create dummy generated tokens
    prompt_tokens = [
        tokenizer(text).input_ids[:PROMPT_LEN] for text in FULL_STRINGS
    ]
    generation_tokens = [
        tokenizer(text).input_ids[PROMPT_LEN:] for text in FULL_STRINGS
    ]
    # Generate prompt strings
    prompt_strings = [
        tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        for prompt_tokens in prompt_tokens
    ]
    prompt_strings_len = [
        len(prompt_string) for prompt_string in prompt_strings
    ]
    return DummyOutputProcessorTestVectors(
        tokenizer=tokenizer,
        tokenizer_group=init_tokenizer_from_configs(
            aphrodite_config.model_config, aphrodite_config.scheduler_config,
            aphrodite_config.lora_config),
        aphrodite_config=aphrodite_config,
        full_tokens=[tokenizer(text).input_ids for text in FULL_STRINGS],
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        prompt_strings=prompt_strings,
        prompt_strings_len=prompt_strings_len,
        generation_strings=[
            text[prompt_len:]
            for text, prompt_len in zip(FULL_STRINGS, prompt_strings_len)
        ],
        prompt_logprobs=[],
        generation_logprobs=[])


@pytest.fixture
def dummy_test_vectors() -> DummyOutputProcessorTestVectors:
    """Generate output processor dummy test vectors, with logprobs
    
    Returns:
      DummyOutputProcessorTestVectors instance with logprobs
    """
    # Build dummy test vectors without logprobs
    dtv = _build_test_vectors_no_logprobs()
    # Inject logprobs into dummy test vectors
    # data structure
    dtv.generation_logprobs = [
        generate_dummy_sample_logprobs(
            sampled_tokens_list=tokens_list,
            num_logprobs=NUM_SAMPLE_LOGPROBS_UNDER_TEST,
            tokenizer=dtv.tokenizer) for tokens_list in dtv.generation_tokens
    ]
    dtv.prompt_logprobs = [
        generate_dummy_prompt_logprobs_tensors(
            prompt_tokens_list=tokens_list,
            num_logprobs=NUM_PROMPT_LOGPROBS_UNDER_TEST,
            tokenizer=dtv.tokenizer) for tokens_list in dtv.prompt_tokens
    ]
    return dtv
