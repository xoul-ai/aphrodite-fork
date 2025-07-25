from typing import Iterable, Set, Tuple

import torch
import torch.nn as nn
from loguru import logger
from transformers import LlamaConfig

from aphrodite.common.config import AphroditeConfig
from aphrodite.compilation.decorators import support_torch_compile
from aphrodite.modeling.layers.logits_processor import LogitsProcessor
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from aphrodite.modeling.model_loader.weight_utils import default_weight_loader
from aphrodite.modeling.models.llama import LlamaDecoderLayer, LlamaForCausalLM

from .utils import AutoWeightsLoader, maybe_prefix


class LlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(
        self,
        config: LlamaConfig,
        disable_input_layernorm: bool,
        prefix: str = "",
    ) -> None:
        super().__init__(config, prefix=prefix)

        # Skip the input_layernorm
        # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427
        if disable_input_layernorm:
            del self.input_layernorm
            self.input_layernorm = nn.Identity()


@support_torch_compile
class LlamaModel(nn.Module):

    def __init__(
        self,
        *,
        aphrodite_config: AphroditeConfig,
        prefix: str = "",
        start_layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = aphrodite_config. \
            speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                self.config,
                i == 0,
                prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
            ) for i in range(self.config.num_hidden_layers)
        ])
        self.fc = torch.nn.Linear(self.config.hidden_size * 2,
                                  self.config.hidden_size,
                                  bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(
            torch.cat((input_embeds, hidden_states), dim=-1))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        hidden_states = hidden_states + residual
        return hidden_states, hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class EagleLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, *, aphrodite_config: AphroditeConfig, start_layer_id: int = 0):
        nn.Module.__init__(self)
        self.config = aphrodite_config. \
            speculative_config.draft_model_config.hf_config
        self.model = LlamaModel(aphrodite_config=aphrodite_config,
                                prefix="model",
                                start_layer_id=start_layer_id)

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                scale=logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )

        model_weights = {}
        for name, loaded_weight in weights:
            if "lm_head" not in name:
                name = "model." + name
            model_weights[name] = loaded_weight

        loader.load_weights(model_weights.items())
