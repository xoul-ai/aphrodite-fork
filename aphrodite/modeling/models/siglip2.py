# SPDX-License-Identifier: Apache-2.0
"""Implementation of Siglip2VisionModel intended to be only used
within a vision language model."""

import math
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from PIL import Image
from torch import nn
from transformers import Siglip2VisionConfig

from aphrodite.attention.layer import MultiHeadAttention
from aphrodite.distributed import divide, get_tensor_model_parallel_world_size
from aphrodite.modeling.layers.activation import get_act_fn
from aphrodite.modeling.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from aphrodite.quantization import QuantizationConfig
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from aphrodite.modeling.model_loader.weight_utils import default_weight_loader
from aphrodite.multimodal.utils import consecutive_placeholder_ranges
from aphrodite.common.sequence import SequenceData

from .vision import VisionEncoderInfo, resolve_visual_encoder_outputs


def get_siglip2_patch_grid_length(*, image_size: int, patch_size: int) -> int:
    # Since interpolation is applied, the image size need not be divisible
    # assert image_size % patch_size == 0
    return image_size // patch_size


def get_siglip2_num_patches(*, image_size: int, patch_size: int) -> int:
    grid_length = get_siglip2_patch_grid_length(image_size=image_size,
                                                patch_size=patch_size)
    return grid_length * grid_length


def get_siglip2_image_feature_size(hf_config: Siglip2VisionConfig) -> int:
    return get_siglip2_num_patches(image_size=hf_config.image_size,
                                   patch_size=hf_config.patch_size)


def get_max_siglip2_image_tokens(hf_config: Siglip2VisionConfig) -> int:
    return get_siglip2_image_feature_size(hf_config)


def dummy_seq_data_for_siglip2(
    hf_config: Siglip2VisionConfig,
    seq_len: int,
    num_images: int,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
    mm_key: str = "image",
):
    if image_feature_size_override is None:
        image_feature_size = get_siglip2_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    return SequenceData.from_prompt_token_counts(
        (image_token_id, image_feature_size * num_images),
        (0, seq_len - image_feature_size * num_images),
    ), {
        mm_key:
        consecutive_placeholder_ranges(num_items=num_images,
                                       item_size=image_feature_size)
    }


def dummy_image_for_siglip2(
    hf_config: Siglip2VisionConfig,
    num_images: int,
    *,
    image_width_override: Optional[int] = None,
    image_height_override: Optional[int] = None,
):
    width = height = hf_config.image_size
    if image_width_override is not None:
        width = image_width_override
    if image_height_override is not None:
        height = image_height_override

    image = Image.new("RGB", (width, height), color=0)
    return {"image": image if num_images == 1 else [image] * num_images}


class Siglip2EncoderInfo(VisionEncoderInfo[Siglip2VisionConfig]):

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        return get_siglip2_image_feature_size(self.vision_config)

    def get_max_image_tokens(self) -> int:
        return get_max_siglip2_image_tokens(self.vision_config)

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        return self.vision_config.patch_size

    def get_patch_grid_length(self) -> int:
        return get_siglip2_patch_grid_length(
            image_size=self.vision_config.image_size,
            patch_size=self.vision_config.patch_size,
        )


# Adapted from https://github.com/huggingface/transformers/blob/v4.49.0-SigLIP-2/src/transformers/models/siglip2/modeling_siglip2.py
class Siglip2VisionEmbeddings(nn.Module):

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = VocabParallelEmbedding(
            self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions, dtype=torch.int64).expand(
                (1, -1)),
            persistent=False,
        )

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int,
                                 width: int) -> torch.Tensor:
        """
        This method is an adapted method for Siglip2 (due to Siglip2 not having
        class embedding unlike other ViTs) that allows the model to interpolate
        the pre-trained position encodings such that it can be usable on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        position_embeddings = self.position_embedding.weight.unsqueeze(0)
        num_patches = embeddings.shape[1]
        num_positions = position_embeddings.shape[1]
        if num_patches == num_positions and height == width:
            return position_embeddings

        dim = embeddings.shape[-1]
        height = height // self.patch_size
        width = width // self.patch_size
        # we add a small number to avoid floating point error
        # in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1

        patch_pos_embed = position_embeddings.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)),
            dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(
                height / math.sqrt(num_positions),
                width / math.sqrt(num_positions),
            ),
            mode="bicubic",
            align_corners=False,
        )
        if (int(height) != patch_pos_embed.shape[-2]
                or int(width) != patch_pos_embed.shape[-1]):
            raise ValueError("Width or height does not match with "
                             "the interpolated position embeddings")

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self,
                pixel_values: torch.Tensor,
                interpolate_pos_encoding: bool = False) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(
                self.position_ids)
        return embeddings


class Siglip2Attention(nn.Module):

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got "
                             "`embed_dim`: {self.embed_dim} and `num_heads`:"
                             f" {self.num_heads}).")

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

        self.attn = MultiHeadAttention(self.num_heads_per_partition,
                                       self.head_dim, self.scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)

        out = self.attn(query_states, key_states, value_states)
        attn_output, _ = self.out_proj(out)

        return attn_output, None


class Siglip2MLP(nn.Module):

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        # Special handling for BNB quantization
        if quant_config and quant_config.get_name() == "bitsandbytes":
            quantizable = True
        else:
            # For other quantization, we require the hidden size to be a
            # multiple of 64
            quantizable = (config.hidden_size % 64 == 0
                           and config.intermediate_size % 64 == 0)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config if quantizable else None,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config if quantizable else None,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.embed_dim = config.hidden_size

        self.self_attn = Siglip2Attention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


class Siglip2Encoder(nn.Module):

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList([
            Siglip2EncoderLayer(config,
                                quant_config=quant_config,
                                prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        return_all_hidden_states: bool,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states, _ = encoder_layer(hidden_states)
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        # If we have multiple feature sample layers, we return all hidden
        # states in order and grab the ones we need by index.
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # TODO: Implement Aphrodite version of MultiheadAttention
        self.attention = torch.nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config=config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.mlp")

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class Siglip2VisionTransformer(nn.Module):

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)

        self.encoder = Siglip2Encoder(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        # If possible, skip post_layernorm to conserve memory
        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim,
                                               eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

        self.use_head = (True if not hasattr(config, "vision_use_head") else
                         config.vision_use_head)
        if self.use_head:
            self.head = Siglip2MultiheadAttentionPoolingHead(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.head",
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = True,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:

        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        return_all_hidden_states = feature_sample_layers is not None

        # Produces either the last layer output or all of the hidden states,
        # depending on if we have feature_sample_layers or not
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=return_all_hidden_states,
        )

        # Handle post-norm (if applicable) and stacks feature layers if needed
        encoder_outputs = resolve_visual_encoder_outputs(
            encoder_outputs, feature_sample_layers, self.post_layernorm,
            self.config.num_hidden_layers)

        # TODO: add this back when pooled_output is used in inference.
        # if self.use_head:
        # pooled_output = self.head(encoder_outputs)

        return encoder_outputs


class Siglip2VisionModel(nn.Module):
    config_class = Siglip2VisionConfig
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.vision_model = Siglip2VisionTransformer(
            config,
            quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
            prefix=f"{prefix}.vision_model",
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:
        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            feature_sample_layers=feature_sample_layers,
        )

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is optional in Siglip2VisionModel
            if (name.startswith("vision_model.post_layernorm")
                    and self.vision_model.post_layernorm is None):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("vision_model.encoder.layers"):
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
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
