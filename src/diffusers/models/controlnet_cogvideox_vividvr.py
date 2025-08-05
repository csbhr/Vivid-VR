from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from ..utils import (
    logging,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from ..loaders import PeftAdapterMixin
from .modeling_utils import ModelMixin
from ..configuration_utils import ConfigMixin, register_to_config
from .embeddings_vividvr import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from .transformers.cogvideox_vividvr_transformer_3d import (
    CogVideoXBlock,
    CogVideoXVividVRTransformer3DModel,
    Transformer2DModelOutput,
)
from .resnet import SpatioTemporalResBlock
from einops import rearrange
import math

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class CogVideoXVividVRControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs

        # 2. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Define controlnet patchify layers and connectors
        self.time_embed_dim = time_embed_dim
        self.control_feat_proj = nn.ModuleList([
            SpatioTemporalResBlock(in_channels, 320, time_embed_dim, merge_strategy="learned", groups=16),
            SpatioTemporalResBlock(320, 320, time_embed_dim, merge_strategy="learned", groups=32),
            SpatioTemporalResBlock(320, in_channels, time_embed_dim, merge_strategy="learned", groups=16)
        ])
        self.control_patch_embed = zero_module(
            CogVideoXPatchEmbed(
                patch_size=patch_size,
                patch_size_t=patch_size_t,
                in_channels=in_channels,
                embed_dim=inner_dim,
                text_embed_dim=text_embed_dim,
                bias=patch_bias,
                sample_width=sample_width,
                sample_height=sample_height,
                sample_frames=sample_frames,
                temporal_compression_ratio=temporal_compression_ratio,
                max_text_seq_length=max_text_seq_length,
                spatial_interpolation_scale=spatial_interpolation_scale,
                temporal_interpolation_scale=temporal_interpolation_scale,
                use_positional_embeddings=False,
                use_learned_positional_embeddings=False,
            )
        )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        conditioning_scale: float = 1.0,
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        # 2.1 Controlnet patch embedding
        B, F, C, H, W = control_states.shape
        control_states = rearrange(control_states, "B F C H W -> (B F) C H W")
        res_emb = emb.repeat(B * F, 1)
        for module in self.control_feat_proj:
            control_states = module(control_states,res_emb,torch.ones((F),device=control_states.device))
        control_states = rearrange(control_states, "(B F) C H W -> B F C H W", B=B, F=F)

        control_states = self.control_patch_embed(encoder_hidden_states, control_states)
        hidden_states = hidden_states + control_states

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        controlnet_inter_states = ()
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
            )
            controlnet_inter_states = controlnet_inter_states + (hidden_states,)

        controlnet_hidden_states = ()
        for state in controlnet_inter_states:
            controlnet_hidden_states = controlnet_hidden_states + ([state],)
        
        # 4. Scaling
        controlnet_hidden_states = [[x * conditioning_scale for x in state] for state in controlnet_hidden_states]

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_hidden_states,)
        return Transformer2DModelOutput(sample=controlnet_hidden_states)

    @classmethod
    def from_transformer(
        cls,
        transformer: CogVideoXVividVRTransformer3DModel,
        num_layers: int = 6,
        load_weights_from_transformer: bool = True,
        load_transformer_weights_interval: bool = False,
    ):
        config = transformer.config.copy()
        config["num_layers"] = num_layers or config.num_layers
        config["use_learned_positional_embeddings"] = False
        config["use_rotary_positional_embeddings"] = True
        controlnet = cls(**config)

        if load_weights_from_transformer:
            controlnet.patch_embed.load_state_dict(transformer.patch_embed.state_dict())
            controlnet.time_embedding.load_state_dict(transformer.time_embedding.state_dict())
            if load_transformer_weights_interval:
                control_interval = math.ceil(transformer.config.num_layers / config["num_layers"])
                logger.info(f"Loading weights from transformer with interval {control_interval}")
                for i in range(num_layers):
                    controlnet.transformer_blocks[i].load_state_dict(transformer.transformer_blocks[control_interval*i].state_dict(), strict=False)                    
            else:
                controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)

        return controlnet