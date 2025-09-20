import math
import inspect
import itertools
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import time
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer

from ...utils import logging
from ...video_processor import VideoProcessor
from ...utils.torch_utils import randn_tensor
from ...models import AutoencoderKLCogVideoX, CogVideoXVividVRTransformer3DModel, CogVideoXVividVRControlNetModel
from ...schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from ...models.embeddings_vividvr import get_3d_rotary_pos_embed
from ..pipeline_utils import DiffusionPipeline
from .pipeline_cogvideox import (
    CogVideoXPipelineOutput, CogVideoXLoraLoaderMixin,
    retrieve_timesteps, get_resize_crop_region_for_grid
)
from .pipeline_cogvideox_image2video import retrieve_latents

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def prepare_rotary_positional_embeddings(
    latent_height: int,
    latent_width: int,
    num_frames: int,
    patch_size: int = 2,
    patch_size_t: Optional[int] = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    sample_height: int = 60,
    sample_width: int = 90,
) -> Tuple[torch.Tensor, torch.Tensor]:

    grid_height = latent_height // patch_size
    grid_width = latent_width // patch_size

    if patch_size_t is None:
        # CogVideoX 1.0 I2V
        base_size_width = sample_width // patch_size
        base_size_height = sample_height // patch_size
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        # CogVideoX 1.5 I2V
        # config from https://github.com/THUDM/CogVideo/blob/2fdc59c3ce48aee1ba7572a1c241e5b3090abffa/sat/configs/cogvideox1.5_5b_i2v.yaml#L33
        max_size_width = 300 // patch_size
        max_size_height = 300 // patch_size
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(max_size_height, max_size_width),
        )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def prepare_tiling_infos_generator(
    latents,
    enable_spatial_tiling=False,
    enable_temporal_tiling=False,
    tile_size=128, tile_stride=64,
    temporal_tile_size=31, temporal_tile_stride=15
):
    if not enable_spatial_tiling and not enable_temporal_tiling:
        yield [slice(None), torch.ones_like(latents)]
        return

    batch_size, num_frames, num_channels, height, width = latents.shape

    if not enable_spatial_tiling:
        tile_size = max(height, width)
    if not enable_temporal_tiling:
        temporal_tile_size = num_frames

    def create_start_indices(size, tile_size, tile_stride):
        if size <= tile_size:
            tile_stride = tile_size
        else:
            num_tiles = (size - tile_size) // tile_stride + 1
            if (size - tile_size) % tile_stride != 0:
                num_tiles += 1
            tile_stride = math.ceil((size - tile_size) / (num_tiles - 1))
        i_list = list(range(0, max(1, size - tile_size + 1), tile_stride))
        if size >= tile_size and (size - tile_size) % tile_stride != 0:
            i_list.append(size - tile_size)
        return i_list, tile_size, tile_stride

    ti_list, t_tile_size, t_tile_stride = create_start_indices(num_frames, temporal_tile_size, temporal_tile_stride)
    hi_list, h_tile_size, h_tile_stride = create_start_indices(height, tile_size, tile_stride)
    wi_list, w_tile_size, w_tile_stride = create_start_indices(width, tile_size, tile_stride)

    def compute_valid_weights_range(i, i_end, size, tile_size, tile_stride):
        float_padding = (tile_size - tile_stride) / 2
        end = tile_size - math.floor(float_padding) if i_end < size else tile_size
        start = math.ceil(float_padding) if i > 0 else 0
        remainder = i % tile_stride
        if remainder > 0:
            start = tile_size - (math.floor(float_padding) + remainder)
        return slice(start, end)

    for ti, hi, wi in itertools.product(ti_list, hi_list, wi_list):
        ti_end = min(ti + t_tile_size, num_frames)
        hi_end = min(hi + h_tile_size, height)
        wi_end = min(wi + w_tile_size, width)
        tile_slice = [slice(None), slice(ti, ti_end), slice(None), slice(hi, hi_end), slice(wi, wi_end)]

        t_valid_slice = compute_valid_weights_range(ti, ti_end, num_frames, t_tile_size, t_tile_stride)
        h_valid_slice = compute_valid_weights_range(hi, hi_end, height, h_tile_size, h_tile_stride)
        w_valid_slice = compute_valid_weights_range(wi, wi_end, width, w_tile_size, w_tile_stride)
        weights = torch.zeros((1, ti_end - ti, 1, hi_end - hi, wi_end - wi))
        weights[:, t_valid_slice, :, h_valid_slice, w_valid_slice] = 1

        yield tile_slice, weights.repeat(batch_size, 1, num_channels, 1, 1).to(latents.device)


class CogVideoXVividVRControlNetPipeline(DiffusionPipeline, CogVideoXLoraLoaderMixin):
    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]
    
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXVividVRTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
        controlnet: CogVideoXVividVRControlNetModel,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, controlnet=controlnet, scheduler=scheduler
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        control_video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        control_latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (control_video.size(2) - 1) // self.vae_scale_factor_temporal + 1 if control_latents is None else control_latents.size(1)

        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if control_latents is None:
            if isinstance(generator, list):
                if len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )

                control_latents = [
                    retrieve_latents(self.vae.encode(control_video[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
                ]
            else:
                control_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in control_video]

            control_latents = torch.cat(control_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            control_latents = self.vae.config.scaling_factor * control_latents
        else:
            # [TODO] force shape of control_latents
            control_latents = control_latents.to(device)

        # For CogVideoX1.5, the latent should pad to be divisible by patch_size_t
        if self.transformer.config.patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        num_latent_padding_frames = shape[1] - control_latents.size(1)
        if num_latent_padding_frames > 0:
            first_frame = control_latents[:, : control_latents.size(1) % self.transformer.config.patch_size_t, ...]
            control_latents = torch.cat([first_frame, control_latents], dim=1)

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        return latents, control_latents, num_latent_padding_frames

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.decode_latents
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents

        frames = self.vae.decode(latents).sample
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        control_video,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        control_latents,
        prompt_embeds,
        negative_prompt_embeds,
        enable_spatial_tiling,
    ):

        if control_video is not None and control_latents is not None:
            raise ValueError("Only one of `control_video` or `control_latents` should be provided")

        # [TODO] check video/latents height/width

        if (height % 8 != 0 or width % 8 != 0) and not enable_spatial_tiling:
            raise ValueError(f"`height` and `width` have to be divisible by 8 if tiling is disabled but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.fuse_qkv_projections
    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.unfuse_qkv_projections
    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        control_video: List[Image.Image] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        control_latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        enable_spatial_tiling: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        restoration_guidance_scale: float = -1.0,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        
        # 1. Prepare latents, prompt_embeds and some denoising arguments
        pre_denoise_return = self.pre_denoise_process(
            control_video=control_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            control_latents=control_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            enable_spatial_tiling=enable_spatial_tiling
        )
        latents = pre_denoise_return["latents"]
        control_latents = pre_denoise_return["control_latents"]
        prompt_embeds = pre_denoise_return["prompt_embeds"]
        negative_prompt_embeds = pre_denoise_return["negative_prompt_embeds"]
        do_classifier_free_guidance = pre_denoise_return["do_classifier_free_guidance"]
        num_latent_padding_frames = pre_denoise_return["num_latent_padding_frames"]
        height = pre_denoise_return["height"]
        width = pre_denoise_return["width"]
        ori_height = pre_denoise_return["ori_height"]
        ori_width = pre_denoise_return["ori_width"]

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        device = self._execution_device

        # 2. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 3. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 4. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # 5. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        old_pred_original_sample = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                print(f'Step {i}, Latent shape {latents.shape}')
                latents, old_pred_original_sample = self.denoise_process(
                    latents=latents,
                    old_pred_original_sample=old_pred_original_sample,
                    control_latents=control_latents,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    ofs_emb=ofs_emb,
                    timesteps=timesteps,
                    timestep_index=i,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    use_dynamic_cfg=use_dynamic_cfg,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    extra_step_kwargs=extra_step_kwargs,
                    attention_kwargs=attention_kwargs,
                    enable_spatial_tiling=enable_spatial_tiling,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                    restoration_guidance_scale=restoration_guidance_scale
                )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        # 6. VAE decoding
        video = self.post_denoise_process(
            latents=latents,
            num_latent_padding_frames=num_latent_padding_frames,
            ori_height=ori_height,
            ori_width=ori_width,
            output_type=output_type,
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)

    def denoise_step(self, timestep_index, latents, control_latents, prompt_embeds, negative_prompt_embeds, timesteps, ofs_emb, old_pred_original_sample,
                     use_dynamic_cfg, guidance_scale, num_inference_steps, do_classifier_free_guidance,
                     attention_kwargs, extra_step_kwargs, restoration_guidance_scale=-1.0):

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timesteps[timestep_index]
        timestep_expand = timestep.expand(latents.shape[0])

        last_time = time.time()
        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                latent_height=latents.shape[-2],
                latent_width=latents.shape[-1],
                num_frames=latents.shape[1],
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
                attention_head_dim=self.transformer.config.attention_head_dim,
                device=latents.device,
                sample_height=self.transformer.config.sample_height,
                sample_width=self.transformer.config.sample_width,
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        print(f'[Time] Prepare RoPE {time.time() - last_time:.2f} s')
        last_time = time.time()

        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

        control_model_input = torch.cat([control_latents] * 2) if do_classifier_free_guidance else control_latents

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)  if do_classifier_free_guidance else prompt_embeds

        concat_latent_model_input = torch.cat([latent_model_input, control_model_input], dim=2)

        control_hidden_states = self.controlnet(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            control_states=control_model_input,
            image_rotary_emb=image_rotary_emb,
            timestep=timestep_expand,
            ofs=ofs_emb,
            return_dict=False,
        )[0]
        control_hidden_states = [[x.to(prompt_embeds.dtype) for x in state] for state in control_hidden_states]
        print(f'[Time] Controlnet {time.time() - last_time:.2f} s')
        last_time = time.time()

        # predict noise model_output
        noise_pred = self.transformer(
            hidden_states=concat_latent_model_input,
            encoder_hidden_states=prompt_embeds,
            control_hidden_states=control_hidden_states,
            image_rotary_emb=image_rotary_emb,
            timestep=timestep_expand,
            ofs=ofs_emb,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.float()
        print(f'[Time] Transformer {time.time() - last_time:.2f} s')
        last_time = time.time()

        # perform guidance
        if use_dynamic_cfg:
            self._guidance_scale = 1 + guidance_scale * (
                (1 - math.cos(math.pi * ((num_inference_steps - timestep.item()) / num_inference_steps) ** 5.0)) / 2
            )
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        if not isinstance(self.scheduler, CogVideoXDPMScheduler):
            latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
        else:
            latents, old_pred_original_sample = self.scheduler.step(
                noise_pred,
                old_pred_original_sample if timestep_index > 0 else None,
                timestep,
                timesteps[timestep_index - 1] if timestep_index > 0 else None,
                latents,
                **extra_step_kwargs,
                return_dict=False,
                restoration_guidance_scale=restoration_guidance_scale,
                restoration_ori_latent=control_latents,
            )
        latents = latents.to(prompt_embeds.dtype)
        print(f'[Time] Scheduler {time.time() - last_time:.2f} s')
        last_time = time.time()

        return latents, old_pred_original_sample
    
    @torch.no_grad()
    def pre_denoise_process(
        self,
        control_video: List[Image.Image] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        control_latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        enable_spatial_tiling: bool = False
    ):

        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            control_video=control_video,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            control_latents=control_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            enable_spatial_tiling=enable_spatial_tiling,
        )

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if enable_spatial_tiling:
            # [TODO] multiple prompts supporting
            num_videos_per_prompt = 1
            batch_size = 1

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # 4. Prepare noisy latents and control latents
        if control_latents is None:
            # [B, T, C, H, W] -> [B, C, T, H, W]
            ori_height, ori_width = control_video.shape[-2:]
            control_video = self.video_processor.preprocess_video(control_video, height=height, width=width)
            control_video = control_video.to(device=device, dtype=prompt_embeds.dtype)

        # [TODO] check here
        latent_channels = 16  # self.transformer.config.in_channels
        latents, control_latents, num_latent_padding_frames = self.prepare_latents(
            control_video=control_video,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=latent_channels,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            control_latents=control_latents,
        )

        return {
            "latents": latents,
            "control_latents": control_latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "num_latent_padding_frames": num_latent_padding_frames,
            "height": height,
            "width": width,
            "ori_height": ori_height,
            "ori_width": ori_width
        }

    @torch.no_grad()
    def denoise_process(
        self,
        latents,
        old_pred_original_sample,
        control_latents,
        prompt_embeds,
        negative_prompt_embeds,
        ofs_emb,
        timesteps,
        timestep_index,
        num_inference_steps: int = 50,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        do_classifier_free_guidance: bool = False,
        extra_step_kwargs: Optional[Dict[str, Any]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        enable_spatial_tiling: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        restoration_guidance_scale: float = -1.0,
    ):
        tiling_infos = list(prepare_tiling_infos_generator(
            latents=latents,
            enable_spatial_tiling=enable_spatial_tiling,
            enable_temporal_tiling=False,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ))

        latents_meshgrid = torch.zeros_like(latents)
        old_pred_original_sample_meshgrid = torch.zeros_like(latents)
        weights_meshgrid = torch.zeros_like(latents)

        for tile_index, (tile_slice, tile_weights) in enumerate(tiling_infos):
            print(f'TileIndex {tile_index}, Slice {tile_slice}')
            last_time = time.time()
            prompt_slice = slice(tile_index, tile_index + 1)
            
            tile_latents, tile_old_pred_original_sample = self.denoise_step(
                timestep_index=timestep_index,
                latents=latents[tile_slice],
                control_latents=control_latents[tile_slice],
                prompt_embeds=prompt_embeds[prompt_slice],
                negative_prompt_embeds=negative_prompt_embeds[prompt_slice],
                timesteps=timesteps,
                ofs_emb=ofs_emb,
                old_pred_original_sample=old_pred_original_sample[tile_slice] if old_pred_original_sample is not None else None,
                use_dynamic_cfg=use_dynamic_cfg,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                do_classifier_free_guidance=do_classifier_free_guidance,
                attention_kwargs=attention_kwargs,
                extra_step_kwargs=extra_step_kwargs,
                restoration_guidance_scale=restoration_guidance_scale
            )

            latents_meshgrid[tile_slice] += tile_latents * tile_weights
            old_pred_original_sample_meshgrid[tile_slice] += tile_old_pred_original_sample * tile_weights
            weights_meshgrid[tile_slice] += tile_weights
            print(f"TileProcessDuration: {time.time() - last_time}")

        latents = latents_meshgrid / weights_meshgrid
        old_pred_original_sample = old_pred_original_sample_meshgrid / weights_meshgrid
        
        return latents, old_pred_original_sample
    
    @torch.no_grad()
    def post_denoise_process(
        self,
        latents: torch.Tensor,
        num_latent_padding_frames: int,
        ori_height: int,
        ori_width: int,
        output_type: str = "pil",
    ):
        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            if num_latent_padding_frames > 0:
                latents = latents[:, num_latent_padding_frames:]
            video = self.decode_latents(latents)
            video = [F.interpolate(i.permute(1, 0, 2, 3), size=(ori_height, ori_width), mode='bilinear') for i in video]
            video = torch.stack(video, dim=0).permute(0, 2, 1, 3, 4)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents
        return video
