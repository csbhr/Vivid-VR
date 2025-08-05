import os
import gc
import math
import itertools
import decord
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple


VALID_IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'png', 'webp', 'bmp')
VALID_VIDEO_EXTENSIONS = ('mp4', 'mov', 'avi', 'webm', 'mkv', 'y4m')


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif is_torch_npu_available():
        torch_npu.empty_cache()


def load_video(video_path):

    def load_image(path):
        img = np.array(Image.open(path), dtype=np.float32)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    decord.bridge.set_bridge("torch")

    if os.path.isdir(video_path):
        fnames = [fn for fn in sorted(os.listdir(video_path)) if fn.split('.')[-1].lower() in VALID_IMAGE_EXTENSIONS]
        fpaths = [os.path.join(video_path, fn) for fn in fnames]
        frames = torch.stack([torch.from_numpy(load_image(fp)) for fp in fpaths])
    else:
        video_reader = decord.VideoReader(uri=video_path, num_threads=1)
        indices = list(range(0, len(video_reader)))
        frames = video_reader.get_batch(indices)

    frames = frames.float().div(255.).clip(0, 1).permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]
    return frames


def export_to_video(video_frames, output_video_path, fps=24):
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    with imageio.get_writer(output_video_path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in video_frames:
            writer.append_data(frame)


def find_nearest_res_bucket(
    resolutions: List[Tuple[int, int, int]], # (frames, height, width)
    height: int,
    width: int
):
    nearest_res = min(resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
    return nearest_res[1], nearest_res[2]


def prepare_tiling_infos_generator(
    latents, 
    enable_spatial_tiling=False, 
    enable_temporal_tiling=False, 
    tile_size=1024, tile_stride=512, 
    temporal_tile_size=121, temporal_tile_stride=60
):
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

    tile_slice_list = []
    for ti, hi, wi in itertools.product(ti_list, hi_list, wi_list):
        ti_end = min(ti + t_tile_size, num_frames)
        hi_end = min(hi + h_tile_size, height)
        wi_end = min(wi + w_tile_size, width)
        tile_slice = [slice(None), slice(ti, ti_end), slice(None), slice(hi, hi_end), slice(wi, wi_end)]
        tile_slice_list.append(tile_slice)

    return tile_slice_list


def prepare_validation_prompts(video_for_caption, video_fps, captioner_model, tile_size, tile_stride, device='cuda'):
    POS_PROMPT = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
    NEG_PROMPT = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'

    tiling_infos = list(prepare_tiling_infos_generator(
        latents=video_for_caption.unsqueeze(0), # [B, T, C, H, W]
        enable_spatial_tiling=True,
        enable_temporal_tiling=False,
        tile_size=tile_size,
        tile_stride=tile_stride,
    ))

    print(f"Captioning video with {len(tiling_infos)} tiles")
    captioner_model.to(device)
    prompts = []
    for tile_slice in tiling_infos:
        tile_video = video_for_caption[tile_slice[1:]]
        response = captioner_model(tile_video, fps=video_fps)
        prompts.append(response)
    captioner_model.to(torch.device('cpu'))

    prompt_list = [f"{prompt} {POS_PROMPT}" for prompt in prompts]
    negative_prompt_list = [NEG_PROMPT for _ in range(len(prompts))]
    print(f"Generating 1 video with prompt: {prompt_list}, and negative prompt: {negative_prompt_list}")
    return prompt_list, negative_prompt_list
