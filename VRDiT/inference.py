
import os
import gc
import sys
import cv2
import math
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import T5EncoderModel
from diffusers import CogVideoXDPMScheduler, AutoencoderKLCogVideoX, CogVideoXVividVRTransformer3DModel, CogVideoXVividVRControlNetModel, CogVideoXVividVRControlNetPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox_vividvr import retrieve_timesteps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../')))

from VRDiT.cogvlm2 import CogVLM2_Captioner
from VRDiT.colorfix import adaptive_instance_normalization
from VRDiT.utils import VALID_IMAGE_EXTENSIONS, VALID_VIDEO_EXTENSIONS, free_memory, load_video, export_to_video, prepare_validation_prompts


def infer_whole_video(
        args,
        captioner_model,
        pipe,
        info,
        control_video,
        gen_height,
        gen_width,
        vae_scale_factor_spatial
    ):
    # Padding to make the number of frames pad_type
    num_padding_frames = 0
    if (control_video.size(0) - 1) % 8 != 0:
        num_padding_frames = 8 - (control_video.size(0) - 1) % 8
        control_video = torch.cat([control_video, control_video[-1:].repeat(num_padding_frames, 1, 1, 1)], dim=0)

    pipeline_args = {
        "control_video": control_video,
        "guidance_scale": args.guidance_scale,
        "use_dynamic_cfg": args.use_dynamic_cfg,
        "height": gen_height,
        "width": gen_width,
        "num_inference_steps": args.num_inference_steps,
        "enable_spatial_tiling": True,
        "tile_size": args.tile_size,
        "tile_stride": args.tile_size // 2,
    }

    # prepare prompt
    video_for_caption = F.interpolate(control_video, size=(gen_height, gen_width), mode='bicubic')
    prompt_list, negative_prompt_list = prepare_validation_prompts(
        video_for_caption=video_for_caption,
        video_fps=info['fps'],
        captioner_model=captioner_model,
        tile_size=args.tile_size * vae_scale_factor_spatial,
        tile_stride=(args.tile_size // 2) * vae_scale_factor_spatial,
        device=args.device
    )
    pipeline_args['prompt'] = prompt_list
    pipeline_args['negative_prompt'] = negative_prompt_list

    # run inference
    video = pipe(
        **pipeline_args,
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
        output_type="np"  # numpy [T, H, W, C]
    ).frames[0]

    if video.shape[0] % 4 == 0:
        video = video[3:]
    if num_padding_frames > 0:
        video = video[:-num_padding_frames]

    return video


def infer_split_clips(
        args,
        captioner_model,
        pipe,
        info,
        control_video,
        gen_height,
        gen_width,
        vae_scale_factor_spatial,
        vae_scale_factor_temporal
    ):
    assert args.num_temporal_process_frames % 8 == 1, "the num_temporal_process_frames should match 8k+1"

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    num_temporal_process_frames = args.num_temporal_process_frames
    num_temporal_overlapped_frames = (num_temporal_process_frames - 1) // 2 + 1
    temporal_frame_stride = num_temporal_process_frames - num_temporal_overlapped_frames

    num_temporal_ov_latents = (num_temporal_overlapped_frames - 1) // vae_scale_factor_temporal
    temporal_latent_stride = ((num_temporal_process_frames - 1) // vae_scale_factor_temporal) - num_temporal_ov_latents

    num_frames = control_video.size(0)
    num_clips = (num_frames - num_temporal_process_frames) // temporal_frame_stride + 1
    if (num_clips - 1) * temporal_frame_stride + num_temporal_process_frames < num_frames:
        num_clips += 1
    num_clips = max(1, num_clips)

    # 1. Cache clips' latents, prompt_embeds and some denoising arguments
    clips_info_cache = {}
    for idx in range(num_clips):
        idx_begin = idx * temporal_frame_stride
        idx_end = min(idx_begin + num_temporal_process_frames, num_frames)
        clip_control_video = control_video[idx_begin:idx_end]

        # Padding to make the number of frames pad_type
        num_padding_frames = 0
        if (clip_control_video.size(0) - 1) % 8 != 0:
            num_padding_frames = 8 - (clip_control_video.size(0) - 1) % 8
            clip_control_video = torch.cat([clip_control_video, clip_control_video[-1:].repeat(num_padding_frames, 1, 1, 1)], dim=0)

        # prepare prompt
        video_for_caption = F.interpolate(clip_control_video, size=(gen_height, gen_width), mode='bicubic')
        prompt_list, negative_prompt_list = prepare_validation_prompts(
            video_for_caption=video_for_caption,
            video_fps=info['fps'],
            captioner_model=captioner_model,
            tile_size=args.tile_size * vae_scale_factor_spatial,
            tile_stride=(args.tile_size // 2) * vae_scale_factor_spatial,
            device=args.device
        )

        # Prepare latents, prompt_embeds and some denoising arguments
        clips_info_cache[idx] = pipe.pre_denoise_process(
            control_video=clip_control_video,
            prompt=prompt_list,
            negative_prompt=negative_prompt_list,
            height=gen_height,
            width=gen_width,
            guidance_scale=args.guidance_scale,
            generator=generator,
            enable_spatial_tiling=True
        )
        clips_info_cache[idx]['num_padding_frames'] = num_padding_frames
        clips_info_cache[idx]['old_pred_original_sample'] = None
    
    pipe._guidance_scale = args.guidance_scale
    pipe._attention_kwargs = None
    pipe._interrupt = False
    device = pipe._execution_device

    # 2. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, args.num_inference_steps, device, None)
    pipe._num_timesteps = len(timesteps)

    # 3. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)

    # 4. Create ofs embeds if required
    ofs_emb = None if pipe.transformer.config.ofs_embed_dim is None else clips_info_cache[0]['latents'].new_full((1,), fill_value=2.0)

    # 5. Prepare temporal latent merging info
    non_fstfr_latents_start_idx = 2  # for 8k+1 padding mode, the first 2 latents represent only the first frame, they are excluded when merging
    clip_id_to_latent_id_map = {}
    valid_latent_id_to_clip_id_map = {}
    for idx in range(num_clips):
        clip_info = clips_info_cache[idx]
        temporal_latent_length = clip_info['latents'].size(1) # get temporal length
        latent_idx_begin = temporal_latent_stride * idx + 1
        latent_idx_end = latent_idx_begin + (temporal_latent_length - non_fstfr_latents_start_idx)
        clip_id_to_latent_id_map[idx] = (latent_idx_begin, latent_idx_end)

        num_valid_latents = (latent_idx_end - latent_idx_begin) - num_temporal_ov_latents
        if idx == 0 or idx == num_clips - 1:
            num_valid_latents = (latent_idx_end - latent_idx_begin) - (num_temporal_ov_latents // 2)
        valid_latend_idx_begin = latent_idx_begin + num_temporal_ov_latents // 2 * int(idx > 0)
        valid_latend_idx_end = valid_latend_idx_begin + num_valid_latents
        for latent_id in range(valid_latend_idx_begin, valid_latend_idx_end):
            valid_latent_id_to_clip_id_map[latent_id] = idx
    print(f"Clip Id - Latent Id Map: {clip_id_to_latent_id_map}")
    print(f"Valid Latent Id - Clip Id Map: {valid_latent_id_to_clip_id_map}")

    # 6. Denoising loop
    for i, t in tqdm(enumerate(timesteps), desc=f"Denoising loop .."):
        if pipe.interrupt:
            continue

        # denose clip latents
        for idx in tqdm(range(num_clips), desc=f"Denoising temporal clips .."):
            clip_info = clips_info_cache[idx]

            latents, old_pred_original_sample = pipe.denoise_process(
                latents=clip_info["latents"],
                old_pred_original_sample=clip_info["old_pred_original_sample"],
                control_latents=clip_info["control_latents"],
                prompt_embeds=clip_info["prompt_embeds"],
                negative_prompt_embeds=clip_info["negative_prompt_embeds"],
                ofs_emb=ofs_emb,
                timesteps=timesteps,
                timestep_index=i,
                num_inference_steps=num_inference_steps,
                guidance_scale=args.guidance_scale,
                use_dynamic_cfg=args.use_dynamic_cfg,
                do_classifier_free_guidance=clip_info["do_classifier_free_guidance"],
                extra_step_kwargs=extra_step_kwargs,
                attention_kwargs=None,
                enable_spatial_tiling=True,
                tile_size=args.tile_size,
                tile_stride=args.tile_size // 2
            )

            clip_info["latents"] = latents
            clip_info["old_pred_original_sample"] = old_pred_original_sample
            clips_info_cache[idx] = clip_info

        # merge clip latents
        for idx in tqdm(range(num_clips), desc=f"Merging temporal clips .."):
            clip_info = clips_info_cache[idx]

            # [B, T, C, H, W]
            latents = clip_info["latents"]
            old_pred_original_sample = clip_info["old_pred_original_sample"]

            latent_id_range = clip_id_to_latent_id_map[idx]
            latent_id_offset = latent_id_range[0] - non_fstfr_latents_start_idx
            for latent_id in range(*latent_id_range):
                target_clip_idx = valid_latent_id_to_clip_id_map[latent_id]
                if target_clip_idx == idx: continue

                target_clip_info = clips_info_cache[target_clip_idx]
                target_clip_latent_id_offset = clip_id_to_latent_id_map[target_clip_idx][0] - non_fstfr_latents_start_idx

                latents[:, latent_id - latent_id_offset, ...] = \
                    target_clip_info['latents'][:, latent_id - target_clip_latent_id_offset, ...]
                old_pred_original_sample[:, latent_id - latent_id_offset, ...] = \
                    target_clip_info['old_pred_original_sample'][:, latent_id - target_clip_latent_id_offset, ...]
            
            clip_info["latents"] = latents
            clip_info["old_pred_original_sample"] = old_pred_original_sample
            clips_info_cache[idx] = clip_info
    
    # 7. VAE decoding
    video_buffer = []
    for idx in tqdm(range(num_clips), desc=f"VAE decoding temporal clips .."):
        clip_info = clips_info_cache[idx]
        video = pipe.post_denoise_process(
            latents=clip_info["latents"],
            num_latent_padding_frames=clip_info["num_latent_padding_frames"],
            ori_height=clip_info["ori_height"],
            ori_width=clip_info["ori_width"],
            output_type="np",
        )[0]
        if video.shape[0] % 4 == 0:
            video = video[3:]
        if clip_info["num_padding_frames"] > 0:
            video = video[:-clip_info["num_padding_frames"]]
        
        if idx > 0:
            video = video[(num_temporal_overlapped_frames + 1) // 2:]
        if idx < num_clips - 1:
            video = video[:-(num_temporal_overlapped_frames // 2)]
        video_buffer.append(video)
    video = np.concatenate(video_buffer, axis=0)
    
    return video


def main():
    parser = argparse.ArgumentParser(description="Vivid-VR")
    parser.add_argument("--ckpt_dir", type=str, default='./ckpts', help="Path to checkpoints directory.")
    parser.add_argument("--cogvideox_ckpt_path", type=str, default='./ckpts/CogVideoX1.5-5B', help="Path to CogVideoX1.5-5B checkpoints directory. Download from https://huggingface.co/zai-org/CogVideoX1.5-5B")
    parser.add_argument("--cogvlm2_ckpt_path", type=str, default='./ckpts/cogvlm2-llama3-caption', help="Path to CogVLM2-Video checkpoints directory. Download from https://huggingface.co/zai-org/cogvlm2-llama3-caption")
    parser.add_argument("--input_dir", type=str, default="./test_samples/inputs", help="Path to input videos directory.")
    parser.add_argument("--output_dir", type=str, default="./test_samples/outputs", help="Path to output videos directory.")
    parser.add_argument("--upscale", type=float, default=0., help='The upsample scale. Default upscale=0, short-size resized to 1024.')
    parser.add_argument('--tile_size', type=int, default=128)
    parser.add_argument('--num_temporal_process_frames', type=int, default=121)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=int, default=6)
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--textfix", action='store_true')
    parser.add_argument("--save_images", action='store_true')
    args = parser.parse_args()

    args.vividvr_ckpt_path = os.path.join(args.ckpt_dir, 'Vivid-VR')
    if args.textfix:
        args.easyocr_ckpt_path = os.path.join(args.ckpt_dir, 'easyocr')
        args.realesrgan_ckpt_path = os.path.join(args.ckpt_dir, 'RealESRGAN/RealESRGAN_x2plus.pth')

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading CogVLM2 captioner")
    captioner_model = CogVLM2_Captioner(model_path=args.cogvlm2_ckpt_path)

    print("Loading T5 text encoder")
    text_encoder = T5EncoderModel.from_pretrained(
        args.cogvideox_ckpt_path,
        subfolder="text_encoder"
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(dtype=torch.bfloat16)

    print("Loading transformer")
    transformer = CogVideoXVividVRTransformer3DModel.from_pretrained(
        args.cogvideox_ckpt_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    transformer.requires_grad_(False)
    transformer.to(dtype=torch.bfloat16)
    transformer.patch_embed.use_positional_embeddings = False
    transformer.patch_embed.use_learned_positional_embeddings = False
    transformer.config.use_learned_positional_embeddings = False
    transformer.config.use_rotary_positional_embeddings = True

    print("Loading vae")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.cogvideox_ckpt_path,
        subfolder="vae"
    )
    vae.requires_grad_(False)
    vae.to(dtype=torch.bfloat16)
    vae.enable_slicing()
    vae.enable_tiling()

    print("Loading controlnet")
    controlnet = CogVideoXVividVRControlNetModel.from_transformer(
        transformer=transformer,
        num_layers=6,
    )
    controlnet.requires_grad_(False)
    controlnet.to(dtype=torch.bfloat16)

    print("Loading scheduler")
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.cogvideox_ckpt_path,
        subfolder="scheduler"
    )

    print('Loading checkpoints')
    transformer.connectors.load_state_dict(torch.load(os.path.join(args.vividvr_ckpt_path, "connectors.pt"), map_location='cpu'))
    transformer.control_feat_proj.load_state_dict(torch.load(os.path.join(args.vividvr_ckpt_path, "control_feat_proj.pt"), map_location='cpu'))
    transformer.control_patch_embed.load_state_dict(torch.load(os.path.join(args.vividvr_ckpt_path, "control_patch_embed.pt"), map_location='cpu'))
    load_model = CogVideoXVividVRControlNetModel.from_pretrained(args.vividvr_ckpt_path, subfolder="controlnet")
    controlnet.register_to_config(**load_model.config)
    controlnet.load_state_dict(load_model.state_dict())
    del load_model

    free_memory()

    # create pipeline
    pipe = CogVideoXVividVRControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path=args.cogvideox_ckpt_path,
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        controlnet=controlnet,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )

    # enable cpu offload
    pipe.enable_model_cpu_offload(device=args.device)  # faster, but use more GPU memory
    # pipe.enable_sequential_cpu_offload(device=args.device)  # slower, but use less GPU memory

    # textfix
    if args.textfix:
        from VRDiT.textfix import TextFixer
        text_fixer = TextFixer(easyocr_model_path=args.easyocr_ckpt_path, enhancer_model_path=args.realesrgan_ckpt_path)

    # read input videos
    validation_infos = []
    vnames = os.listdir(args.input_dir)
    if os.path.isdir(vnames[0]):
        for vn in vnames:
            fnames = os.listdir(os.path.join(args.input_dir, vn))
            if fnames[0].split('.')[-1].lower() not in VALID_IMAGE_EXTENSIONS:
                print(f'Skip {vn}, because it is not a video!')
                continue
            img = cv2.imread(os.path.join(args.input_dir, vn, fnames[0]))
            height, width = img.shape[:2]
            validation_infos.append({
                'path': os.path.join(args.input_dir, vn),
                'width': width,
                'height': height,
                'fps': 24,
            })
    else:
        for vn in vnames:
            if vn.split('.')[-1].lower() not in VALID_VIDEO_EXTENSIONS:
                print(f'Skip {vn}, because it is not a video!')
                continue
            cap = cv2.VideoCapture(os.path.join(args.input_dir, vn))
            if not cap.isOpened():
                print('Open video file fail!')
                continue
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            validation_infos.append({
                'path': os.path.join(args.input_dir, vn),
                'width': width,
                'height': height,
                'fps': fps,
            })

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)

    for info in tqdm(validation_infos, 'Restoration'):
        basename = os.path.splitext(info['path'].split('/')[-1])[0]
        save_filepath = os.path.join(args.output_dir, "videos", f"{basename}.mp4")

        if os.path.isfile(save_filepath):
            print(f"{info['path']} has already been processed, skipping...")
            continue

        # [F, C, H, W]
        control_video = load_video(info['path'])
        if args.upscale == 0.:
            scale_factor = 1024. / min(control_video.size()[2], control_video.size()[3])
            control_video = F.interpolate(control_video, scale_factor=scale_factor, mode='bicubic').clip(0, 1)
            info['height'], info['width'] = control_video.size()[2], control_video.size()[3]
        elif args.upscale != 1.0:
            control_video = F.interpolate(control_video, scale_factor=args.upscale, mode='bicubic').clip(0, 1)
            info['height'], info['width'] = control_video.size()[2], control_video.size()[3]
        print(f"Processing {info['path']} with shape {control_video.shape}.")

        vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
        gen_height = 8 * math.ceil(info['height'] / 8) if info['height'] < args.tile_size * vae_scale_factor_spatial else info['height']
        gen_width = 8 * math.ceil(info['width'] / 8) if info['width'] < args.tile_size * vae_scale_factor_spatial else info['width']
        print(f"Generate resolution {gen_height}x{gen_width} for {info['path']}.")

        if control_video.size(0) > args.num_temporal_process_frames:
            video = infer_split_clips(
                args=args,
                captioner_model=captioner_model,
                pipe=pipe,
                info=info,
                control_video=control_video,
                gen_height=gen_height,
                gen_width=gen_width,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                vae_scale_factor_temporal=vae.config.temporal_compression_ratio
            )
        else:
            video = infer_whole_video(
                args=args,
                captioner_model=captioner_model,
                pipe=pipe,
                info=info,
                control_video=control_video,
                gen_height=gen_height,
                gen_width=gen_width,
                vae_scale_factor_spatial=vae_scale_factor_spatial
            )
        
        # colorfix
        samples = adaptive_instance_normalization(torch.from_numpy(video).permute(0, 3, 1, 2).to(args.device), control_video.to(args.device))

        # textfix
        if args.textfix:
            samples = text_fixer(video=samples, ref_video=control_video, device=args.device)

        # save output video and image
        samples = samples.cpu().clip(0, 1).permute(0, 2, 3, 1).float().numpy()
        export_to_video(samples, save_filepath, fps=info['fps'])
        if args.save_images:
            image_dir = os.path.join(args.output_dir, "images", f"{basename}")
            os.makedirs(image_dir, exist_ok=True)
            for i in range(len(samples)):
                Image.fromarray((samples[i] * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(image_dir, f"{i:06d}.png"))

        # print GPU memory usage
        print(torch.cuda.memory_summary(abbreviated=False))

        del video, control_video, samples
        free_memory()


if __name__ == "__main__":
    main()
