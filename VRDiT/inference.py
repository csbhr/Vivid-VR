
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../')))

from VRDiT.cogvlm2 import CogVLM2_Captioner
from VRDiT.colorfix import adaptive_instance_normalization
from VRDiT.textfix import TextFixer
from VRDiT.utils import VALID_IMAGE_EXTENSIONS, VALID_VIDEO_EXTENSIONS, free_memory, load_video, export_to_video, prepare_validation_prompts


def main():
    parser = argparse.ArgumentParser(description="Vivid-VR")
    parser.add_argument("--ckpt_dir", type=str, default='./ckpts', help="Path to checkpoints directory.")
    parser.add_argument("--cogvideox_ckpt_path", type=str, default='./ckpts/CogVideoX1.5-5B', help="Path to CogVideoX1.5-5B checkpoints directory. Download from https://huggingface.co/zai-org/CogVideoX1.5-5B")
    parser.add_argument("--cogvlm2_ckpt_path", type=str, default='./ckpts/cogvlm2-llama3-caption', help="Path to CogVLM2-Video checkpoints directory. Download from https://huggingface.co/zai-org/cogvlm2-llama3-caption")
    parser.add_argument("--input_dir", type=str, default="./test_samples/inputs", help="Path to input videos directory.")
    parser.add_argument("--output_dir", type=str, default="./test_samples/outputs", help="Path to output videos directory.")
    parser.add_argument("--upscale", type=float, default=0., help='The upsample scale. Default upscale=0, short-size resized to 1024.')
    parser.add_argument('--tile_size', type=int, default=128)
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
        args.easyocr_ckpt_path = os.path.join(args.ckpt_dir, 'easyocr'),
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

    # textfix
    if args.textfix:
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

        # Padding to make the number of frames pad_type
        num_padding_frames = 0
        if (control_video.size(0) - 1) % 8 != 0:
            num_padding_frames = 8 - (control_video.size(0) - 1) % 8
            control_video = torch.cat([control_video, control_video[-1:].repeat(num_padding_frames, 1, 1, 1)], dim=0)

        vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
        gen_height = 8 * math.ceil(info['height'] / 8) if info['height'] < args.tile_size * vae_scale_factor_spatial else info['height']
        gen_width = 8 * math.ceil(info['width'] / 8) if info['width'] < args.tile_size * vae_scale_factor_spatial else info['width']
        print(f"Generate resolution {gen_height}x{gen_width} for {info['path']}.")

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
        pipe.enable_model_cpu_offload(device=args.device)
        video = pipe(
            **pipeline_args,
            generator=torch.Generator(device=args.device).manual_seed(args.seed),
            output_type="np"  # numpy [T, H, W, C]
        ).frames[0]

        if video.shape[0] % 4 == 0:
            video = video[3:]
        if num_padding_frames > 0:
            video = video[:-num_padding_frames]
            control_video = control_video[:-num_padding_frames]
        
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

        del video, control_video, samples
        free_memory()


if __name__ == "__main__":
    main()
