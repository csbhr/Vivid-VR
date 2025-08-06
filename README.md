<div align="center">

<h1>
    Vivid-VR:<br> 
    Distilling Concepts from Diffusion Transformer for Photorealistic Video Restoration
</h1>

<div>
    <a href='https://csbhr.github.io/' target='_blank'>Haoran Bai</a>,&emsp;
    <a href='https://github.com/chenxx89' target='_blank'>Xiaoxu Chen</a>,&emsp;
    <a href='https://ieeexplore.ieee.org/author/37088928879' target='_blank'>Canqian Yang</a>,&emsp;
    <a href='https://github.com/HeZongyao' target='_blank'>Zongyao He</a>,&emsp;
    <a href='https://scholar.google.com/citations?user=brmDxnsAAAAJ&hl=zh-CN' target='_blank'>Sibin Deng</a>,&emsp;
    <a href='https://scholar.google.com/citations?user=NpTmcKEAAAAJ&hl=en' target='_blank'>Ying Chen<sup>∗</sup></a>
</div>
<div>
    Alibaba Group - Taobao & Tmall Group
</div>
<div>
    * Corresponding author
</div>

<a href='#' target='_blank'>Paper (<span style='color:red;'>Coming soon!</span>)</a> | 
<a href='https://csbhr.github.io/projects/vivid-vr/' target='_blank'>Project Page</a>


<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="assets/teaser.png">
</div>

For more quantitative results and visual results, go checkout our <a href="https://csbhr.github.io/projects/vivid-vr/" target="_blank">[project page]</a>

---
</div>


## 🔥 Update
- [2025.08.06] UGC50 and AIGC50 testsets are made publicly available from [[link]](https://huggingface.co/csbhr/Vivid-VR/blob/main/testset.zip).
- [2025.08.05] Inference code is released.
- [2025.08.05] This repo is created.

## 🎬 Overview
![overall_structure](assets/framework.png)

## 🔧 Dependencies and Installation
1. Clone Repo
    ```bash
    git clone https://github.com/csbhr/Vivid-VR.git
    cd Vivid-VR
    ```

2. Create Conda Environment and Install Dependencies
    ```bash
    # create new conda env
    conda create -n Vivid-VR python=3.10
    conda activate Vivid-VR

    # install pytorch
    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

    # install python dependencies
    pip install -r requirements.txt

    # install easyocr [Optional, for text fix]
    pip install easyocr
    pip install numpy==1.26.4  # numpy2.x maybe installed when installing easyocr, which will cause conflicts
    ```

3. Download Models

   - [**Required**] Download CogVideoX1.5-5B checkpoints from [[huggingface]](https://huggingface.co/zai-org/CogVideoX1.5-5B).
   - [**Required**] Download cogvlm2-llama3-caption checkpoints from [[huggingface]](https://huggingface.co/zai-org/cogvlm2-llama3-caption).
   - [**Required**] Download Vivid-VR checkpoints from [[huggingface]](https://huggingface.co/csbhr/Vivid-VR).
   - [**Optional, for text fix**] Download easyocr checkpoints [[english_g2]](https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip) [[zh_sim_g2]](https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/zh_sim_g2.zip) [[craft_mlt_25k]](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip).
   - [**Optional, for text fix**] Download Real-ESRGAN checkpoints [[RealESRGAN_x2plus]](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth).
   - Put them under the `./ckpts` folder.

   The [`ckpts`](./ckpts) directory structure should be arranged as:

    ```
    ├── ckpts
    │   ├── CogVideoX1.5-5B
    │   │   ├── ...
    │   ├── cogvlm2-llama3-caption
    │   │   ├── ...
    │   ├── Vivid-VR
    │   │   ├── controlnet
    │   │       ├── config.json
    │   │       ├── diffusion_pytorch_model.safetensors
    │   │   ├── connectors.pt
    │   │   ├── control_feat_proj.pt
    │   │   ├── control_patch_embed.pt
    │   ├── easyocr
    │   │   ├── craft_mlt_25k.pth
    │   │   ├── english_g2.pth
    │   │   ├── zh_sim_g2.pth
    │   ├── RealESRGAN
    │   │   ├── RealESRGAN_x2plus.pth
    ```


## ☕️ Quick Inference

Run the following commands to try it out:

```shell
python VRDiT/inference.py \
    --ckpt_dir=./ckpts \
    --cogvideox_ckpt_path=./ckpts/CogVideoX1.5-5B \
    --cogvlm2_ckpt_path=./ckpts/cogvlm2-llama3-caption \
    --input_dir=/dir/to/input/videos \
    --output_dir=/dir/to/output/videos \
    --upscale=0 \  # Optional, if set to 0, the short-size of output videos will be 1024
    --textfix \  # Optional, if given, the text region will be replaced by the output of Real-ESRGAN
    --save_images  # Optional, if given, the video frames will be saved

```


## 📧 Citation

   If you find our repo useful for your research, please consider citing it:

   ```bibtex
   @misc{bai2025vividvr,
      title={Vivid-VR: Distilling Concepts from Diffusion Transformer for Photorealistic Video Restoration}, 
      author={Haoran Bai and Xiaoxu Chen and Canqian Yang and Zongyao He and Sibin Deng and Ying Chen},
      year={2025},
      url={https://github.com/csbhr/Vivid-VR}
    }
   ```


## 📄 License
- diffusers is distributed under the terms of the [Apache License 2.0](https://github.com/huggingface/diffusers/blob/main/LICENSE).
- CogVideoX1.5-5B models are distributed under the terms of the [CogVideoX License](https://huggingface.co/zai-org/CogVideoX1.5-5B/blob/main/LICENSE).
- cogvlm2-llama3-caption models are distributed under the terms of the [CogVLM2 License](https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-base/file/view/master?fileName=LICENSE&status=0) and [LLAMA3 License](https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-base/file/view/master?fileName=LLAMA3_LICENSE&status=0).
- Real-ESRGAN models are distributed under the terms of the [BSD 3-Clause License](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).
- easyocr models are distributed under the terms of the [JAIDED.AI Terms and Conditions](https://www.jaided.ai/terms/).

