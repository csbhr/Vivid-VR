import cv2
import numpy as np
import torch
import torch.nn.functional as F
import easyocr
from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
from easyocr.imgproc import resize_aspect_ratio, normalizeMeanVariance

from VRDiT.enhancer import Enhancer


def text_detect(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device,
                estimate_num_chars=False, return_score=False):
    """
    Detect text in the image using the given text detector.
    """

    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize
    for img in image_arrs:
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size,
                                                                      interpolation=cv2.INTER_LINEAR,
                                                                      mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    x = [np.transpose(normalizeMeanVariance(n_img), (2, 0, 1))
         for n_img in img_resized_list]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    boxes_list, polys_list = [], []
    for out in y:
        # make score and link map
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys, mapper = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    if return_score:
        scores = []
        for i, score in enumerate(y):
            # Resize each score map `score` back to the original image size
            original_height, original_width = image_arrs[i].shape[:2]
            resized_score = cv2.resize(score.cpu().numpy(),
                                       (original_width, original_height),
                                       interpolation=cv2.INTER_CUBIC)
            scores.append(resized_score)

        results = []
        for polys in polys_list:
            single_img_result = []
            for i, box in enumerate(polys):
                poly = np.array(box).astype(np.int32).reshape((-1))
                single_img_result.append(poly)
            results.append(single_img_result)

        return scores, results
    else:
        return boxes_list, polys_list


def enhance_score(score_text: torch.Tensor, text_threshold=0.7,
                  denoise_radius=5, dilate_radius=5, blur_radius=1):
    """
    Enhance text prob by denoising, dilating, and blurring.
    """
    score_text = torch.clamp((score_text - 0) / (text_threshold - 0), 0, 1)

    # denoise
    if denoise_radius > 0:
        max_pool = F.max_pool2d(score_text, kernel_size=2 * denoise_radius + 1, stride=1, padding=denoise_radius)
        mask = (max_pool < 1) & (score_text < 1)
        score_text[mask] *= score_text[mask]
    
    # dilate
    if dilate_radius > 0:
        score_text = F.max_pool2d(score_text, kernel_size=2 * dilate_radius + 1, stride=1, padding=dilate_radius)

    # blur
    if blur_radius > 0:
        mask = (score_text < 1)
        score_text[mask] = F.avg_pool2d(score_text, kernel_size=2 * blur_radius + 1, stride=1, padding=blur_radius)[mask]

    return score_text


def replace_text_regions(
    video, ref_video, text_detector,
    text_threshold=0.7, denoise_radius=5,
    dilate_radius=5, blur_radius=1, device='cpu'
):
    result_frames = []
    for this_frame, ref_frame in zip(video, ref_video):
        ref_frame_np = ref_frame.cpu().numpy().transpose(1, 2, 0) * 255
        ref_frame_scores, _ = text_detect(canvas_size=2560, mag_ratio=1., net=text_detector,
                                                        image=ref_frame_np, text_threshold=text_threshold,
                                                        link_threshold=0.4, low_text=0.4, poly=False,
                                                        device=device, estimate_num_chars=None, return_score=True)
        ref_frame_score_text = torch.from_numpy(ref_frame_scores[0]).to(device)[:, :, 0].unsqueeze(0)
        ref_frame_score_text = enhance_score(ref_frame_score_text, text_threshold=text_threshold,
                                            denoise_radius=denoise_radius, dilate_radius=dilate_radius,
                                            blur_radius=blur_radius)
        # Replace text regions
        result_frame = ref_frame_score_text * ref_frame + (1 - ref_frame_score_text) * this_frame
        result_frames.append(result_frame)
    result_video = torch.stack(result_frames, dim=0)
    return result_video


class TextFixer():
    def __init__(self, easyocr_model_path, enhancer_model_path):
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], model_storage_directory=easyocr_model_path, download_enabled=False)
        self.enhancer = Enhancer(ckpt_path=enhancer_model_path)
        self.text_threshold = 0.7
        self.denoise_radius = 5
        self.dilate_radius = 5
        self.blur_radius = 1

    def __call__(self, video, ref_video, device='cuda'):
        # [F, C, H, W]
        video = video.to(device)
        ref_video = ref_video.to(device)

        self.enhancer.to(device)
        ref_video = self.enhancer(ref_video, device=device)
        self.enhancer.to('cpu')

        out_video = replace_text_regions(
            video, ref_video, self.ocr_reader.detector,
            text_threshold=self.text_threshold,
            denoise_radius=self.denoise_radius,
            dilate_radius=self.dilate_radius,
            blur_radius=self.blur_radius,
            device=device
        )

        return out_video
