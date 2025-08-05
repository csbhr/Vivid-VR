"""
Require: transformers==4.42.4
"""

import cv2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CogVLM2_Captioner():
    def __init__(self, model_path, torch_type=torch.bfloat16):
        super().__init__()
        self.torch_type = torch_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=self.torch_type, trust_remote_code=True).eval()
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)

    def __call__(
        self,
        video,
        prompt='Please describe this video in detail.',
        temperature=0.1, fps=None, start_frame=None, end_frame=None, crop_center=False
    ):
        '''
        video: str or tensor
            str: the video path
            tensor: [T, C, H, W], torch.float32, (0,1)
        '''
        if isinstance(video, str):
            video = self.load_video(video, fps=fps, start_frame=start_frame, end_frame=end_frame, crop_center=crop_center)
            response = self.predict(prompt, video, temperature)
            return str(response).strip()
        else:
            video = video * 255  # (0,1) -> (0,255)
            fps = fps if fps else min(15, video.shape[0])
            video = torch.stack(list(map(video.__getitem__, self.get_index(video.shape[0], fps))), dim=0)  # sample frames by fps
            video = video.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]
            response = self.predict(prompt, video, temperature)
            return str(response).strip()
    
    def predict(self, prompt, video, temperature):
        inputs = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=prompt,
            images=[video],
            history=[],
            template_version='chat'
        )
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[inputs['images'][0].to(self.device).to(self.torch_type)]],
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": False,
            "top_p": 0.1,
            "temperature": temperature,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
    
    def get_index(self, total_frames, fps):
        fps = round(fps)
        num_segments = max(total_frames // fps, 1)
        index_list = [i * fps for i in range(num_segments)]
        if index_list[-1] != total_frames - 1:
            index_list.append(total_frames - 1)
        return index_list

    def load_video(self, video_path, fps=None, start_frame=None, end_frame=None, crop_center=False):
        video_data_list = []

        if os.path.isfile(video_path):
            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_frame = start_frame if start_frame else 0
            end_frame = end_frame if end_frame else num_frames
            total_frames = end_frame - start_frame
            fps = fps if fps else float(cap.get(cv2.CAP_PROP_FPS))

            frame_indices = self.get_index(total_frames, fps)
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_index)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if crop_center:
                    h, w = frame.shape[:2]
                    c_size = min(h, w)
                    top, left = (h - c_size) // 2, (w - c_size) // 2
                    frame = frame[top:top + c_size, left:left + c_size, :]
                    print(f"Info: Crop center [{top}:{top + c_size}, {left}:{left + c_size}] from [{h}, {w}].")
                frame = torch.tensor(frame, dtype=torch.uint8)
                video_data_list.append(frame)
            cap.release()
        else:
            fnames = sorted(os.listdir(video_path))
            num_frames = len(fnames)
            start_frame = start_frame if start_frame else 0
            end_frame = end_frame if end_frame else num_frames
            total_frames = end_frame - start_frame
            fps = fps if fps else min(15, total_frames)

            frame_indices = self.get_index(total_frames, fps)
            for frame_index in frame_indices:
                frame = cv2.imread(os.path.join(video_path, fnames[start_frame + frame_index]), cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if crop_center:
                    h, w = frame.shape[:2]
                    c_size = min(h, w)
                    top, left = (h - c_size) // 2, (w - c_size) // 2
                    frame = frame[top:top + c_size, left:left + c_size, :]
                    print(f"Info: Crop center [{top}:{top + c_size}, {left}:{left + c_size}] from [{h}, {w}].")
                frame = torch.tensor(frame, dtype=torch.uint8)
                video_data_list.append(frame)
        assert len(video_data_list) > 0, f'Read video error: len(video_data_list)={len(video_data_list)}.'
        video_data = torch.stack(video_data_list, dim=0)
        video_data = video_data.permute(3, 0, 1, 2)  # [C, T, H, W]
        return video_data
