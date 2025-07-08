import torch
import torchaudio
from einops import rearrange
from src.audiox.inference.generation import generate_diffusion_cond
from src.audiox.data.utils import read_video, merge_video_audio
from src.audiox.data.utils import load_and_process_audio

from src.audiox.models.factory import create_model_from_config
from src.audiox.models.utils import load_ckpt_state_dict
import json
from pathlib import Path
import argparse
import numpy as np


MODEL_CONFIG_PATH = "/mnt/data/audiosplat/model/audiox_config.json"
MODEL_CKPT_PATH = "/mnt/data/audiosplat/model/audiox_model.ckpt"

class AudioXWrapper:
    def __init__(self, seed=1102, device="cuda"):
        self.device = device
        with open(MODEL_CONFIG_PATH) as f:
            model_config = json.load(f)
        self.model_config = model_config
        self.sample_rate = model_config["sample_rate"]
        self.sample_size = model_config["sample_size"]
        self.target_fps = model_config["video_fps"]
        
        self.model = create_model_from_config(model_config)
        self.model.load_state_dict(load_ckpt_state_dict(MODEL_CKPT_PATH))
        self.model = self.model.to(device)
        
        self.rng = np.random.default_rng(seed)
    
    def _generate(self, video_tensor, audio_tensor, text_prompt, seconds_total=10.0, seed=1102):
        conditioning = [{
            "video_prompt": [video_tensor.unsqueeze(0)],        
            "text_prompt": text_prompt,
            "audio_prompt": audio_tensor.unsqueeze(0),
            "seconds_start": 0,
            "seconds_total": seconds_total
        }]
        
        output = generate_diffusion_cond(
            self.model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=self.sample_size,
            sigma_min=0.3,
            sigma_max=500,
            seed=seed,
            sampler_type="dpmpp-3m-sde",
            device=self.device
        )
        
        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        return output

    def generate(self, video, audio, text_prompt, seconds_total, num_samples=1):
        if isinstance(video, str):
            video = read_video(video, seek_time=0, duration=seconds_total, target_fps=self.target_fps)
        
        if isinstance(audio, str) or audio is None:
            audio = load_and_process_audio(audio, self.sample_rate, seconds_start=0, seconds_total=seconds_total)
        
        seed_list = self.rng.choice(np.arange(1000000), size=num_samples, replace=False)
        
        outputs = [
            self._generate(video, audio, text_prompt, seconds_total, seed)
            for seed in seed_list
        ]
        
        return outputs, seed_list
        


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1102)
    parser.add_argument("--total_seconds", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=3)
    return parser.parse_args()

if __name__ == "__main__":
    args = argument_parser()
    
    audiox = AudioXWrapper(seed=args.seed)
    
    result_json_path = Path("assets") / args.dataset / "results.json"
    
    with open(result_json_path, "r") as f:
        results = json.load(f)
    
    for img_name, result in results.items():
        video_path = result["image_path"]
        audio_path = None
        
        interaction = "Generate general audio for the image"
        
        # for interaction in result["interactions"]:
        outputs, seed_list = audiox.generate(video_path, audio_path, interaction, args.duration, args.num_samples)

        output_dir = result_json_path.parent / "audiox" / img_name.split(".")[0] / interaction.replace(" ", "_")
        output_dir.mkdir(parents=True, exist_ok=True)

        for output, seed in zip(outputs, seed_list):
            torchaudio.save(output_dir / f"{seed}.wav", output, audiox.sample_rate)
