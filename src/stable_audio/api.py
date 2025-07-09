import argparse
from pathlib import Path
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

import torch
import torchaudio

from einops import rearrange


def generate(
    model,
    model_config,
    prompt,
    negative_prompt="Low quality.",
    seconds_total=5,
    cfg_scale=7.0,
    steps=100,
    seed=1102,
    sampler_type="dpmpp-3m-sde",
    sigma_min=0.01,
    sigma_max=100,
    rho=1.0,
    cfg_rescale=0.0,
    init_audio=None,
    init_noise_level=0.1,
    batch_size=1,
    device="cuda",
):
    sample_size = model_config["sample_size"]
    sample_rate = model_config["sample_rate"]

    # Return fake stereo audio
    conditioning_dict = {"prompt": prompt, "seconds_start": 0, "seconds_total": seconds_total}

    conditioning = [conditioning_dict] * batch_size

    if negative_prompt:
        negative_conditioning_dict = {"prompt": negative_prompt, "seconds_start": 0, "seconds_total": seconds_total}

        negative_conditioning = [negative_conditioning_dict] * batch_size
    else:
        negative_conditioning = None

    generate_args = {
        "model": model,
        "conditioning": conditioning,
        "negative_conditioning": negative_conditioning,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "cfg_interval": (0.0, 1.0),
        "batch_size": batch_size,
        "sample_size": sample_size,
        "seed": int(seed),
        "device": device,
        "sampler_type": sampler_type,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "init_audio": init_audio,
        "init_noise_level": init_noise_level,
        "callback": None,
        "scale_phi": cfg_rescale,
        "rho": rho
    }
    audio = generate_diffusion_cond(**generate_args)
    audio = audio[:,:,:int(seconds_total*sample_rate)]
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()    

    return audio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("output"))
    parser.add_argument("-s", "--seed", type=int, default=1102)
    parser.add_argument("-n", "--num_samples", type=int, default=1)
    return parser.parse_args()

def main(args):
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model.to("cuda").eval().requires_grad_(False)
    model.to(torch.float16)

    for i in range(args.num_samples):
        seed = args.seed + i
        audio = generate(
            model,
            model_config,
            prompt=args.prompt,
            seconds_total=5.0,
            seed=seed,
            device="cuda"
        )

        audio_path = args.output_dir / f"{args.prompt.replace(' ', '_')}/{seed}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(audio_path, audio, model_config["sample_rate"])

if __name__ == "__main__":
    args = parse_args()
    main(args)
