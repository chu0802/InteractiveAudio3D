import argparse
from pathlib import Path
from diffusers import StableAudioPipeline
import json
import torch
import soundfile as sf

def generate_audio(pipe, prompt, duration, seed=1102):
    negative_prompt = "Low quality."
    audio = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=100,
        audio_end_in_s=duration,
        num_waveforms_per_prompt=1,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).audios

    return audio[0].T.float().cpu().numpy()

def main(args):
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    result_json_path = Path("output") / args.dataset / args.mode / "intermediate_results" / f"{args.image_id:05d}" / "results.json"

    with result_json_path.open("r") as f:
        results = json.load(f)

    for img_name, result in results.items():
        audio_dir = result_json_path.parent / "audio" / img_name.split(".")[0]
        audio_dir.mkdir(parents=True, exist_ok=True)

        for interaction in result["interactions"]:
            audio = generate_audio(pipe, interaction, args.duration, args.seed)
            audio_path = audio_dir / f"{interaction.replace(' ', '_')}_{args.seed}.wav"
            sf.write(audio_path, audio, pipe.vae.sampling_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--image_id", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1102)
    args = parser.parse_args()
    main(args)
