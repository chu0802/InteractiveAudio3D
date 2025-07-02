from pathlib import Path
import json
from script.stable_audio import generate_audio
from diffusers import StableAudioPipeline
import soundfile as sf
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="0118_bathroom")
    parser.add_argument("--image_id", type=int, default=0)
    parser.add_argument("--interaction_id", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1102)
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    return parser.parse_args()

def main(args):
    with open("assets/results.json", "r") as f:
        results = json.load(f)

    data = results[args.scene_name][f"{args.image_id:03d}"]
    
    with open(args.output_dir / args.scene_name / f"{args.image_id:03d}" / "interactions.json", "r") as f:
        interactions = json.load(f)
    
    for interaction_id in range(5):
        interaction_prompt = interactions["interactions"][interaction_id]
        
        # interaction_prompt = data["interactions"][args.interaction_id]["prompt"]

        pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        
        for i in range(args.num_samples):
            seed = args.seed + i
            audio = generate_audio(pipe, interaction_prompt, duration=5.0, seed=seed)
            
            audio_path = args.output_dir / args.scene_name / f"{args.image_id:03d}" / f"{'_'.join(interaction_prompt.split(' '))}/{seed}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)

            sf.write(audio_path, audio, pipe.vae.sampling_rate)
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
