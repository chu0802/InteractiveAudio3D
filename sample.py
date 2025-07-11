from pathlib import Path
import json
from script.stable_audio import generate_audio
from diffusers import StableAudioPipeline
import soundfile as sf
import argparse
import torch
from src.stable_audio.api import load_model, generate
import torchaudio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("datasets"))
    parser.add_argument("-c", "--scene_name", type=str, default="0118_bathroom")
    parser.add_argument("-s", "--seed", type=int, default=1102)
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("output"))
    
    parser.add_argument("-n", "--num_samples", type=int, default=10)
    parser.add_argument("-m", "--model_config", type=Path, default=None)
    parser.add_argument("-l", "--lora_ckpt_path", type=Path, default=None)
    return parser.parse_args()

def main(args):
    with (args.dataset_dir / args.scene_name / "interaction_results.json").open("r") as f:
        info = json.load(f)["objects"]
        
    model, model_config = load_model(args.model_config, args.lora_ckpt_path)
    
    for object_name, object_info in info.items():
        for description in object_info["descriptions"]:
            prompt = f"{description}. High quality, realistic, and clear."
            audio = generate(
                model,
                model_config,
                steps=100,
                prompt=prompt,
                seconds_total=5,
                seed=args.seed,
                batch_size=args.num_samples,
                device="cuda"
            )
            
            for i in range(args.num_samples):
                audio_path = args.output_dir / args.scene_name / f"{object_name.replace(' ', '_')}/{description.replace(' ', '_')}/{args.seed}_{i}.wav"
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(audio_path, audio[i], model_config["sample_rate"])

if __name__ == "__main__":
    args = parse_args()
    main(args)
