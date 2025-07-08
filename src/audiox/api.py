from diffusers import StableAudioPipeline
from script.stable_audio import generate_audio
import argparse
from pathlib import Path
import soundfile as sf
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("output"))
    parser.add_argument("-s", "--seed", type=int, default=1102)
    parser.add_argument("-n", "--num_samples", type=int, default=1)
    return parser.parse_args()


def main(args):
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    for i in range(args.num_samples):
        seed = args.seed + i
        audio = generate_audio(pipe, args.prompt, duration=5.0, seed=seed)

        audio_path = args.output_dir / f"{args.prompt.replace(' ', '_')}/{seed}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, audio, pipe.vae.sampling_rate)


if __name__ == "__main__":
    args = parse_args()
    main(args)
