from pathlib import Path
import json
import argparse
import torch
from src.stable_audio.api import load_model, generate
import torchaudio
from accelerate import PartialState
from accelerate.utils import gather_object
from numpy.random import default_rng


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("datasets"))
    parser.add_argument("-c", "--scene_name", type=str, default="0118_bathroom")
    parser.add_argument("-s", "--seed", type=int, default=1102)
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("output"))
    parser.add_argument("-t", "--target_obj", type=str, default=None)
    parser.add_argument("-n", "--num_samples", type=int, default=10)
    parser.add_argument("-m", "--model_config", type=Path, default="stable_audio_config/model_config.json")
    parser.add_argument("-l", "--lora_ckpt_path", type=Path, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    return parser.parse_args()

def generate_jobs(info_list, num_samples, batch_size, seed=1102):
    seed_rng = default_rng(seed)
    num_job_per_prompt = (num_samples // batch_size + 1)
    seed_list = seed_rng.integers(0, 1e6, num_job_per_prompt * len(info_list))
    jobs = []
    for i, info in enumerate(info_list):
        for j in range(num_job_per_prompt):
            jobs.append({
                "obj_name": info["obj_name"],
                "description": info["description"],
                "prompt": info["prompt"],
                "seed": seed_list[i * num_job_per_prompt + j],
                "batch_size": batch_size if j != num_job_per_prompt - 1 else num_samples % batch_size,
            })
    return jobs

def main(args):
    with (args.dataset_dir / args.scene_name / "interaction_results.json").open("r") as f:
        info = json.load(f)["objects"]
    distributed_state = PartialState()

    model, model_config = load_model(args.model_config, args.lora_ckpt_path, device=distributed_state.device)

    info_list = []
    for object_name, object_info in info.items():
        if args.target_obj and object_name != args.target_obj:
            continue
        for description in object_info["descriptions"]:
            info_list.append({"obj_name": object_name, "description": description, "prompt": f"{description}. High quality, realistic, and clear."})

    jobs = generate_jobs(info_list, args.num_samples, args.batch_size, args.seed)
    overall_res = []
    with distributed_state.split_between_processes(jobs, apply_padding=True) as distributed_jobs:
        for job in distributed_jobs:
            audio = generate(
                model,
                model_config,
                steps=100,
                prompt=job["prompt"],
                seconds_total=5,
                seed=job["seed"],
                batch_size=job["batch_size"],
                device=distributed_state.device,
            )
            overall_res.append({"obj_name": job["obj_name"], "description": job["description"], "seed": job["seed"], "audio": audio})
    
    overall_res = gather_object(overall_res)
    overall_res = overall_res[:len(jobs)]
    
    if distributed_state.is_main_process:
        for res in overall_res:
            for i, audio in enumerate(res["audio"]):
                audio_path = args.output_dir / args.scene_name / f"{res['obj_name'].replace(' ', '_')}/{res['description'].replace(' ', '_')}/{res['seed']}_{i}.wav"
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(audio_path, audio, model_config["sample_rate"])

if __name__ == "__main__":
    args = parse_args()
    main(args)
