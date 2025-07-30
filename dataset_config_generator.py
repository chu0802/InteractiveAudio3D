from pathlib import Path
import json
import argparse
DATASET_CONFIG_TEMPLATE = {
    "dataset_type": "audio_dir",
    "datasets": [],
    "random_crop": True,
    "drop_last": False
}

def generate_dataset_config(args):
    dataset_config = DATASET_CONFIG_TEMPLATE.copy()

    dir_name = f"pos_overall_{args.filter_threshold}"
    if args.iter > 1 and args.epoch is not None:
        dir_name += f"_epoch{args.epoch}"

    dataset_config["datasets"].append({
        "id": args.object_name,
        "path": (
            Path("logs/datasets") /
            args.scene_name /
            args.object_name /
            f"iter{args.iter}" /
            dir_name
        ).as_posix(),
        "custom_metadata_module": "src/stable_audio/custom_metadata.py",
    })
    
    return dataset_config
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filter_threshold", type=int, default=95)
    parser.add_argument("-s", "--scene_name", type=str, default="0118_bathroom")
    parser.add_argument("-i", "--iter", type=int, default=1)
    parser.add_argument("-e", "--epoch", type=int, default=50)
    parser.add_argument("-o", "--object_name", type=str, default="ceramic_mug")
    args = parser.parse_args()
    args.object_name = args.object_name.replace(" ", "_")
    config = generate_dataset_config(args)
    
    with open(f"stable_audio_config/dataset_config.json", "w") as f:
        json.dump(config, f, indent=4)
