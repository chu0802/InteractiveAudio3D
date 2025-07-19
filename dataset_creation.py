import json
import numpy as np
import argparse
from pathlib import Path
import shutil
from collections import defaultdict


def main(args):
    data_dir = args.dataset_dir / args.scene_name

    # clear the output directory
    if not args.statistics:
        if (args.output_dir / args.scene_name).exists():
            for sub_dir in (args.output_dir / args.scene_name).iterdir():
                for file in sub_dir.iterdir():
                    if file.is_symlink():
                        file.unlink()
                shutil.rmtree(sub_dir)

    for sub_dir in sorted(data_dir.iterdir()):
        if args.target_obj and sub_dir.name.replace("_", " ") != args.target_obj:
            continue

        print("-"*50)
        print("Object Name: ", sub_dir.name)


        with open(sub_dir / "rewards.json", "r") as f:
            data = json.load(f)

        catagorized_data = defaultdict(dict)
        for p, v in data.items():
            catagorized_data[Path(p).parent.name][p] = v

        filtered_data = {}

        for action, action_data in catagorized_data.items():
            overall_scores = [entry['scores']['overall'] for entry in action_data.values()]

            filter_threshold = np.percentile(overall_scores, args.filter_threshold)
            
            if args.mode == "gt":
                new_data = {k: v for k, v in action_data.items() if v['scores']['overall'] >= filter_threshold}
            else:
                new_data = {k: v for k, v in action_data.items() if v['scores']['overall'] <= filter_threshold}

            filtered_data.update(new_data)


            # print statistcs of the new data
            print(f"action: {action}, number of samples: {len(new_data)}")
            
            print(f"max score: {max(overall_scores)}")
            print(f"min score: {min(overall_scores)}")
            print(f"mean score: {np.mean(overall_scores)}")
            print(f"25th percentile: {np.percentile(overall_scores, 25)}")
            print(f"medium score: {np.percentile(overall_scores, 50)}")
            print(f"75th percentile: {np.percentile(overall_scores, 75)}")
            
            if args.filter_threshold not in [25, 50, 75]:
                print(f"{args.filter_threshold}th percentile score: {filter_threshold}")

            print("="*25)

        if not args.statistics:
            for audio_path in filtered_data.keys():
                audio_path_relative = Path(audio_path).relative_to(data_dir)
                audio_path_output = args.output_dir / audio_path_relative
                audio_path_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(audio_path, audio_path_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("output"))
    parser.add_argument("-n", "--scene_name", type=str, default="0118_bathroom")
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("positive_samples"))
    parser.add_argument("-f", "--filter_threshold", type=int, default=95)
    parser.add_argument("-s", "--statistics", action="store_true", default=False)
    parser.add_argument("-m", "--mode", type=str, default="gt", choices=["gt", "lt"])
    parser.add_argument("-t", "--target_obj", type=str, default=None)
    args = parser.parse_args()
    main(args)
