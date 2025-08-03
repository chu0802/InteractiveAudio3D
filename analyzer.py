from pathlib import Path
import json
from collections import defaultdict
from itertools import product, combinations
import numpy as np
from compute_scores import result_2_list_to_dict

# write a function to calculate pair-wise difference in a list
def diff(x):
    if len(x) == 1:
        return None
    total_diff = 0
    for i, j in combinations(x, 2):
        total_diff += abs(i - j)
    return total_diff / (len(x) * (len(x) - 1) / 2)

if __name__ == "__main__":
    dir = Path("logs/audios/0118_bathroom/ceramic_mug/iter0")
    
    reward_files = list(dir.glob("rewards*.json"))
    qwen_files = list(dir.glob("improved_stage_1_results*.json"))

    configs = [qwen_file.stem.split("_")[4:8] for qwen_file in qwen_files]
    
    all_config_list = [list(set([config[i] for config in configs])) for i in range(4)]
    
    all_rewards = {}
    num_possibility = {}
    for temp, top_p, top_k in product(*all_config_list[:-1]):
        check_path = list(dir.glob(f"rewards{temp}_{top_p}_{top_k}*.json"))
        if len(check_path) == 0:
            continue
        
        data = defaultdict(dict)
        poss = defaultdict(list)
        for seed in all_config_list[-1]:
            reward_path = dir / f"rewards{temp}_{top_p}_{top_k}_{seed}.json"
            qwen_path = dir / f"improved_stage_1_results_{temp}_{top_p}_{top_k}_{seed}.json"
            
            with open(reward_path, "r") as f:
                reward_data = json.load(f)
            
            with open(qwen_path, "r") as f:
                qwen_data = json.load(f)
            
            qwen_res = result_2_list_to_dict(qwen_data)

            for audio_info in qwen_data:
                audio_path = Path(audio_info["audio_path"])
                audio_res = qwen_res[audio_path.as_posix()]
                if audio_res is None:
                    pass
                else:
                    poss[audio_path.stem].append(len(audio_res))
                reward_info = reward_data.get(audio_path.as_posix(), None)
                if reward_info is not None:
                    reward = reward_info["scores"]["overall"]
                    data[audio_path.stem][seed] = reward
        all_rewards[f"{temp}_{top_p}_{top_k}"] = data
        num_possibility[f"{temp}_{top_p}_{top_k}"] = poss
    # sort all_rewards by key
    all_rewards = dict(sorted(all_rewards.items(), key=lambda x: x[0], reverse=True))
    num_possibility = dict(sorted(num_possibility.items(), key=lambda x: x[0], reverse=True))
    # aggregated_rewards = {}
    # for config, data in all_rewards.items():
    #     aggregated_data = defaultdict(dict)
        
    #     # seed_groups = [['1102', '2204'], ['3306', '4408'], ['5510', '6612'], ['7714', '8816']]
    #     seed_groups = [['1102', '2204', '3306'], ['4408', '5510', '6612'], ['7714', '8816', '9918']]
    #     for audio_name, rewards in data.items():
    #         for i, seed_group in enumerate(seed_groups):
    #             scores = []
    #             for seed in seed_group:
    #                 if seed in rewards:
    #                     scores.append(rewards[seed])
    #             if len(scores) > 0:
    #                 aggregated_data[audio_name][i] = sum(scores) / len(scores)
    #     aggregated_rewards[config] = aggregated_data        
    # with open("aggregated_rewards.json", "w") as f:
    #     json.dump(aggregated_rewards, f, indent=4)
    
    # for config, data in aggregated_rewards.items():

    for config, data in all_rewards.items():
        stats = {}
        total_len = 0
        
        total_diff = 0
        total_has_diff = 0
        maxv = 0
        minv = 100
        total_scores = []
        n_poss = []
        for audio_name, rewards in data.items():
            rewards = list(rewards.values())
            max_reward = max(rewards)
            if max_reward > maxv:
                maxv = max_reward
                
            if max_reward < minv:
                minv = max_reward
                
            total_len += len(rewards)
            has_diff = diff(rewards)
            std = np.std(rewards)
            total_scores += rewards
            
            if has_diff is not None:
                total_has_diff += 1
                total_diff += has_diff
            
            n_poss += num_possibility[config][audio_name]
        # stats["total_diff"] = f"{total_diff:.2f}"
        stats["avg_diff"] = f"{total_diff / total_has_diff:.4f}"
        stats["std"] = f"{std:.4f}"
        stats["max"] = f"{maxv:.4f}"
        stats["min"] = f"{minv:.4f}"
        stats["total_std"] = f"{np.std(total_scores):.4f}"
        stats["total_avg"] = f"{np.mean(total_scores):.4f}"
        stats["n_poss"] = f"{np.mean(n_poss):.4f}"
        print(config, stats)
