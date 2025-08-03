import argparse
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# this is a sub-optimal function due to the imperfect json format of result2
# we should fix the format of result2 in the future
def result_2_list_to_dict(result_list):
    res_dict = {}
    for result in result_list:
        info = result["result"][0] if isinstance(result["result"], list) else result["result"]
        try:
            res_dict[result["audio_path"]] = [json.loads(res["Attributes"]) if isinstance(res["Attributes"], str) else res["Attributes"] for res in info.values()]
        except:
            res_dict[result["audio_path"]] = None
    return res_dict


def main(args):
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to("cuda")
    
    with open(args.dataset_dir / args.scene_name / "hands_only_interaction_results.json", "r") as f:
        info = json.load(f)
    cnt = 0
    for img_id, obj_info in info.items():
        obj_name = obj_info["object_name"]
        audio_dir = args.output_dir / args.scene_name / f"{obj_name.replace(' ', '_')}" / f"iter{args.iter}"
        if args.target_obj and obj_name.replace(" ", "_") != args.target_obj.replace(" ", "_"):
            continue

        with open(audio_dir / f"improved_stage_1_results_{args.temperature}_{args.top_p}_{args.top_k}_{args.random_seed}.json", "r") as f:
            audio_results = result_2_list_to_dict(json.load(f))
        
        rewards = {}
        for interaction_info in tqdm(obj_info["interactions"], desc=f"Processing {obj_name}", total=len(obj_info["interactions"])):
            gt_action = interaction_info["action"]
            prompt = interaction_info["description"]

            audio_prompt_dir = (audio_dir / f"{prompt.replace(' ', '_')}")
            for audio_path in sorted(audio_prompt_dir.glob("*.wav")):
                audio_path_key = audio_path.as_posix()

                audio_res = audio_results[audio_path_key]
                if audio_res is None:
                    print("bad audio id: ", audio_path_key)
                    continue

                gt_action_emb = sentence_model.encode(gt_action, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0)
                gt_object_emb = sentence_model.encode(obj_name, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0)

                max_overall_score = 0
                max_action_score = 0
                max_object_score = 0
                max_possibility = None
                for i, possibility in enumerate(audio_res):
                    try:
                        action = possibility.get("action", possibility.get("Action", None))
                        object = possibility.get("object", possibility.get("Object", None))
                    except:
                        print("bad audio id possibility: ", audio_path_key, i)
                        continue
                    if action is None or object is None:
                        print("bad audio id possibility: ", audio_path_key, i)
                        continue

                    action_emb = sentence_model.encode(action, convert_to_tensor=True, normalize_embeddings=True)
                    object_emb = sentence_model.encode(object, convert_to_tensor=True, normalize_embeddings=True)

                    if action_emb.ndim == 1:
                        action_emb = action_emb.unsqueeze(0)
                    if object_emb.ndim == 1:
                        object_emb = object_emb.unsqueeze(0)

                    cos_sim_action = torch.einsum("ik,jk->ij", gt_action_emb, action_emb)
                    cos_sim_object = torch.einsum("ik,jk->ij", gt_object_emb, object_emb)

                    avg_action_sim = cos_sim_action.max()
                    avg_object_sim = cos_sim_object.max()

                    overall_score = (avg_action_sim + avg_object_sim).item()

                    if overall_score > max_overall_score:
                        max_overall_score = overall_score
                        max_action_score = avg_action_sim.item()
                        max_object_score = avg_object_sim.item()
                        max_possibility = i
                        

                    rewards[audio_path_key] = {
                        "scores": {
                            "action": max_action_score,
                            "object": max_object_score,
                            "overall": max_overall_score
                        },
                        "max_possibility": max_possibility,
                    }

            with open(audio_dir / f"rewards{args.temperature}_{args.top_p}_{args.top_k}_{args.random_seed}.json", "w") as f:
                json.dump(rewards, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("logs/audios"))
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("datasets"))
    parser.add_argument("-s", "--scene_name", type=str, default="0118_bathroom")
    parser.add_argument("-t", "--target_obj", type=str, default=None)
    parser.add_argument("-i", "--iter", type=int, default=0)
    parser.add_argument("-c", "--temperature", type=float, default=0.7)
    parser.add_argument("-p", "--top_p", type=float, default=1.0)
    parser.add_argument("-k", "--top_k", type=int, default=50)
    parser.add_argument("-r", "--random_seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
