import argparse
from pathlib import Path
from src.gptwrapper import GPTWrapper
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
        res_dict[result["audio_path"]] = result["result"]
    return res_dict


def main(args):
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to("cuda")
    
    with open(args.dataset_dir / args.scene_name / "overall_results.json", "r") as f:
        info = json.load(f)
    cnt = 0
    for obj_name, obj_info in info["objects"].items():
        audio_dir = args.output_dir / args.scene_name / f"{obj_name.replace(' ', '_')}"
        if args.target_obj and obj_name != args.target_obj:
            continue

        with open(audio_dir / f"stage_2_results.json", "r") as f:
            audio_results = result_2_list_to_dict(json.load(f))
        
        rewards = {}
        for gt_action, prompt in tqdm(zip(obj_info["actions"], obj_info["descriptions"]), desc=f"Processing {obj_name}"):
            audio_prompt_dir = (audio_dir / f"{prompt.replace(' ', '_')}")
            for audio_path in sorted(audio_prompt_dir.glob("*.wav")):
                audio_path_key = audio_path.as_posix()
                
                
                if audio_path_key == "output/0118_bathroom/ceramic_mug/Knocking_the_ceramic_mug/90845_8.wav":
                    breakpoint()
                audio_res = audio_results[audio_path_key]
                if not isinstance(audio_res, list):
                    print("bad audio id: ", audio_path_key)
                    continue

                gt_object = info["objects"][obj_name]["image_properties"]["object"]
                gt_materials = info["objects"][obj_name]["image_properties"]["material_properties"]

                gt_action_emb = sentence_model.encode(gt_action, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0)
                gt_object_emb = sentence_model.encode(gt_object, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0)
                gt_materials_emb = sentence_model.encode(gt_materials, convert_to_tensor=True, normalize_embeddings=True)

                max_overall_score = 0
                max_action_score = 0
                max_object_score = 0
                max_material_score = 0
                max_possibility = None
                for i, possibility in enumerate(audio_res):
                    try:
                        action = possibility["action"]
                        object = possibility["object"]
                        material = possibility["material"]
                    except:
                        print("bad audio id possibility: ", audio_path_key, i)
                        continue
                    if len(action) == 0 or len(object) == 0 or len(material) == 0:
                        continue

                    action_emb = sentence_model.encode(action, convert_to_tensor=True, normalize_embeddings=True)
                    object_emb = sentence_model.encode(object, convert_to_tensor=True, normalize_embeddings=True)
                    material_emb = sentence_model.encode(material, convert_to_tensor=True, normalize_embeddings=True)

                    if action_emb.ndim == 1:
                        action_emb = action_emb.unsqueeze(0)
                    if object_emb.ndim == 1:
                        object_emb = object_emb.unsqueeze(0)
                    if material_emb.ndim == 1:
                        material_emb = material_emb.unsqueeze(0)

                    cos_sim_action = torch.einsum("ik,jk->ij", gt_action_emb, action_emb)
                    cos_sim_object = torch.einsum("ik,jk->ij", gt_object_emb, object_emb)
                    cos_sim_material = torch.einsum("ik,jk->ij", gt_materials_emb, material_emb).max(dim=1).values

                    avg_action_sim = cos_sim_action.max()
                    avg_object_sim = cos_sim_object.max()
                    avg_material_sim = cos_sim_material.mean()

                    overall_score = (avg_action_sim + avg_object_sim + avg_material_sim).item()

                    if overall_score > max_overall_score:
                        max_overall_score = overall_score
                        max_action_score = avg_action_sim.item()
                        max_object_score = avg_object_sim.item()
                        max_material_score = avg_material_sim.item()
                        max_possibility = i
                        

                    rewards[audio_path_key] = {
                        "scores": {
                            "action": max_action_score,
                            "object": max_object_score,
                            "material": max_material_score,
                            "overall": max_overall_score
                        },
                        "max_possibility": max_possibility,
                    }

            with open(audio_dir / f"rewards.json", "w") as f:
                json.dump(rewards, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("output"))
    parser.add_argument("-m", "--model", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("datasets"))
    parser.add_argument("-s", "--scene_name", type=str, default="0118_bathroom")
    parser.add_argument("-t", "--target_obj", type=str, default=None)
    args = parser.parse_args()
    main(args)
