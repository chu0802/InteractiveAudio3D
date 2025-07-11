import argparse
from pathlib import Path
from src.gptwrapper import GPTWrapper
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import torch

def main(args):
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to("cuda")
    
    with open(Path("datasets") / args.scene_name / "overall_results.json", "r") as f:
        info = json.load(f)
    
    for obj_name, obj_info in info["objects"].items():
        for action, prompt in zip(obj_info["actions"], obj_info["descriptions"]):
            audio_dir = Path("output") / args.scene_name / f"{obj_name.replace(' ', '_')}/{prompt.replace(' ', '_')}"
            
            with open(audio_dir / f"{args.model}_results.json", "r") as f:
                audio_results = json.load(f)
            
            rewards = {}
            
            gt_action = audio_results["action"]
            
            gt_object = info["objects"][obj_name]["image_properties"]["object"]
            gt_materials = info["objects"][obj_name]["image_properties"]["material_properties"]
            
            gt_action_emb = sentence_model.encode(gt_action, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0)
            gt_object_emb = sentence_model.encode(gt_object, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0)
            gt_materials_emb = sentence_model.encode(gt_materials, convert_to_tensor=True, normalize_embeddings=True)
            
            for audio_id, audio_res in audio_results["results"].items():
                if isinstance(audio_res, str):
                    try:
                        audio_res = json.loads(audio_res)
                    except:
                        audio_res = audio_res.replace("\n", "")
                        audio_res = audio_res.replace(" ", "")
                        alist = audio_res.split("likelihood")
                        try:
                            audio_res = [json.loads(alist[0][1:-2]+"}")]
                            for i in range(1, len(alist)-1):
                                audio_res.append(json.loads(alist[i][8:-2] + "}"))
                        except:
                            print("bad audio id: ", obj_name, prompt, audio_id)
                            continue
                max_score = 0
                max_possibility = None
                for i, possibility in enumerate(audio_res):
                    try:
                        action = possibility["action"]
                        object = possibility["object"]
                        material = possibility["material"]
                    except:
                        print("bad audio id possibility: ", obj_name, prompt, audio_id, i)
                        continue
                    if len(action) == 0 or len(object) == 0 or len(material) == 0:
                        continue
                    action_emb = sentence_model.encode(action, convert_to_tensor=True, normalize_embeddings=True)
                    object_emb = sentence_model.encode(object, convert_to_tensor=True, normalize_embeddings=True)
                    material_emb = sentence_model.encode(material, convert_to_tensor=True, normalize_embeddings=True)

                    cos_sim_action = torch.einsum("ik,jk->ij", gt_action_emb, action_emb)
                    cos_sim_object = torch.einsum("ik,jk->ij", gt_object_emb, object_emb)
                    cos_sim_material = torch.einsum("ik,jk->ij", gt_materials_emb, material_emb).max(dim=1).values
                    
                    avg_action_sim = cos_sim_action.max()
                    avg_object_sim = cos_sim_object.max()
                    avg_material_sim = cos_sim_material.mean()
                    
                    overall_score = (avg_action_sim + avg_object_sim + avg_material_sim).item()
                    
                    if overall_score > max_score:
                        max_score = overall_score
                        max_possibility = i
                
                print(f"Audio ID: {audio_id}, Max Score: {max_score}, Max Possibility: {max_possibility}")
                rewards[audio_id] = {
                    "scores": {
                        "action": avg_action_sim.item(),
                        "object": avg_object_sim.item(),
                        "material": avg_material_sim.item(),
                        "overall": overall_score
                    },
                    "max_possibility": max_possibility,
                }
            with open(audio_dir / f"{args.model}_rewards.json", "w") as f:
                json.dump(rewards, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("datasets"))
    parser.add_argument("-s", "--scene_name", type=str, default="0118_bathroom")
    args = parser.parse_args()
    main(args)
