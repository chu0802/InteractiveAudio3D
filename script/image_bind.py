from src.imagebind.models import imagebind_model
from src.imagebind.models.imagebind_model import ModalityType
from src.imagebind import data as imagebind_data

import torch
import json

if __name__ == "__main__":
    with open("assets/index.json", "r") as f:
        data = json.load(f)
    model = imagebind_model.imagebind_huge(pretrained=True)

    model.eval()
    model.to("cuda")

    results = {}
    correct = 0
    total = 0
    
    for scene_name, scene_data in data.items():
        results[scene_name] = {}
        image_paths = [f"assets/{img_data['seg_image_path']}" for img_id, img_data in scene_data.items() if img_id.isdigit()]
        object_names = [img_data["object_name"] for img_id, img_data in scene_data.items() if img_id.isdigit()]
        
        image_data = imagebind_data.load_and_transform_vision_data(image_paths, "cuda")
        
        for img_id, img_data in scene_data.items():
            if not img_id.isdigit():
                continue
            results[scene_name][img_id] = []
            for interaction in img_data["interactions"]:
                audio_paths = ["assets/" + interaction["audio_path"]]
                
                inputs = {
                    ModalityType.VISION: image_data,
                    ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data(audio_paths, "cuda"),
                }

                with torch.no_grad():
                    embeddings = model(inputs)
                
                normalized_audio = embeddings[ModalityType.AUDIO] / embeddings[ModalityType.AUDIO].norm(dim=1, keepdim=True)
                normalized_vision = embeddings[ModalityType.VISION] / embeddings[ModalityType.VISION].norm(dim=1, keepdim=True)
                scores = normalized_audio @ normalized_vision.T

                confidence = scores.softmax(dim=1)

                idx = int(img_id)
                
                pred = confidence.argmax(dim=1).item()
                gt = idx
                pred_name = object_names[pred]
                gt_name = object_names[gt]
                is_correct = (pred == gt) or (pred_name.lower() == gt_name.lower())
                
                results[scene_name][img_id].append({
                    "original_prompt": interaction["prompt"],
                    "confidence": confidence.detach().cpu().numpy().tolist()[0],
                    "score": scores.detach().cpu().numpy().tolist()[0][idx],
                    "predicted_image_id": confidence.argmax(dim=1).item(),
                    "ground_truth_image_id": idx,
                    "predicted_object_name": object_names[confidence.argmax(dim=1).item()],
                    "ground_truth_object_name": object_names[idx],
                    "ground_truth_confidence": confidence[0][idx].item(),
                    "predicted_confidence": confidence[0][pred].item(),
                    "is_correct": is_correct
                })

                if is_correct:
                    correct += 1
                total += 1

    print(f"# Correct: {correct}, # Total: {total}, Accuracy: {correct / total}")
    
    with open("image_bind_res_norm.json", "w") as f:
        json.dump(results, f, indent=4)
