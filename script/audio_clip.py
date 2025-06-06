from src.audio_clip.model import AudioCLIP
from src.audio_clip.api import load_audio, load_image
import torch
import json

if __name__ == "__main__":
    with open("assets/index.json", "r") as f:
        data = json.load(f)

    pretrained_path = "../langsplat/ckpts/audio_clip/AudioCLIP-Full-Training.pt"
    aclp = AudioCLIP(pretrained=pretrained_path)
    aclp.to("cuda")
    aclp.eval()

    results = {}
    
    correct = 0
    total = 0
    
    for scene_name, scene_data in data.items():
        results[scene_name] = {}
        images = load_image([f"assets/{img_data['seg_image_path']}" for img_id, img_data in scene_data.items() if img_id.isdigit()])
        object_names = [img_data["object_name"] for img_id, img_data in scene_data.items() if img_id.isdigit()]
        with torch.no_grad():
            ((_, image_features, _), _), _ = aclp(image=images)
        for img_id, img_data in scene_data.items():
            if not img_id.isdigit():
                continue
            results[scene_name][img_id] = []
            for interaction in img_data["interactions"]:
                audio_path = "assets/" + interaction["audio_path"]
                audio = load_audio(audio_path)
                
                with torch.no_grad():
                    ((audio_features, _, _), _), _ = aclp(audio=audio)

                scale_audio_image = torch.clamp(aclp.logit_scale_ai.exp(), min=1.0, max=100.0)
                logits_audio_image = scale_audio_image * audio_features @ image_features.T
                confidence = logits_audio_image.softmax(dim=1)
                
                idx = int(img_id)
                
                pred = confidence.argmax(dim=1).item()
                gt = idx
                pred_name = object_names[pred]
                gt_name = object_names[gt]
                is_correct = (pred == gt) or (pred_name.lower() == gt_name.lower())
                
                results[scene_name][img_id].append({
                    "original_prompt": interaction["prompt"],
                    "confidence": confidence.detach().cpu().numpy().tolist()[0],
                    "score": logits_audio_image.detach().cpu().numpy().tolist()[0][idx],
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
    
    with open("audio_clip_res.json", "w") as f:
        json.dump(results, f, indent=4)
                
    # print(res)
    
    
    
    # audios = ["assets/" + d["audio_path"] for d in d]
    # image_path = "/home/chu980802/interactive_audio_3d/assets/0118_bathroom/seg_img/007.png"
    # audio = load_audio(audios)
    # image = load_image(image_path)
    # ((audio_features, _, _), _), _ = aclp(audio=audio)
    # ((_, image_features, _), _), _ = aclp(image=image)
    
    # print(torch.cosine_similarity(audio_features, image_features, dim=1))