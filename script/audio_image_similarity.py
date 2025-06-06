# from src.audio_clip.model import AudioCLIP
# from src.audio_clip.api import load_audio, load_image
# import torch
from src.gptwrapper import GPTWrapper
from src.gptwrapper.response import AudioUnderstandingResponse
import json

INSTRUCTION = "This audio file contains an action being performed on an object. Please describe the action, the object it is performed on, and the material of that object. Then, provide a brief justification of the audio."

SYSTEM_PROMPT = "You are a helpful assistant that can describe the sound in the audio."

if __name__ == "__main__":
    # audio_paths = ["assets/" + d["audio_path"] for d in d]
    # image_path = "/home/chu980802/interactive_audio_3d/assets/0118_bathroom/seg_img/007.png"
    
    with open("assets/index.json", "r") as f:
        data = json.load(f)

    gpt = GPTWrapper(model_name="gemini-2.5-flash-preview-04-17")

    results = {}
    
    for scene_name, scene_data in data.items():
        results[scene_name] = {}
        for img_id, img_data in scene_data.items():
            if not img_id.isdigit():
                continue
            results[scene_name][img_id] = []
            for interaction in img_data["interactions"]:
                audio_path = "assets/" + interaction["audio_path"]
                res = gpt.ask(
                    audio=audio_path,
                    text=INSTRUCTION,
                    system_message=SYSTEM_PROMPT,
                    response_format=AudioUnderstandingResponse
                )
                
                results[scene_name][img_id].append({
                    "original_prompt": interaction["prompt"],
                    "audio_understanding": res.model_dump()
                })
                
                with open("overall_res.json", "w") as f:
                    json.dump(results, f, indent=4)
                
    # print(res)
    
    # pretrained_path = "../langsplat/ckpts/audio_clip/AudioCLIP-Full-Training.pt"
    # aclp = AudioCLIP(pretrained=pretrained_path)
    # aclp.to("cuda")
    # aclp.eval()
    
    # audios = ["assets/" + d["audio_path"] for d in d]
    # image_path = "/home/chu980802/interactive_audio_3d/assets/0118_bathroom/seg_img/007.png"
    # audio = load_audio(audios)
    # image = load_image(image_path)
    # ((audio_features, _, _), _), _ = aclp(audio=audio)
    # ((_, image_features, _), _), _ = aclp(image=image)
    
    # print(torch.cosine_similarity(audio_features, image_features, dim=1))