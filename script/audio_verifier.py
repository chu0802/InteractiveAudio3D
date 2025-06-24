from src.gptwrapper import GPTWrapper
from src.gptwrapper.response import AudioUnderstandingResponse, GeneralResponse
import json
from pathlib import Path

INSTRUCTION = "This audio file contains an action being performed on an object. Please describe the action, the object it is performed on, and the material of that object. Then, provide a brief justification of the audio."

DESCRIPTION_SYSTEM_PROMPT = "You are a helpful assistant that can describe the sound in the audio."

if __name__ == "__main__":
    # audio_paths = ["assets/" + d["audio_path"] for d in d]
    # image_path = "/home/chu980802/interactive_audio_3d/assets/0118_bathroom/seg_img/007.png"
    
    scene_name = "0118_bathroom"
    gpt = GPTWrapper(model_name="gemini-2.5-flash-preview-04-17")
    
    with open("assets/results.json", "r") as f:
        data = json.load(f)[scene_name]
        
    results = {}
        
    for img_id in data.keys():
        if not img_id.isdigit():
            continue
        img_data = data[img_id]
        img_path = "assets/" + img_data["seg_image_path"]
        
        results[img_id] = []
        
        for interaction_id in range(1, 6):
            
            interaction_prompt = img_data["interactions"][interaction_id]["prompt"]
            audio_path = "assets/" + img_data["interactions"][interaction_id]["audio_path"]
            
            system_prompt = "You are a helpful assistant that can describe the sound in the audio."
            prompt = f"Recognize the object in the image. Considering its shape, size, material, and texture, and imagine interacting with the object in real world with the given interaction prompt: {interaction_prompt}. Verify if the provided audio demonstrates the correct sound of the interaction with the object in the image, given the object's properties. First answer yes/no, then provide a brief justification of your answer."
            
            
            res = gpt.ask(
                audio=audio_path,
                image=img_path,
                text=prompt,
                system_message=system_prompt,
                response_format=AudioUnderstandingResponse
            )
            
            results[img_id].append({
                "original_prompt": interaction_prompt,
                "audio_understanding": res.model_dump()
            })
            
        with open(f"assets/{scene_name}/gemini_res.json", "w") as f:
            json.dump(results, f, indent=4)
