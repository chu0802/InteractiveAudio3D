import argparse
from pathlib import Path
from src.gptwrapper import GPTWrapper
from pathlib import Path
from src.gptwrapper.response import ObjectRecognitionResponse, InteractionResponse, DetailedObjectRecognitionResponse
# from src.gptwrapper.config.system_prompt import INTERACTION_SYSTEM_PROMPT
import json

RECOGNITION_SYSTEM_PROMPT = "You are an expert visual recognition assistant. When shown an image, identify the main object it contains along with the material it is made of. Use clear, commonly recognized object names with their general material type (e.g., “wooden chair,” “plastic bottle,” “ceramic mug”). Be accurate but not overly technical. Focus on the most prominent object in the image. If multiple objects are present, select the most central or visually dominant one. If the material is unclear, provide your best reasonable estimate, avoiding speculation."

RECOGNITION_PROMPT = "What is the name of the main object in the image, including the material it is made of? Please provide your answer in the following format ```json\n{\"object_name\": \"<object_name with its material>\"}\n, \"object_name_without_material\": \"<object_name without its material>\", \"material\": \"<material>\"\n```. Avoid overly specific or technical terms—use clear and general descriptions."

DETAILED_RECOGNITION_PROMPT = "What is the name of the object in the image? Show me the precise name of the object and its corresponding material"
INTERACTION_PROMPT = "show me 5 different interactions to interact with {object_name} that can produce unique and interesting sounds. If the object is a static object, the format should be action + object_name. e.g., opening the door. If the object can produce sound by itself, the format should be object_name + action, e.g., dog barking"
DETAILED_INTERACTION_PROMPT = "show me 5 different interactions to interact with {material} {object_name} that can produce unique and interesting sounds. Be aware of the material of the object. If the object is a static object, the format should be action + object_name. e.g., opening the door. If the object can produce sound by itself, the format should be object_name + action, e.g., dog barking"

def main(args):
    model_name = "gemini-2.5-flash" if args.model == "gemini" else "gpt-4o"
    gpt = GPTWrapper(model_name=model_name)
    img_dir = args.dataset_dir / args.scene_name / "seg_img_30"
    
    results = {}

    for img in sorted(img_dir.iterdir()):
        response = gpt.ask(
            image=img,
            text=RECOGNITION_PROMPT,
            system_message=RECOGNITION_SYSTEM_PROMPT,
            parse_json=True,
        )
        print(response)
        results[img.name] = response
        results[img.name]["image_path"] = img.as_posix()

    with (img_dir.parent / "new_results.json").open("w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("datasets"))
    parser.add_argument("-s", "--scene_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
