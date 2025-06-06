import argparse
from pathlib import Path
from src.gptwrapper import GPTWrapper
from pathlib import Path
from src.gptwrapper.response import ObjectRecognitionResponse, InteractionResponse, DetailedObjectRecognitionResponse
from src.gptwrapper.config.system_prompt import RECOGNITION_SYSTEM_PROMPT, INTERACTION_SYSTEM_PROMPT
import json

RECOGNITION_PROMPT = "What is the name of the object in the image? Show me the precise name of the object"
DETAILED_RECOGNITION_PROMPT = "What is the name of the object in the image? Show me the precise name of the object and its corresponding material"
INTERACTION_PROMPT = "show me 5 different interactions to interact with {object_name} that can produce unique and interesting sounds. If the object is a static object, the format should be action + object_name. e.g., opening the door. If the object can produce sound by itself, the format should be object_name + action, e.g., dog barking"
DETAILED_INTERACTION_PROMPT = "show me 5 different interactions to interact with {material} {object_name} that can produce unique and interesting sounds. Be aware of the material of the object. If the object is a static object, the format should be action + object_name. e.g., opening the door. If the object can produce sound by itself, the format should be object_name + action, e.g., dog barking"

def main(args):
    gpt = GPTWrapper(model_name="gemini-2.5-flash-preview-05-20")
    img_dir = Path("output") / args.dataset / args.mode / "intermediate_results" / f"{args.image_id:05d}" / "seg_img"
    
    results = {}

    for img in sorted(img_dir.iterdir()):
        response = gpt.ask(
            image=img,
            text=RECOGNITION_PROMPT,
            system_message=RECOGNITION_SYSTEM_PROMPT,
            response_format=DetailedObjectRecognitionResponse,
        )

        interaction_res = gpt.ask(
            text=DETAILED_INTERACTION_PROMPT.format(object_name=response.object_name, material=response.material),
            system_message=INTERACTION_SYSTEM_PROMPT,
            response_format=InteractionResponse,
        )

        results[img.name] = {
            "image_path": img.as_posix(),
            "object_name": response.object_name,
            "material": response.material,
            "interactions": interaction_res.to_list(),
        }

        with (img_dir.parent / "detailed_results.json").open("w") as f:
            json.dump(results, f, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--image_id", type=int, default=0)
    args = parser.parse_args()
    main(args)
