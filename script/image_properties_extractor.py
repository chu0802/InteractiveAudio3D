import argparse
from pathlib import Path
from src.gptwrapper import GPTWrapper
from pathlib import Path
import json

SYSTEM_PROMPT = """
You are a visual and material reasoning assistant. Your task is to analyze named household objects and infer their likely material properties based on common real-world knowledge.

For each object, output a JSON object with two keys:
- "object": the object name as provided.
- "material_properties": a list of material types that the object is likely made of (e.g., plastic, ceramic, metal, glass, rubber, fabric, etc.).

Use only relevant and realistic material types. Focus on the most common or likely materials for each item.
"""

USER_PROMPT = "Identify the object type and its likely material properties in the provided image. Think about it step by step and response in the following format: <thinking process>, result: ```json{{\"object\": \"<object_name>\", \"material_properties\": [\"<material_property1>\", \"<material_property2>\", ...]}}```"



def main(args):
    model_name = "gemini-2.5-flash" if args.model == "gemini" else "gpt-4o"
    gpt = GPTWrapper(model_name=model_name)
    
    
    with open(args.dataset_dir / args.scene_name / "interaction_results.json", "r") as f:
        results = json.load(f)
        
    for obj_name, obj_dict in results["objects"].items():
        image_path = obj_dict["image_path"]

        response = gpt.ask(
            image=image_path,
            text=USER_PROMPT,
            system_message=SYSTEM_PROMPT,
            parse_json=True,
        )
        print(response)
        
        results["objects"][obj_name]["image_properties"] = response

    with (args.dataset_dir / args.scene_name / "overall_results.json").open("w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("datasets"))
    parser.add_argument("-s", "--scene_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
