import argparse
from pathlib import Path
from src.gptwrapper import GPTWrapper
from pathlib import Path
import json

SYSTEM_PROMPT = """
You are a knowledgeable assistant helping to generate sound-relevant human-object interaction data. First, identify 10 general, sound-producing actions that can occur in a bathroom context. Each action should be audible and applicable to multiple household objects made of plastic, metal, and ceramic.
Then, for each object in a provided list, select 3 actions from the 10 that would make an audible and realistic sound when applied to that object. For each action-object pair, write a short, clear description in the format: "<action-ing> the <material> <object>".
"""

USER_PROMPT = "Given the following list of bathroom objects, identify 10 general audible human-object interaction actions. Then for each object, pick 3 of those actions that would realistically produce sound, and generate a short descriptive phrase for each action-object pair in the format \"<action-ing> the <material> <object>\". Output everything in JSON format as specified. Input: {json_content}. Response format: ```json{{\"actions\": [\"tap\", \"squeeze\", ..., \"bang\"], \"objects\": {{\"plastic soap dispenser\": {{\"actions\": [\"tap\", \"squeeze\", \"pump\"], \"descriptions\": [\"Tapping the plastic soap dispenser\", \"Squeezing the plastic soap dispenser\", \"Pumping the plastic soap dispenser\"]}}, ...}}}}"



def main(args):
    model_name = "gemini-2.5-flash" if args.model == "gemini" else "gpt-4o"
    gpt = GPTWrapper(model_name=model_name)
    
    
    with open(args.dataset_dir / args.scene_name / "results.json", "r") as f:
        results = json.load(f)

    response = gpt.ask(
        text=USER_PROMPT.format(json_content=json.dumps(results)),
        system_message=SYSTEM_PROMPT,
        parse_json=True,
    )
    
    print(response)
    
    for image_name, obj_dict in results.items():
        obj_dict["actions"] = response["objects"][obj_dict["object_name"]]["actions"]
        obj_dict["descriptions"] = response["objects"][obj_dict["object_name"]]["descriptions"]
        obj_dict["image_path"] = args.dataset_dir / args.scene_name / "seg_img_30" / image_name
    
    results["actions"] = response["actions"]
    # for k, v in response.items():
    #     results[k]["interactions"] = v
            
    with (args.dataset_dir / args.scene_name / "interaction_results.json").open("w") as f:
        json.dump(response, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("datasets"))
    parser.add_argument("-s", "--scene_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
