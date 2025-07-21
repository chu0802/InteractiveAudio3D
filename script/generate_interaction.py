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

ADVANCED_USER_PROMPT = "For the given object: {object_name}, choose 5 actions that would realistically produce a distinctive sound. For each action-object pair, generate a short, precise descriptive phrase that includes: 1. The action (e.g., tapping, dropping, sliding) 2. The material and object name (e.g., plastic soap dispenser) 3. The instrument used to perform the action (e.g., fingertip, metal rod, rubber mallet) 4. The environment or surface involved (e.g., wooden table, ceramic tile, carpet) 5. A description in the form:\"<Action>-ing the <material> <object> with <instrument> on/against <environment>\". 6. Ensure each entry is acoustically plausible and richly contextualized for accurate sound generation. Output everything in the following example JSON format:```json[{{\"action\": \"tap\",\"material\": \"plastic\",\"object\": \"soap dispenser\",\"instrument\": \"fingertip\",\"environment\": \"marble counter\",\"description\": \"Tapping the plastic soap dispenser with a fingertip on a marble counter\"}},{{\"action\": \"squeeze\",\"material\": \"plastic\",\"object\": \"soap dispenser\",\"instrument\": \"one hand\",\"environment\": \"above a bathroom sink\",\"description\": \"Squeezing the plastic soap dispenser with one hand over a bathroom sink\"}},{{\"action\": \"drop\",\"material\": \"plastic\",\"object\": \"soap dispenser\",\"instrument\": \"hand\",\"environment\": \"tiled bathroom floor\",\"description\": \"Dropping the plastic soap dispenser from a hand onto a tiled bathroom floor\"}}, ...]```"

HANDS_ONLY_USER_PROMPT = "For the given object: {object_name}, choose 5 actions that would realistically produce a distinctive sound. For each action-object pair, generate a short, precise descriptive phrase that includes: 1. The action (e.g., tapping, squeezing, shaking) 2. The material and object name (e.g., plastic soap dispenser) 3. The instrument, which should always be a form of \"hand\", e.g., \"fingertips\", \"palm\", \"both hands\", etc. A description in the format: \"<action-ing> the <material> <object> with <instrument>\". 5. Ensure each entry is acoustically plausible, richly contextualized for accurate sound generation, and specifically tied to manual (hand-based) interaction. Output in JSON as follows: ```json[{{\"action\": \"tap\",\"material\": \"plastic\", \"object\": \"soap dispenser\", \"instrument\": \"fingertips\", \"description\": \"Tapping the plastic soap dispenser with fingertips\"}}, {{\"action\": \"squeeze\",\"material\": \"plastic\",\"object\": \"soap dispenser\",\"instrument\": \"one hand\",\"description\": \"Squeezing the plastic soap dispenser with one hand\"}},{{\"action\": \"shake\",\"material\": \"plastic\",\"object\": \"soap dispenser\",\"instrument\": \"both hands\",\"description\": \"Shaking the plastic soap dispenser with both hands\"}}, ...]```"

def main(args):
    model_name = "gemini-2.5-flash" if args.model == "gemini" else "gpt-4o"
    gpt = GPTWrapper(model_name=model_name)
    
    
    with open(args.dataset_dir / args.scene_name / "new_results.json", "r") as f:
        results = json.load(f)
    
    for image_id, image_data in results.items():
        response = gpt.ask(
            text=HANDS_ONLY_USER_PROMPT.format(object_name=image_data["object_name"]),
            system_message=SYSTEM_PROMPT,
            parse_json=True,
        )
        print(response)
        results[image_id]["interactions"] = response

    with (args.dataset_dir / args.scene_name / "hands_only_interaction_results.json").open("w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("datasets"))
    parser.add_argument("-s", "--scene_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
