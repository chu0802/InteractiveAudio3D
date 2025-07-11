import argparse
from pathlib import Path
from src.gptwrapper import GPTWrapper
from pathlib import Path
import json

system_prompt = """
You are an expert audio‐forensics analyst with deep knowledge of acoustics, material science, and everyday physical interactions. You excel at rigorous, step-by-step reasoning and at extracting clean, valid JSON.  
"""

cot_stage_1_prompt = "Analyze the sound and reason through what may have caused it. For each sound, consider all plausible interpretations—what action might have produced it, what object was involved, and what materials the object likely had. Think through each possibility step by step and rank them by the likelihood from 1 to 10. Use the following format for each possible explanation: Possibility 1: ```json{thinking_process: <thinking process>, description: 'The sound could be caused by [action] a [material] [object].', Likelihood: 'x/10'}```, Possibility 2: {...}, ..."

cot_stage_2_prompt = "Given the audio input and the reasoning process below: {thinking_res}. For each possibility, extract the action, the object involved, as well as the involved object’s likely materials. Present your response in the following JSON format: ```json[{{action: [action1 (only the action, no other words), ...], object: [object1, ...], material: [material1, ...], likelihood: x/10}}, {{action: [action2, ...], object: [object2, ...], material: [material2, ...], likelihood: y/10}}, ...]```"


def main(args):
    if args.model == "openai":
        gpt = GPTWrapper(model_name="gpt-4o-audio-preview")
    elif args.model == "gemini":
        gpt = GPTWrapper(model_name="gemini-2.5-flash")
    
    with open(args.dataset_dir / args.scene_name / "overall_results.json", "r") as f:
        results = json.load(f)
    
    for obj_name, obj_info in results["objects"].items():
        for action, prompt in zip(obj_info["actions"], obj_info["descriptions"]):
            if prompt == "Tapping the plastic soap dispenser":
                continue
            results = {}
            results["action"] = action
            results["prompt"] = prompt
            results["results"] = {}
            audio_dir = Path("output") / args.scene_name / f"{obj_name.replace(' ', '_')}/{prompt.replace(' ', '_')}"
            for audio_path in sorted(audio_dir.iterdir()):
                thinking_res = gpt.ask(audio=audio_path, text=cot_stage_1_prompt)
                res = gpt.ask(audio=audio_path, text=cot_stage_2_prompt.format(thinking_res=thinking_res), parse_json=True)
                print(res)
                results["results"][audio_path.stem] = res
                with open(audio_dir / f"{args.model}_results.json", "w") as f:
                    json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("-d", "--dataset_dir", type=Path, default=Path("datasets"))
    parser.add_argument("-s", "--scene_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
