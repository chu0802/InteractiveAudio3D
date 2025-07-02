import argparse
import json
from pathlib import Path
from script.audio_verifier import verify_audio
from src.gptwrapper import GPTWrapper
from src.gptwrapper.response import ImageDecompositionResponse, AudioVerificationResponse, GeneralResponse, RecognitionInteractionResponse, AudioUnderstandingCandidateResponse, ImageUnderstandingResponse
from script.stable_audio import generate_audio
from diffusers import StableAudioPipeline
import soundfile as sf
import torch

image_understanding_prompt = "Considering the object in the provided image, identify the object, and infer the object's possible materials and physical properties."


audio_understanding_candidate_prompt = "Given an audio clip of a sound, identify what kind of action likely caused the sound, the object involved, and infer the object's possible materials and physical properties. List all potential action, the object involved, and the object’s likely material and physical properties."

image_decomposition_prompt = "Considering the provided object's material and physical properties. First list the properties of the object in the provided image, then list a checklist to check if an provided audio perfectly and preciously simulate the sound of {interaction_prompt} and reflect the object's properties."

audio_verification_prompt = "Given the provided audio and the checklist: {checklist}, verify if the audio perfectly and preciously simulate the sound of {interaction_prompt} and reflect the object's properties. Answering yes/no for each item in the checklist, finally provide a brief justification of your answer."

interaction_justification_prompt = "Given an object in the provided image, identify the provided object's name and its properties, and provide five plausible prompts describing an interaction with that object that will make unique sound, the interaction should physically plausible and consistent with how the object would naturally be used in the real world. The prompt should clearly mention the object's name and its properties, and the plausible interaction should be simple, concise and clear."

def main(args):
    gpt = GPTWrapper(model_name="gemini-2.5-flash")
    
    with open("assets/results.json", "r") as f:
        data = json.load(f)[args.scene_name][f"{args.image_id:03d}"]

    
    img_path = "assets/" + data["seg_image_path"]

    justification = gpt.ask(
        image=img_path,
        text=interaction_justification_prompt,
        response_format=RecognitionInteractionResponse
    )
    print(justification)
    
    output_dir = Path("output") / args.scene_name / f"{args.image_id:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "interactions.json", "w") as f:
        json.dump(justification.model_dump(), f, indent=4)
    
    img_property = gpt.ask(
        image=img_path,
        text=image_understanding_prompt,
        response_format=ImageUnderstandingResponse
    )
    
    with open(output_dir / "image_properties.json", "w") as f:
        json.dump(img_property.model_dump(), f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="0118_bathroom")
    parser.add_argument("--image_id", type=int, default=0)
    parser.add_argument("--interaction_id", type=int, default=4)
    args = parser.parse_args()
    
    main(args)
