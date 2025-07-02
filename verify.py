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

interaction_justification_prompt = "Given an object in the provided image, identify the provided object's name and its properties, and provide five plausible prompts describing an interaction with that object that will make unique sound, the interaction should physically plausible and consistent with how the object would naturally be used in the real world. The prompt should clearly mention the object's name and its properties, and the plausible interaction should be concise and clear."

def main(args):
    gpt = GPTWrapper(model_name="gemini-2.5-flash")
    
    with open("assets/results.json", "r") as f:
        data = json.load(f)[args.scene_name][f"{args.image_id:03d}"]

    
    img_path = "assets/" + data["seg_image_path"]

    output_dir = Path("output") / args.scene_name / f"{args.image_id:03d}"
    
    with open(output_dir / "interactions.json", "r") as f:
        interactions = json.load(f)
        
    interaction_prompt = interactions["interactions"][args.interaction_id]

    audio_dir = Path("output") / args.scene_name / f"{args.image_id:03d}" / f"{'_'.join(interaction_prompt.split(' '))}"
    audio_paths = sorted(list(audio_dir.glob("*.wav")))
    
    results = {}

    for audio_path in audio_paths:
        res = gpt.ask(
            audio=audio_path,
            text=audio_understanding_candidate_prompt,
            response_format=AudioUnderstandingCandidateResponse
        )
        results[audio_path.stem] = res.model_dump()

        with (audio_dir / "audio_understanding_candidates.json").open("w") as f:
            json.dump(results, f, indent=4)
     
    
    # audio_dir = output_dir / f"{'_'.join(interaction_prompt.split(' '))}"
    
    # audio_paths = sorted(list(audio_dir.glob("*.wav")))
    
    # results = {}
    # for audio_path in audio_paths:
    #     res = verify_audio(gpt, audio_path, img_path, interaction_prompt)
    #     results[audio_path.stem] = res.model_dump()

    #     with (audio_dir / "results.json").open("w") as f:
    #         json.dump(results, f, indent=4)


    # # interaction_prompt = data["interactions"][args.interaction_id]["prompt"]
    
    # justification = gpt.ask(
    #     image=img_path,
    #     text=interaction_justification_prompt,
    #     response_format=RecognitionInteractionResponse
    # )
    # print(justification)
    
    # pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")
    
    # output_dir = Path("output") / args.scene_name / f"{args.image_id:03d}"
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    # for interaction in justification.interactions:
    #     seed = 1102
    #     audio = generate_audio(pipe, interaction, duration=5.0, seed=seed)
    #     audio_path = output_dir / f"{'_'.join(interaction.split(' '))}/{seed}.wav"
    #     audio_path.parent.mkdir(parents=True, exist_ok=True)
    #     sf.write(audio_path, audio, pipe.vae.sampling_rate)
    
    # with open(output_dir / "interactions.json", "w") as f:
    #     json.dump(justification.model_dump(), f, indent=4)
    # criteria = gpt.ask(
    #     image=img_path,
    #     text=image_decomposition_prompt.format(interaction_prompt=interaction_prompt),
    #     response_format=ImageDecompositionResponse
    # )
    
    # print(criteria)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="0118_bathroom")
    parser.add_argument("--image_id", type=int, default=0)
    parser.add_argument("--interaction_id", type=int, default=4)
    args = parser.parse_args()
    
    main(args)
