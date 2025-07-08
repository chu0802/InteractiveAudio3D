from src.qwen.api import QwenAPI, compose_messages
from src.gptwrapper.api import GPTWrapper
from src.gptwrapper.response import GeneralResponse
import argparse

if __name__ == "__main__":
    audio_path = "/home/chuyu/vllab/interactive_audio_3d/output/0118_bathroom/000/dropping_Dial_hand_soap/1105.wav"
    
    naive_text_prompt = "Given an audio recording of a sound, determine the likely cause by identifying the action that produced it, the object involved, and the object's possible materials and physical characteristics. Provide multiple possible actions, the object, and its likely materials using the following JSON structure: ```json{{action: [action1 (only the action, no other words), action2, action3, ...], object: [object1, ...], material: [material1, ...], likelihood: x/10}}```"

    thinking_text = "Analyze the sound and reason through what may have caused it. For each sound, consider all plausible interpretations—what action might have produced it, what object was involved, and what materials the object likely had. Think through each possibility step by step and rank them by the likelihood from 1 to 10. Use the following format for each possible explanation: Possibility 1: ```json{thinking_process: <thinking process>, description: 'The sound could be caused by [action] a [material] [object].', Likelihood: 'x/10'}```, Possibility 2: {...}, ..."

    cot_stage_2_prompt = "Given the audio input and the reasoning process below: {thinking_res}. For each possibility, extract the action, the object involved, as well as the involved object’s likely materials. Present your response in the following JSON format: ```json[{{action: [action1 (only the action, no other words), ...], object: [object1, ...], material: [material1, ...], likelihood: x/10}}, {{action: [action2, ...], object: [object2, ...], material: [material2, ...], likelihood: y/10}}, ...]```"
    
    
    audio_verification_prompt = "Given the provided audio and the checklist: {checklist}, verify if the audio perfectly and preciously simulate the sound of {interaction_prompt} and reflect the object's properties. Answering yes/no for each item in the checklist, finally provide a brief justification of your answer."

    check_list_prompt = "List a checklist to check if an provided audio perfectly and preciously simulate the sound of {interaction_prompt} and reflect the object's properties."
    # checklist = "1. Does the audio include the primary impact sound of a metal can hitting a surface? 2. Does the audio perfectly and preciously simulate the sound of knocking a metal can onto a wooden table and reflect the object's properties?"

    interaction_prompt = "knocking a metal can onto a wooden table"
    gpt = GPTWrapper(model_name="gemini-2.5-flash")
    
    check_list = gpt.ask(text=check_list_prompt.format(interaction_prompt=interaction_prompt)).response
    print(check_list)
    res = gpt.ask(audio=audio_path, text=audio_verification_prompt.format(checklist=check_list, interaction_prompt=interaction_prompt)).response
    # cot = True
    # seed = 1102
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", type=str, default="qwen")
    # args = parser.parse_args()
    
    # if args.model == "qwen":
    #     qwen = QwenAPI()
    #     if not cot:
    #         res = qwen.ask(audio_path=audio_path, text=naive_text_prompt, is_json=True)
    #     else:
    #         thinking_res = qwen.ask(audio_path=audio_path, text=thinking_text, is_json=False, seed=seed)
    #         print(thinking_res)
    #         res = qwen.ask(audio_path=audio_path, text=cot_stage_2_prompt.format(thinking_res=thinking_res), is_json=True, seed=seed)
        
    # elif args.model == "gemini":
    #     gpt = GPTWrapper(model_name="gemini-2.5-flash")
    #     if not cot:
    #         res = gpt.ask(audio=audio_path, text=naive_text_prompt, response_format=GeneralResponse).response
    #     else:
    #         thinking_res = gpt.ask(audio=audio_path, text=thinking_text, response_format=GeneralResponse).response
    #         print(thinking_res)
    #         res = gpt.ask(audio=audio_path, text=cot_stage_2_prompt.format(thinking_res=thinking_res), response_format=GeneralResponse).response
    
    print("-" * 20)
    print(res)
