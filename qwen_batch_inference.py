import json
import torch
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import set_seed
from qwen_omni_utils import process_mm_info
from accelerate import PartialState
from accelerate.utils import gather_object
import argparse
from pathlib import Path

def prepare_prompt(audio=None, text=None, system_prompt=None):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    content = []
    if audio:
        content.append({"type": "audio", "audio": audio})
    if text:
        content.append({"type": "text", "text": text})
    msgs.append({"role": "user", "content": content})

    return msgs

def preprocessing_fn(processor, audio_paths=None, prompts=None, system_prompt=None):
    messages = []
    if prompts and not isinstance(prompts, list):
        prompts = [prompts]
    
    for i in range(max(len(audio_paths) if audio_paths else 0, len(prompts) if prompts else 0)):
        audio_path = audio_paths[i] if audio_paths else None
        prompt = prompts[i%len(prompts)]
        messages.append(prepare_prompt(audio=audio_path, text=prompt, system_prompt=system_prompt))

    texts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    audios, images, videos = process_mm_info(
        messages,
        use_audio_in_video=True,
    )
    inputs = processor(
        text=texts,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    
    return inputs

def batchify_inputs(data, batch_size=10):
    return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    
def jsonify(texts):
    try:
        if isinstance(texts, list):
            return [json.loads(t.split("```json")[1].replace("```", "")) for t in texts]
        else:
            return json.loads(texts.split("```json")[1].replace("```", ""))
    except:
        return texts


def main(args):
    distributed_state = PartialState()
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.bfloat16,
        device_map=distributed_state.device,
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    
    for audio_dir in sorted((args.audio_dir / args.scene_name).iterdir()):
        if (args.target_obj is not None) and (args.target_obj != audio_dir.name):
            continue
            
        audio_dir = audio_dir / f"iter{args.iter}"

        audio_paths = [
            audio_path.as_posix()
            for audio_path in audio_dir.glob("*/*.wav")
        ]
        
        if distributed_state.is_main_process:
            print(f"Processing {audio_dir}... Total number of audio files: {len(audio_paths)}")

        system_prompt = """
        You are an expert audio‐forensics analyst with deep knowledge of acoustics, material science, and everyday physical interactions. You excel at rigorous, step-by-step reasoning and at extracting clean, valid JSON.  
        """
        
        improved_cot_prompt = "Analyze the audio and reason through what may have caused it. All sounds are generated only through human hands interacting with objects. Consider all plausible interpretations. For each plausible explanation, include: 1. A step-by-step thinking process. 2. A natural description sentence: \"The sound could be caused by <action> a <material> <object_name> with hands\". - <action> must be a common action word to interact with the <material> <object_name>. - <material> must be a common material word. - <object_name> must be a specific, common object. avoid use vague terms like: \"object\", \"item\", \"thing\", \"stuff\", \"material object\", or similar generic placeholders. If unsure, GUESS the most plausible common object based on the sound characteristics. 3. A structured attributes JSON: ```json{{\"action\": \"<only the verb>\",\"material & object name\": \"<material> <object_name>\",\"instrument\": \"hands\"}}```. Output all possibilities in this format: ```json{{\"Possibility 1\": {{\"Thinking stage\": <step-by-step reasoning>, \"Description\": \"The sound could be caused by <action> a <material> <object_name> with hands\", \"Attributes\": <structured attributes JSON>}}, \"Possibility 2\": {{...}}, ...}}}```"

        # improved_cot_prompt = "Analyze the audio and reason through what may have caused it. All sounds are generated only through human hands interacting with objects — using parts like hands, fingertips, palms. Consider all plausible interpretations. For each plausible explanation, include: 1. A step-by-step thinking process. 2. A natural description sentence: \"The sound could be caused by <action> a <material> <object> with <part of the hand>\". 3. A structured attributes JSON: ```json{{\"action\": \"<only the verb>\",\"object\": \"<material> <object>\",\"instrument\": \"<part of the hand>\"}}```. Output all possibilities in this format: ```json{{\"Possibility 1\": {{\"Thinking stage\": <step-by-step reasoning>, \"Description\": \"The sound could be caused by <action> a <material> <object> with <part of the hand>\", \"Attributes\": <structured attributes JSON>}}, \"Possibility 2\": {{...}}, ...}}}```"

        preprocessed_batched_inputs = [
            preprocessing_fn(
                processor, 
                audio_paths=audio_paths, 
                prompts=improved_cot_prompt, 
                system_prompt=system_prompt,
            ) 
            for audio_paths in batchify_inputs(audio_paths, args.batch_size)
        ]
        
        stage_1_results = []
        
        # stage 1
        
        with distributed_state.split_between_processes(preprocessed_batched_inputs, apply_padding=True) as batched_prompts:
            for batch in batched_prompts:
                batch = batch.to(distributed_state.device)
                output = model.generate(
                    **batch, 
                    use_audio_in_video=True,
                    max_new_tokens=1024,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    do_sample=True,
                    return_dict_in_generate=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                
                text = processor.batch_decode(output.sequences[:, batch["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                stage_1_results += text

        stage_1_gather = gather_object(stage_1_results)
        thinking_res = stage_1_gather[: len(audio_paths)]

        if distributed_state.is_main_process:
            with open(audio_dir / f"improved_stage_1_results.json", "w") as f:
                json.dump([{"audio_path": a, "result": jsonify(t)} for t, a in zip(thinking_res, audio_paths)], f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio_dir", type=Path, default=Path("logs/audios"))
    parser.add_argument("-s", "--scene_name", type=str, default="0118_bathroom")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-t", "--target_obj", type=str, default=None)
    parser.add_argument("-c", "--temperature", type=float, default=1.0)
    parser.add_argument("-p", "--top_p", type=float, default=1.0)
    parser.add_argument("-k", "--top_k", type=int, default=1)
    parser.add_argument("-i", "--iter", type=int, default=0)
    parser.add_argument("-r", "--random_seed", type=int, default=None)
    args = parser.parse_args()
    
    if args.target_obj is not None:
        args.target_obj = args.target_obj.replace(" ", "_")
    
    set_seed(args.random_seed)
    main(args)
