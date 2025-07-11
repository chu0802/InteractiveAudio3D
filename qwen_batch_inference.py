import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.feature_extraction_utils import BatchFeature
from qwen_omni_utils import process_mm_info
from functools import partial
from accelerate import PartialState
from accelerate.utils import gather_object


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


# 1) Prepare an audio-only Dataset
class AudioDataset(Dataset):
    def __init__(self, audio_paths, prompt, system_prompt):
        self.audio_paths = audio_paths
        self.prompt = prompt
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]

        return prepare_prompt(
            audio=audio_path, 
            text=self.prompt, 
            system_prompt=self.system_prompt
        )

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

def main():
    distributed_state = PartialState()
    # 3) Init Accelerator and load model & processor across GPUs
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.bfloat16,
        device_map=distributed_state.device,
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    with open("audio_paths.json", "r") as f:
        audio_paths = json.load(f)

    system_prompt = """
    You are an expert audio‐forensics analyst with deep knowledge of acoustics, material science, and everyday physical interactions. You excel at rigorous, step-by-step reasoning and at extracting clean, valid JSON.  
    """

    cot_stage_1_prompt = "Analyze the sound and reason through what may have caused it. For each sound, consider all plausible interpretations—what action might have produced it, what object was involved, and what materials the object likely had. Think through each possibility step by step and rank them by the likelihood from 1 to 10. Use the following format for each possible explanation: Possibility 1: ```json{thinking_process: <thinking process>, description: 'The sound could be caused by <action> a <material> <object>.', Likelihood: 'x/10'}```, Possibility 2: ```json{...}```, ..."

    cot_stage_2_prompt = "Given the audio input and the reasoning process below: {thinking_res}. For each possibility, extract the action, the object involved, as well as the involved object’s likely materials. Present your response in the following JSON format: ```json[{{action: [action1 (only the action, no other words), ...], object: [object1, ...], material: [material1, ...], likelihood: 'x/10'}}, {{action: [action2, ...], object: [object2, ...], material: [material2, ...], likelihood: 'y/10'}}, ...]```"

    batch_size = 32
    preprocessed_batched_inputs = [
        preprocessing_fn(
            processor, 
            audio_paths=audio_paths, 
            prompts=cot_stage_1_prompt, 
            system_prompt=system_prompt,
        ) 
        for audio_paths in batchify_inputs(audio_paths, batch_size)
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
                temperature=0.7,
            )
            
            text = processor.batch_decode(output.sequences[:, batch["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            stage_1_results += text

    stage_1_gather = gather_object(stage_1_results)
    thinking_res = stage_1_gather[: len(audio_paths)]

    if distributed_state.is_main_process:
        with open("thinking_process.json", "w") as f:
            json.dump([{"audio_path": a, "result": jsonify(t)} for t, a in zip(thinking_res, audio_paths)], f, indent=4)
    
    stage_2_prompts = [cot_stage_2_prompt.format(thinking_res=thinking_res[i]) for i in range(len(thinking_res))]
    # stage 2
    preprocessed_batched_inputs = [
        preprocessing_fn(
            processor, 
            prompts=prompts, 
            system_prompt=system_prompt,
        ) 
        for prompts in batchify_inputs(stage_2_prompts, batch_size)
    ]
    
    stage_2_results = []
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
                temperature=0.7,
            )
            
            text = processor.batch_decode(output.sequences[:, batch["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            stage_2_results += text
    stage_2_gather = gather_object(stage_2_results)
    
    stage_2_res = stage_2_gather[: len(stage_2_prompts)]
    
    if distributed_state.is_main_process:
        with open("stage_2_results.json", "w") as f:
            json.dump([{"audio_path": a, "result": jsonify(t)} for t, a in zip(stage_2_res, audio_paths)], f, indent=4)

if __name__ == "__main__":
    main()
