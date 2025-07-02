import os
import base64
import json
from openai import OpenAI
from src.gptwrapper.response import AudioUnderstandingCandidateResponse


def base64_encode(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def get_text_response(completion):
    response = "".join([chunk.choices[0].delta.content for chunk in completion if chunk.choices and chunk.choices[0].delta.content is not None])
    return response

def parse_json_response(response):
    if response.startswith("```json"):
        response = response[len("```json"):]
    if response.endswith("```"):
        response = response[:-len("```")]
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response")
    
class QwenAPI:
    def __init__(self):
        self.model = "qwen2.5-omni-7b"
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )   
        
    def get_text_response(self, completion):
        return get_text_response(completion)
    
    def parse_json_response(self, response):
        return parse_json_response(response)
    
    def ask(self, audio_path=None, image_path=None, text=None, is_json=True):
        messages = compose_messages(audio_path=audio_path, image_path=image_path, text=text)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            modalities=["text"],
            stream=True,
        )
        res = self.get_text_response(completion)

        if is_json:
            return self.parse_json_response(res)
        else:
            return res

def compose_messages(audio_path=None, image_path=None, text=None):
    if (audio_path is None and image_path is None and text is None):
        raise ValueError("At least one of audio_path, image_path, or text must be provided")
    
    if (audio_path is not None and image_path is not None):
        raise ValueError("Only one of audio_path or image_path can be provided")

    contents = []

    if audio_path:
        # check if audio_path is a url
        if audio_path.startswith("http"):
            audio_data = audio_path
        else:
            audio_data = f"data:;base64,{base64_encode(audio_path)}"
        contents.append({
            "type": "input_audio",
            "input_audio": {
                "data": audio_data,
                "format": "wav",
            },
        })
    if image_path:
        image_data = base64_encode(image_path)
        contents.append({
            "type": "image_url",
            "image_url": {
                "url": image_data,
            },
        })
    if text:
        contents.append({
            "type": "text",
            "text": text,
        })

    messages = [
        {
            "role": "user",
            "content": contents,
        }
    ]
    return messages

if __name__ == "__main__":
    audio_path = "/home/chuyu/vllab/interactive_audio_3d/output/0118_bathroom/000/squeezing_Dial_hand_soap/1103.wav"
    text = "Given an audio clip of a sound, identify what kind of action likely caused the sound, the object involved, and infer the object's possible materials and physical properties. List several potential actions, the object involved, and the object’s likely material and physical properties in the following json format: ```json{\"action\": [\"action1\", \"action2\", \"action3\", ...], \"object\": [\"object1\", ...], \"material\": [\"material1\", ...], \"physical_properties\": [\"physical_property1\", ...]}```"

    qwen = QwenAPI()
    print(qwen.ask(audio_path=audio_path, text=text, is_json=True))
