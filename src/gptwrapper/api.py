import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import base64
from pathlib import Path
from .response import GeneralResponse
from .models import BaseModel, get_model
import os
from openai import OpenAI


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

@dataclass
class BaseContent:
    text: Optional[str] = None
    image: Optional[str | Path] = None
    audio: Optional[str | Path] = None

    def to_dict(self):
        if self.text is None and self.image is None and self.audio is None:
            raise ValueError("Either text or image or audio should be provided")
        type = "text" if self.text else ("image_url" if self.image else "input_audio")

        if isinstance(self.image, Path):
            self.image = self.image.as_posix()

        if isinstance(self.image, str):
            self.image = encode_image(self.image)
        
        if isinstance(self.audio, Path):
            self.audio = self.audio.as_posix()
        
        if isinstance(self.audio, str):
            self.audio = encode_audio(self.audio)

        content = (
            self.text
            if type == "text"
            else (
                {
                    "url": f"data:image/jpeg;base64,{self.image}"
                }
                if type == "image_url"
                else {
                    "data": self.audio,
                    "format": "wav"
                }
            )
        )
        return {"type": type, type: content}

class PromptMessages:
    def __init__(self, system_message: Optional[str] = None):
        self._messages = []
        if system_message:
            self.reset_message(system_message=system_message)

    @property
    def messages(self):
        return self._messages

    def add_message(self, role="user", image: Optional[str | Path] = None, text: Optional[str] = None, audio: Optional[str | Path] = None):
        contents = []
        if image:
            contents.append(BaseContent(image=image))
        if audio:
            contents.append(BaseContent(audio=audio))
        if text:
            contents.append(BaseContent(text=text))

        self._add_message(role, contents)

    def _add_message(self, role: str, content: Union[BaseContent, List[BaseContent]]):
        if not isinstance(content, list):
            content = [content]

        self._messages.append({"role": role, "content": [c.to_dict() for c in content]})

    def reset_message(self, system_message: Optional[str] = None):
        self._messages = []
        if system_message:
            self.add_message(role="system", text=system_message)
        return self

@dataclass
class GPTCost:
    model: BaseModel
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0

    def __add__(self, other: "GPTCost"):
        assert self.model == other.model
        return GPTCost(
            self.model,
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cost + other.cost,
        )

    def __radd__(self, other: "GPTCost"):
        return self.__add__(other)

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0
        return self

    @classmethod
    def from_gpt_results(cls, model: BaseModel, result: Any):
        pt = result.usage.prompt_tokens
        ct = result.usage.completion_tokens
        rates = model.rates
        prompt_rate, completion_rate = rates
        cost = (pt / 1_000_000) * prompt_rate + (ct / 1_000_000) * completion_rate
        return cls(model, pt, ct, cost)

    def __repr__(self):
        output_dict = {
            "model": self.model.name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
        }
        return json.dumps(output_dict, indent=4)

class GPTWrapper:
    def __init__(self, 
        model_name: str = "gpt-4o", 
        **model_params,
    ):
        self.model = get_model(model_name)
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"), 
            base_url=self.model.base_url,
        )
        self.params = {
            "model": model_name,
            **model_params,
        }
        self.total_cost = GPTCost(model=self.model)
        self.error_requests = []

    def add_cost(self, results: List[Any]):
        if not isinstance(results, list):
            results = [results]
        for result in results:
            self.total_cost += GPTCost.from_gpt_results(self.model, result)

    def show_cost(self):
        return self.total_cost

    def ask(
        self, 
        image: Optional[str | Path] = None, 
        text: Optional[str] = None, 
        audio: Optional[str | Path] = None,
        system_message: Optional[str] = None,
        response_format: Any = None,
        return_full_response: bool = False,
        parse_json: bool = False,
    ):
        msgs = PromptMessages(system_message)
        msgs.add_message(image=image, text=text, audio=audio)

        if response_format:
            result = self.client.beta.chat.completions.parse(
                messages=msgs.messages, 
                response_format=response_format, 
                **self.params,
            )
        else:
            result = self.client.chat.completions.create(
                messages=msgs.messages, 
                **self.params,
            )

        self.add_cost(result)
        if return_full_response:
            return result
        else:
            if response_format:
                return result.choices[0].message.parsed
            elif parse_json and "```json" in result.choices[0].message.content:
                content = result.choices[0].message.content.split("```json")[1].split("```")[0].strip()
                content.replace("'", '"')
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content
            else:
                return result.choices[0].message.content
