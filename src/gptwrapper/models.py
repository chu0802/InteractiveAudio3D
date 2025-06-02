from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseModel:
    name: str
    rates: tuple[float, float] = (0.0, 0.0) # per million tokens
    base_url: Optional[str] = None

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

@dataclass
class GPT4o(BaseModel):
    name: str = "gpt-4o"
    rates: tuple[float, float] = (2.5, 10.0)

@dataclass
class GPT4oMini(BaseModel):
    name: str = "gpt-4o-mini"
    rates: tuple[float, float] = (0.15, 0.6)

@dataclass
class Gemini25Pro(BaseModel):
    name: str = "gemini-2.5-pro"
    rates: tuple[float, float] = (1.25, 10.0)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass
class Gemini15Pro(BaseModel):
    name: str = "gemini-1.5-pro"
    rates: tuple[float, float] = (1.25, 5.0)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

@dataclass
class Gemini20Flash(BaseModel):
    name: str = "gemini-2.0-flash"
    rates: tuple[float, float] = (0.1, 0.4)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

@dataclass
class Gemini25Flash(BaseModel):
    name: str = "gemini-2.5-flash"
    rates: tuple[float, float] = (0.1, 0.4)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_model(name: str) -> BaseModel:
    model_dict = BaseModel.__subclasses__()
    for model in model_dict:
        if model.name in name:
            return model(name)
    raise ValueError(f"Model {name} not found")
