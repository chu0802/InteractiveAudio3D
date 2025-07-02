from pydantic import BaseModel
from typing import List

class AudioUnderstandingCandidateResponse(BaseModel):
    potential_actions: List[str]
    potential_objects: List[str]
    potential_materials_and_properties: List[str]

class AudioUnderstandingResponse(BaseModel):
    answer: str
    justification: str

class AudioRankingResponse(BaseModel):
    ranking: int
    justification: str
