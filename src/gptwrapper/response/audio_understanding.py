from pydantic import BaseModel

class AudioUnderstandingResponse(BaseModel):
    action: str
    object: str
    material: str
    justification: str

class AudioRankingResponse(BaseModel):
    ranking: int
    justification: str
