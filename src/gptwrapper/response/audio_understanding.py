from pydantic import BaseModel

class AudioUnderstandingResponse(BaseModel):
    answer: str
    justification: str

class AudioRankingResponse(BaseModel):
    ranking: int
    justification: str
