from pydantic import BaseModel

class ImageDecompositionResponse(BaseModel):
    image_properties: list[str]
    checklist: list[str]
    
class AudioVerificationResponse(BaseModel):
    verification: list[str]
    justification: str

class ImageUnderstandingResponse(BaseModel):
    object_name: str
    properties: list[str]