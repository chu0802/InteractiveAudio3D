from pydantic import BaseModel

class ObjectRecognitionResponse(BaseModel):
    name: str
