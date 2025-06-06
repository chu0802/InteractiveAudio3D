from pydantic import BaseModel

class ObjectRecognitionResponse(BaseModel):
    name: str

class DetailedObjectRecognitionResponse(BaseModel):
    material: str
    object_name: str
