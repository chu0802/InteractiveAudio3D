from pydantic import BaseModel
from typing import List, Dict

class InteractionCandidateResponse(BaseModel):
    action: str
    material: str
    object: str
    description: str

class InteractionResponse(BaseModel):
    interactions: List[InteractionCandidateResponse]


# class RecognitionInteractionResponse(BaseModel):
#     object_name: str
#     properties: List[str]
#     interactions: List[str]

# class InteractionResponse(BaseModel):
#     interaction_1: str
#     interaction_2: str
#     interaction_3: str
#     interaction_4: str
#     interaction_5: str
    
#     def to_list(self):
#         return [
#             self.interaction_1,
#             self.interaction_2,
#             self.interaction_3,
#             self.interaction_4,
#             self.interaction_5,
#         ]
