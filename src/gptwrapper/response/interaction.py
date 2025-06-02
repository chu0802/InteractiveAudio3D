from pydantic import BaseModel
from typing import List

class InteractionResponse(BaseModel):
    interaction_1: str
    interaction_2: str
    interaction_3: str
    interaction_4: str
    interaction_5: str
    
    def to_list(self):
        return [
            self.interaction_1,
            self.interaction_2,
            self.interaction_3,
            self.interaction_4,
            self.interaction_5,
        ]
