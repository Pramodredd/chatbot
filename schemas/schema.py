from pydantic import BaseModel
from typing import List ,Dict ,Any

class ConversationPayload(BaseModel):
    query: str
    thread_id: str

class AIResponse(BaseModel):
    response: str

