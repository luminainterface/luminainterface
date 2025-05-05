from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Message(BaseModel):
    """A message in the chat."""
    role: str = Field(..., description="The role of the message sender (system, user, or assistant)")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    model: str = Field(..., description="The model to use for chat completion")
    messages: List[Message] = Field(..., description="The messages in the chat")
    stream: bool = Field(default=False, description="Whether to stream the response")
    temperature: Optional[float] = Field(default=1.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens to generate")
    top_p: Optional[float] = Field(default=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(default=0.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=0.0, description="Presence penalty") 