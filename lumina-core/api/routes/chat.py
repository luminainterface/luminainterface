from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import httpx
import os

router = APIRouter()

class ChatMessage(BaseModel):
    message: str
    session: str = "default"

class ChatResponse(BaseModel):
    reply: str
    confidence: float
    cite_ids: List[str]

@router.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    # Get LLM response
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{os.getenv('LLM_ENGINE_URL', 'http://llm-engine:11434')}/api/generate",
                json={
                    "model": "phi",
                    "prompt": message.message,
                    "stream": False
                }
            )
            response.raise_for_status()
            llm_response = response.json()
            
            return ChatResponse(
                reply=llm_response["response"],
                confidence=0.8,  # TODO: Implement proper confidence scoring
                cite_ids=[]  # TODO: Implement citation system
            )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}") 