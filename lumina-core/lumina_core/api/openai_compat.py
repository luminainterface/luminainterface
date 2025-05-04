from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import time
import uuid

from lumina_core.llm.ollama_bridge import OllamaBridge
from lumina_core.memory.qdrant_store import QdrantStore

router = APIRouter(prefix="/v1")
ollama = OllamaBridge()
memory = QdrantStore()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Convert messages to prompt
        prompt = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in request.messages
        ])
        
        # Get similar messages for context
        similar_messages = await memory.get_similar_messages(prompt)
        context = [msg["content"] for msg in similar_messages]
        
        if request.stream:
            async def generate():
                full_response = ""
                async for chunk in ollama.generate_stream(prompt, context):
                    if "response" in chunk:
                        full_response += chunk["response"]
                        yield f"data: {json.dumps({
                            'id': str(uuid.uuid4()),
                            'object': 'chat.completion.chunk',
                            'created': int(time.time()),
                            'model': request.model,
                            'choices': [{
                                'index': 0,
                                'delta': {'content': chunk['response']},
                                'finish_reason': None
                            }]
                        })}\n\n"
                
                # Store conversation
                await memory.upsert_messages([
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ] + [{
                    "role": "assistant",
                    "content": full_response
                }])
                
                # Send final chunk
                yield f"data: {json.dumps({
                    'id': str(uuid.uuid4()),
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': request.model,
                    'choices': [{
                        'index': 0,
                        'delta': {},
                        'finish_reason': 'stop'
                    }]
                })}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            full_response = ""
            async for chunk in ollama.generate_stream(prompt, context):
                if "response" in chunk:
                    full_response += chunk["response"]
            
            # Store conversation
            await memory.upsert_messages([
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ] + [{
                "role": "assistant",
                "content": full_response
            }])
            
            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(time.time()),
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_response
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(full_response.split()),
                    "total_tokens": len(prompt.split()) + len(full_response.split())
                }
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """OpenAI-compatible models endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "phi2",
                "object": "model",
                "created": 1704067200,  # Jan 1, 2024
                "owned_by": "lumina"
            }
        ]
    } 