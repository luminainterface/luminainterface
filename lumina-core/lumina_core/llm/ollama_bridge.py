import os
import json
import httpx
from typing import AsyncGenerator, Dict, Any

class OllamaBridge:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://llm-engine:11434")
        self.model = os.getenv("OLLAMA_MODEL", "phi2")
        self.client = httpx.AsyncClient(timeout=60.0)
        self._tokens_used = 0

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    async def generate_stream(
        self, 
        prompt: str,
        context: list[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream responses from Ollama API."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "context": context or [],
                "stream": True
            }
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            if "eval_count" in chunk:
                                self._tokens_used += chunk["eval_count"]
                            yield chunk
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {str(e)}")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose() 