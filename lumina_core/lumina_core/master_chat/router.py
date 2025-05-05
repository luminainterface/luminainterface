from typing import Dict, Any, AsyncGenerator
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from loguru import logger
from .plans import get_plan
from ..agents import get_agent
from ..nn_growth import bump_node

router = APIRouter()

async def run_agent(agent_name: str, payload: Dict[str, Any], sse_writer) -> Dict[str, Any]:
    """
    Run an agent and handle its output.
    
    Args:
        agent_name: Name of the agent to run
        payload: Input payload for the agent
        sse_writer: SSE writer for streaming updates
        
    Returns:
        Agent's output
    """
    try:
        # Get agent instance
        agent = get_agent(agent_name)
        
        # Log start
        await sse_writer.send(event="log", data={
            "agent": agent_name,
            "status": "start",
            "ts": int(time.time()),
            "detail": f"Starting {agent_name}"
        })
        
        # Run agent
        result = await agent.run(payload)
        
        # Bump node count on success
        if result["status"] == "success":
            bump_node(agent_name)
        
        # Log end
        await sse_writer.send(event="log", data={
            "agent": agent_name,
            "status": "end",
            "ts": int(time.time()),
            "detail": f"Completed {agent_name}"
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Agent {agent_name} failed: {str(e)}")
        
        # Log error
        await sse_writer.send(event="log", data={
            "agent": agent_name,
            "status": "error",
            "ts": int(time.time()),
            "detail": str(e)
        })
        
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/plan")
async def create_plan(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create and execute a plan.
    
    Args:
        request: Dictionary containing:
            - mode: str, The mode to run
            - question: str, The question to answer (for wiki-qa)
            
    Returns:
        Dictionary containing the plan's output
    """
    mode = request.get("mode")
    if not mode:
        raise HTTPException(status_code=400, detail="Mode is required")
    
    # Get plan steps
    plan_steps = await get_plan(mode)
    if not plan_steps:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")
    
    # Initialize payload
    payload = request.copy()
    result = None
    
    # Execute plan steps
    for step in plan_steps:
        agent_name = step["agent"]
        result = await run_agent(agent_name, payload, sse_writer)
        
        # Stop on error
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Update payload for next step
        payload = result
    
    return result

@router.get("/stream")
async def stream_plan(request: Dict[str, Any]) -> EventSourceResponse:
    """
    Stream plan execution updates.
    
    Args:
        request: Dictionary containing:
            - mode: str, The mode to run
            - question: str, The question to answer (for wiki-qa)
            
    Returns:
        SSE response with plan execution updates
    """
    async def event_generator():
        try:
            # Get plan steps
            plan_steps = await get_plan(request["mode"])
            if not plan_steps:
                yield {
                    "event": "error",
                    "data": f"Unknown mode: {request['mode']}"
                }
                return
            
            # Initialize payload
            payload = request.copy()
            result = None
            
            # Execute plan steps
            for step in plan_steps:
                agent_name = step["agent"]
                result = await run_agent(agent_name, payload, event_generator)
                
                # Stop on error
                if result["status"] == "error":
                    yield {
                        "event": "error",
                        "data": result["error"]
                    }
                    return
                
                # Update payload for next step
                payload = result
            
            # Send final result
            yield {
                "event": "final",
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Plan execution failed: {str(e)}")
            yield {
                "event": "error",
                "data": str(e)
            }
    
    return EventSourceResponse(event_generator()) 