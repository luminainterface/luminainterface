import asyncio
import logging
from agents import get_agent, PLANS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_agents")

async def test_wiki_qa():
    """Test the Wikipedia QA workflow"""
    try:
        # Get the plan
        plan = PLANS["wiki_qa"]
        
        # Initialize context
        context = {
            "question": "What is quantum computing?",
            "topic": "quantum computing"
        }
        
        # Execute plan
        results = {}
        for step in plan:
            agent_name = step["agent"]
            agent = get_agent(agent_name)
            
            # Format input with context
            input_data = {}
            for key, value in step["input"].items():
                if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                    context_key = value[1:-1]
                    input_data[key] = context.get(context_key)
                else:
                    input_data[key] = value
            
            # Execute agent
            logger.info(f"Executing {agent_name}: {step['description']}")
            result = await agent.run(input_data)
            
            # Store result in context
            context[f"{agent_name.lower()}_result"] = result
            results[agent_name] = result
            
            logger.info(f"{agent_name} completed successfully")
        
        # Print final answer
        print("\nFinal Answer:")
        print(results["QAAgent"]["answer"])
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_wiki_qa()) 