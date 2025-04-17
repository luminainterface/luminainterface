#!/usr/bin/env python3
"""
LLM Prompt Enhancer

A script that enhances LLM prompts with memory context by connecting to the
Memory API Server and integrating contextually relevant memories.

This allows existing LLM systems to benefit from memory capabilities without
requiring direct integration.
"""

import os
import sys
import json
import argparse
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhance_llm_prompt")

# Default API server URL
DEFAULT_API_URL = "http://localhost:8000"


class PromptEnhancer:
    """Class that enhances LLM prompts with memory context"""
    
    def __init__(self, api_url: str = DEFAULT_API_URL):
        """Initialize the PromptEnhancer.
        
        Args:
            api_url: URL of the Memory API Server
        """
        self.api_url = api_url
        self.session = requests.Session()
        self._verify_connection()
    
    def _verify_connection(self) -> bool:
        """Verify connection to the Memory API Server.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            response.raise_for_status()
            logger.info(f"Connected to Memory API Server at {self.api_url}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Memory API Server: {str(e)}")
            return False
    
    def _make_api_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the Memory API Server.
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response from the API server
            
        Raises:
            RuntimeError: If the request fails
        """
        try:
            response = self.session.post(
                f"{self.api_url}{endpoint}",
                json=data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise RuntimeError(f"Failed to make API request: {str(e)}")
    
    def enhance_prompt(self, 
                     prompt: str, 
                     mode: str = "contextual", 
                     max_memories: int = 5,
                     store_prompt: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
        """Enhance an LLM prompt with memory context.
        
        Args:
            prompt: Original LLM prompt
            mode: Enhancement mode ('contextual', 'synthesized', or 'combined')
            max_memories: Maximum number of memories to include
            store_prompt: Whether to store the prompt in memory
            
        Returns:
            Tuple of (enhanced prompt, list of memories used)
        """
        # Store the prompt in memory if requested
        if store_prompt:
            try:
                store_data = {"message": prompt, "metadata": {"source": "llm_prompt"}}
                self._make_api_request("/memory/conversation", store_data)
                logger.debug("Prompt stored in memory")
            except Exception as e:
                logger.warning(f"Failed to store prompt in memory: {str(e)}")
        
        # Enhance the prompt with memory context
        enhance_data = {
            "message": prompt,
            "enhance_mode": mode,
            "max_memories": max_memories
        }
        
        response = self._make_api_request("/memory/enhance", enhance_data)
        
        if response.get("status") != "success":
            error_msg = response.get("error", "Unknown error")
            logger.error(f"Failed to enhance prompt: {error_msg}")
            return prompt, []
        
        enhanced_prompt = response.get("data", {}).get("enhanced_message", prompt)
        memories = response.get("data", {}).get("memories", [])
        
        return enhanced_prompt, memories


def format_memories(memories: List[Dict[str, Any]]) -> str:
    """Format memories for display.
    
    Args:
        memories: List of memory objects
        
    Returns:
        Formatted string of memories
    """
    if not memories:
        return "No memories used."
    
    formatted = []
    for i, memory in enumerate(memories, 1):
        content = memory.get("content", "")
        relevance = memory.get("relevance", 0)
        timestamp = memory.get("timestamp", "")
        
        formatted.append(f"{i}. [{relevance:.2f}] {content}")
        if timestamp:
            formatted[-1] += f" (from {timestamp})"
    
    return "\n".join(formatted)


def create_basic_prompt(prompt_type: str, topic: str) -> str:
    """Create a basic prompt based on the specified type and topic.
    
    Args:
        prompt_type: Type of prompt to create
        topic: Topic for the prompt
        
    Returns:
        Generated prompt
    """
    if prompt_type == "question":
        return f"What can you tell me about {topic}?"
    
    elif prompt_type == "explain":
        return f"Explain {topic} as if I'm a beginner."
    
    elif prompt_type == "compare":
        parts = topic.split("_and_")
        if len(parts) == 2:
            return f"Compare and contrast {parts[0]} and {parts[1]}."
        else:
            return f"Compare and contrast different aspects of {topic}."
    
    elif prompt_type == "summarize":
        return f"Summarize the key points about {topic}."
    
    elif prompt_type == "creative":
        return f"Write a creative short story involving {topic}."
    
    else:
        return f"Tell me about {topic}."


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LLM Prompt Enhancer")
    
    # API URL
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                       help=f"Memory API Server URL (default: {DEFAULT_API_URL})")
    
    # Enhance mode
    parser.add_argument("--mode", choices=["contextual", "synthesized", "combined"],
                       default="contextual",
                       help="Enhancement mode (default: contextual)")
    
    # Max memories
    parser.add_argument("--max-memories", type=int, default=5,
                       help="Maximum memories to include (default: 5)")
    
    # Whether to store the prompt
    parser.add_argument("--no-store", action="store_true",
                       help="Don't store the prompt in memory")
    
    # Verbosity
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    # Output format
    parser.add_argument("--output", "-o", choices=["text", "json", "markdown"],
                       default="text",
                       help="Output format (default: text)")
    
    # Input methods (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    # Read from file
    input_group.add_argument("--file", "-f", 
                           help="Read prompt from file")
    
    # Read from command line
    input_group.add_argument("--prompt", "-p", 
                           help="Prompt text")
    
    # Generate a basic prompt
    generate_group = input_group.add_argument_group("Generate prompt")
    input_group.add_argument("--generate", choices=["question", "explain", "compare", 
                                                  "summarize", "creative"],
                           help="Generate a basic prompt")
    generate_group.add_argument("--topic", 
                              help="Topic for generated prompt")
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get prompt text
    prompt = None
    
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            logger.info(f"Read prompt from file: {args.file}")
        except Exception as e:
            logger.error(f"Failed to read prompt from file: {str(e)}")
            sys.exit(1)
    
    elif args.prompt:
        prompt = args.prompt
    
    elif args.generate:
        if not args.topic:
            logger.error("Topic is required for generated prompts")
            sys.exit(1)
        prompt = create_basic_prompt(args.generate, args.topic)
        logger.info(f"Generated prompt: {prompt}")
    
    # Initialize enhancer
    enhancer = PromptEnhancer(args.api_url)
    
    # Enhance prompt
    try:
        enhanced_prompt, memories = enhancer.enhance_prompt(
            prompt=prompt,
            mode=args.mode,
            max_memories=args.max_memories,
            store_prompt=not args.no_store
        )
        
        # Format output
        if args.output == "text":
            print("\n--- ORIGINAL PROMPT ---")
            print(prompt)
            print("\n--- ENHANCED PROMPT ---")
            print(enhanced_prompt)
            print("\n--- MEMORIES USED ---")
            print(format_memories(memories))
        
        elif args.output == "json":
            result = {
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "memories": memories
            }
            print(json.dumps(result, indent=2))
        
        elif args.output == "markdown":
            print("# LLM Prompt Enhancement\n")
            print("## Original Prompt\n")
            print(f"```\n{prompt}\n```\n")
            print("## Enhanced Prompt\n")
            print(f"```\n{enhanced_prompt}\n```\n")
            print("## Memories Used\n")
            if memories:
                for i, memory in enumerate(memories, 1):
                    content = memory.get("content", "")
                    relevance = memory.get("relevance", 0)
                    timestamp = memory.get("timestamp", "")
                    
                    print(f"{i}. **Relevance: {relevance:.2f}**")
                    print(f"   {content}")
                    if timestamp:
                        print(f"   *From: {timestamp}*")
                    print()
            else:
                print("No memories used.")
        
    except Exception as e:
        logger.error(f"Failed to enhance prompt: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 