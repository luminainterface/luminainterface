#!/usr/bin/env python3
"""
Memory API Client

Command-line client for interacting with the Memory API Server.
Provides a simple interface for testing and interacting with the memory system.
"""

import os
import sys
import json
import argparse
import requests
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from tabulate import tabulate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger("memory_api_client")

# Default API server URL
DEFAULT_API_URL = "http://localhost:8000"


class MemoryAPIClient:
    """Client for interacting with the Memory API Server"""

    def __init__(self, base_url: str = DEFAULT_API_URL):
        """Initialize the Memory API Client.

        Args:
            base_url: Base URL of the Memory API Server
        """
        self.base_url = base_url
        self.session = requests.Session()
        # Test connection on initialization
        try:
            self.health_check()
            logger.info(f"Connected to Memory API Server at {base_url}")
        except Exception as e:
            logger.warning(f"Could not connect to Memory API Server: {e}")

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the API server.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data (for POST requests)

        Returns:
            Response JSON
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=10)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the API server.

        Returns:
            Health check response
        """
        return self._make_request("GET", "/health")

    def store_conversation(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store a conversation message in memory.

        Args:
            message: Message text
            metadata: Optional metadata about the conversation

        Returns:
            Storage result
        """
        data = {"message": message}
        if metadata:
            data["metadata"] = metadata
            
        return self._make_request("POST", "/memory/conversation", data)

    def retrieve_memories(self, message: str) -> Dict[str, Any]:
        """Retrieve memories relevant to a message.

        Args:
            message: Query message text

        Returns:
            Retrieved memories
        """
        return self._make_request("POST", "/memory/retrieve", {"message": message})

    def synthesize_topic(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """Synthesize memories around a specific topic.

        Args:
            topic: Topic to synthesize
            depth: How deep to search for related memories

        Returns:
            Synthesis result
        """
        return self._make_request("POST", "/memory/synthesize", {"topic": topic, "depth": depth})

    def enhance_message(self, message: str, enhance_mode: str = "contextual") -> Dict[str, Any]:
        """Enhance a message with memory context for LLM integration.

        Args:
            message: Message to enhance
            enhance_mode: Enhancement mode (contextual, synthesized, or combined)

        Returns:
            Enhanced message
        """
        return self._make_request("POST", "/memory/enhance", {"message": message, "enhance_mode": enhance_mode})

    def get_training_examples(self, topic: str, count: int = 3) -> Dict[str, Any]:
        """Generate training examples for a specific topic.

        Args:
            topic: Topic to generate examples for
            count: Number of examples to generate

        Returns:
            Training examples
        """
        return self._make_request("POST", "/training/examples", {"topic": topic, "count": count})

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system.

        Returns:
            Memory system statistics
        """
        return self._make_request("GET", "/stats")


def print_response(response: Dict[str, Any]) -> None:
    """Print a formatted response from the API.

    Args:
        response: API response
    """
    if "status" not in response:
        logger.error("Invalid response format")
        print(json.dumps(response, indent=2))
        return
        
    # Print status
    status = response.get("status")
    if status == "success":
        logger.info("Request successful")
    else:
        logger.error(f"Request failed: {response.get('error', 'Unknown error')}")
        
    # Print timestamp if available
    if "timestamp" in response:
        logger.info(f"Timestamp: {response['timestamp']}")
        
    # Print data in a structured format
    if "data" in response:
        data = response["data"]
        
        # Handle different data formats
        if isinstance(data, dict):
            if "stats" in data:
                # Stats format
                stats = data["stats"]
                tables = []
                
                for stat_category, stat_values in stats.items():
                    if isinstance(stat_values, dict):
                        # Create table for this category
                        table_data = []
                        for key, value in stat_values.items():
                            table_data.append([key, value])
                        
                        tables.append((stat_category, table_data))
                
                # Print each table
                for category, table_data in tables:
                    print(f"\n{category.upper()}")
                    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="simple"))
            
            elif "memories" in data:
                # Memories format
                memories = data["memories"]
                if isinstance(memories, list):
                    table_data = []
                    for i, memory in enumerate(memories, 1):
                        content = memory.get("content", "")
                        # Truncate content if too long
                        if len(content) > 80:
                            content = content[:77] + "..."
                        
                        relevance = memory.get("relevance", 0)
                        source = memory.get("source", "unknown")
                        timestamp = memory.get("timestamp", "")
                        
                        table_data.append([i, content, relevance, source, timestamp])
                    
                    print("\nRETRIEVED MEMORIES")
                    print(tabulate(table_data, 
                                  headers=["#", "Content", "Relevance", "Source", "Timestamp"], 
                                  tablefmt="simple"))
            
            elif "synthesis" in data:
                # Synthesis format
                synthesis = data["synthesis"]
                print("\nSYNTHESIS")
                print("=" * 80)
                print(synthesis)
                print("=" * 80)
                
                if "related_topics" in data:
                    print("\nRELATED TOPICS")
                    for topic in data["related_topics"]:
                        print(f"- {topic}")
            
            elif "enhanced_message" in data:
                # Enhanced message format
                print("\nENHANCED MESSAGE")
                print("=" * 80)
                print(data["enhanced_message"])
                print("=" * 80)
                
                if "context_sources" in data:
                    sources = data["context_sources"]
                    if isinstance(sources, list) and len(sources) > 0:
                        print("\nCONTEXT SOURCES")
                        for i, source in enumerate(sources, 1):
                            print(f"{i}. {source}")
            
            else:
                # Generic format
                print("\nRESPONSE DATA")
                print(json.dumps(data, indent=2))
        
        else:
            # Generic format for non-dict data
            print("\nRESPONSE DATA")
            print(json.dumps(data, indent=2))


def interactive_mode(client: MemoryAPIClient) -> None:
    """Run the client in interactive mode.

    Args:
        client: MemoryAPIClient instance
    """
    print("\n=== Memory API Interactive Client ===")
    print(f"Connected to: {client.base_url}")
    print("Type 'help' for available commands, 'exit' to quit.")
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() in ("exit", "quit"):
                break
                
            elif command.lower() == "help":
                print("\nAvailable commands:")
                print("  health              - Check API server health")
                print("  store <message>     - Store a conversation message")
                print("  retrieve <message>  - Retrieve memories relevant to a message")
                print("  synthesize <topic>  - Synthesize memories around a topic")
                print("  enhance <message>   - Enhance a message with memory context")
                print("  examples <topic>    - Generate training examples for a topic")
                print("  stats               - Get memory system statistics")
                print("  exit                - Exit the client")
            
            elif command.lower() == "health":
                response = client.health_check()
                print_response(response)
                
            elif command.lower().startswith("store "):
                message = command[6:].strip()
                if message:
                    response = client.store_conversation(message)
                    print_response(response)
                else:
                    logger.error("Message cannot be empty")
            
            elif command.lower().startswith("retrieve "):
                message = command[9:].strip()
                if message:
                    response = client.retrieve_memories(message)
                    print_response(response)
                else:
                    logger.error("Message cannot be empty")
            
            elif command.lower().startswith("synthesize "):
                topic = command[11:].strip()
                if topic:
                    response = client.synthesize_topic(topic)
                    print_response(response)
                else:
                    logger.error("Topic cannot be empty")
            
            elif command.lower().startswith("enhance "):
                message = command[8:].strip()
                if message:
                    response = client.enhance_message(message)
                    print_response(response)
                else:
                    logger.error("Message cannot be empty")
            
            elif command.lower().startswith("examples "):
                topic = command[9:].strip()
                if topic:
                    response = client.get_training_examples(topic)
                    print_response(response)
                else:
                    logger.error("Topic cannot be empty")
            
            elif command.lower() == "stats":
                response = client.get_stats()
                print_response(response)
            
            else:
                logger.error(f"Unknown command: {command}")
                print("Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Memory API Client")
    
    # Server URL
    parser.add_argument("--url", default=DEFAULT_API_URL,
                      help=f"Memory API Server URL (default: {DEFAULT_API_URL})")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Health check
    subparsers.add_parser("health", help="Check API server health")
    
    # Store conversation
    store_parser = subparsers.add_parser("store", help="Store a conversation message")
    store_parser.add_argument("message", help="Message text")
    store_parser.add_argument("--metadata", help="Optional metadata as JSON string")
    
    # Retrieve memories
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve memories relevant to a message")
    retrieve_parser.add_argument("message", help="Message text")
    
    # Synthesize topic
    synthesize_parser = subparsers.add_parser("synthesize", help="Synthesize memories around a topic")
    synthesize_parser.add_argument("topic", help="Topic to synthesize")
    synthesize_parser.add_argument("--depth", type=int, default=3, help="Search depth (default: 3)")
    
    # Enhance message
    enhance_parser = subparsers.add_parser("enhance", help="Enhance a message with memory context")
    enhance_parser.add_argument("message", help="Message text")
    enhance_parser.add_argument("--mode", default="contextual", 
                             choices=["contextual", "synthesized", "combined"],
                             help="Enhancement mode (default: contextual)")
    
    # Get training examples
    examples_parser = subparsers.add_parser("examples", help="Generate training examples for a topic")
    examples_parser.add_argument("topic", help="Topic to generate examples for")
    examples_parser.add_argument("--count", type=int, default=3, help="Number of examples (default: 3)")
    
    # Get stats
    subparsers.add_parser("stats", help="Get memory system statistics")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Run in interactive mode")
    
    return parser.parse_args()


def main():
    """Main entry point for the Memory API Client."""
    args = parse_args()
    
    # Initialize client
    client = MemoryAPIClient(args.url)
    
    # If no command is specified, default to interactive mode
    if not args.command:
        interactive_mode(client)
        return
    
    # Process commands
    if args.command == "health":
        response = client.health_check()
    
    elif args.command == "store":
        metadata = None
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError:
                logger.error("Invalid metadata JSON")
                return
        response = client.store_conversation(args.message, metadata)
    
    elif args.command == "retrieve":
        response = client.retrieve_memories(args.message)
    
    elif args.command == "synthesize":
        response = client.synthesize_topic(args.topic, args.depth)
    
    elif args.command == "enhance":
        response = client.enhance_message(args.message, args.mode)
    
    elif args.command == "examples":
        response = client.get_training_examples(args.topic, args.count)
    
    elif args.command == "stats":
        response = client.get_stats()
    
    elif args.command == "interactive":
        interactive_mode(client)
        return
    
    # Print the response
    print_response(response)


if __name__ == "__main__":
    main() 