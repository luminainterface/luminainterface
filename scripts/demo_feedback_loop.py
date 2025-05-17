#!/usr/bin/env python3
import asyncio
import httpx
import redis.asyncio as redis
import argparse
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cosine
import difflib
from prometheus_client import CollectorRegistry, Counter, Gauge, push_to_gateway
import os
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

class DemoConfig:
    def __init__(self):
        # Use Docker service names
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        self.feedback_logger_url = os.getenv("FEEDBACK_LOGGER_URL", "http://feedback-logger:8901")
        self.concept_trainer_url = os.getenv("CONCEPT_TRAINER_URL", "http://concept-trainer:8906")
        self.concept_dict_url = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8000")
        self.prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
        self.prom_pushgateway = os.getenv("PUSHGATEWAY_URL", "http://prometheus:9091")

    async def check_services(self):
        """Check if required services are available."""
        errors = []
        async with httpx.AsyncClient() as client:
            # Check Redis
            try:
                redis_client = redis.from_url(self.redis_url)
                await redis_client.ping()
                console.print("[green]✓[/green] Redis is available")
            except Exception as e:
                errors.append(f"Redis is not available: {str(e)}")
                console.print("[red]✗[/red] Redis is not available")
            
            # Check Feedback Logger
            try:
                response = await client.get(f"{self.feedback_logger_url}/health")
                if response.status_code == 200:
                    console.print("[green]✓[/green] Feedback Logger is available")
                else:
                    errors.append("Feedback Logger returned non-200 status")
                    console.print("[red]✗[/red] Feedback Logger is not healthy")
            except Exception as e:
                errors.append(f"Feedback Logger is not available: {str(e)}")
                console.print("[red]✗[/red] Feedback Logger is not available")
            
            # Check Concept Trainer
            try:
                response = await client.get(f"{self.concept_trainer_url}/health")
                if response.status_code == 200:
                    console.print("[green]✓[/green] Concept Trainer is available")
                else:
                    errors.append("Concept Trainer returned non-200 status")
                    console.print("[red]✗[/red] Concept Trainer is not healthy")
            except Exception as e:
                errors.append(f"Concept Trainer is not available: {str(e)}")
                console.print("[red]✗[/red] Concept Trainer is not available")
        
        if errors:
            console.print("\n[red]Error:[/red] Some services are not available:")
            for error in errors:
                console.print(f"  • {error}")
            console.print("\nPlease ensure all services are running:")
            console.print("  docker-compose up -d redis feedback-logger concept-trainer")
            return False
        
        return True

class MetricsCollector:
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Demo metrics
        self.demo_requests = Counter(
            'demo_requests_total',
            'Total number of demo requests',
            registry=self.registry
        )
        
        self.demo_drift = Gauge(
            'demo_drift_score',
            'Drift score between LLM and NN responses',
            ['concept_id'],
            registry=self.registry
        )
        
        self.demo_confidence = Gauge(
            'demo_confidence_score',
            'Confidence score for responses',
            ['model', 'concept_id'],
            registry=self.registry
        )
        
        self.demo_latency = Gauge(
            'demo_response_latency_seconds',
            'Response latency in seconds',
            ['model'],
            registry=self.registry
        )
    
    def push_metrics(self, pushgateway_url: str):
        """Push metrics to Pushgateway."""
        try:
            from prometheus_client import push_to_gateway
            push_to_gateway(pushgateway_url, job='lumina_demo', registry=self.registry)
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")

class ResponseSimulator:
    """Simulates LLM and NN responses for demo purposes."""
    
    def __init__(self):
        self.llm_confidence_range = (0.7, 0.95)
        self.nn_confidence_range = (0.5, 0.85)
    
    def _generate_embedding(self, text: str, dim: int = 768) -> List[float]:
        """Generate a mock embedding vector."""
        # Use text hash as seed for reproducibility
        np.random.seed(hash(text) % 2**32)
        return np.random.normal(0, 1, dim).tolist()
    
    def _calculate_drift(self, text1: str, text2: str) -> float:
        """Calculate drift between two text responses."""
        # Use difflib for text similarity
        matcher = difflib.SequenceMatcher(None, text1, text2)
        text_similarity = matcher.ratio()
        
        # Calculate semantic similarity using embeddings
        emb1 = self._generate_embedding(text1)
        emb2 = self._generate_embedding(text2)
        semantic_similarity = 1 - cosine(emb1, emb2)
        
        # Combine text and semantic similarity
        return 1 - ((text_similarity + semantic_similarity) / 2)

    async def generate_responses(self, prompt: str, concept_id: str) -> Tuple[Dict, Dict, float]:
        """Generate simulated LLM and NN responses."""
        # Simulate LLM response
        llm_response = {
            "text": f"The concept of {concept_id} refers to a fundamental principle in quantum mechanics...",
            "confidence": np.random.uniform(*self.llm_confidence_range),
            "embedding": self._generate_embedding(concept_id)
        }
        
        # Simulate NN response with some drift
        drift_factor = np.random.uniform(0.1, 0.4)
        nn_text = f"In quantum mechanics, {concept_id} is a key principle that describes..."
        nn_response = {
            "text": nn_text,
            "confidence": np.random.uniform(*self.nn_confidence_range),
            "embedding": self._generate_embedding(nn_text)
        }
        
        # Calculate drift
        drift_score = self._calculate_drift(llm_response["text"], nn_response["text"])
        
        return llm_response, nn_response, drift_score

class FeedbackLoopDemo:
    def __init__(self, config: DemoConfig):
        self.config = config
        self.metrics = MetricsCollector()
        self.simulator = ResponseSimulator()
        self.redis_client = redis.from_url(config.redis_url)
        self.http_client = httpx.AsyncClient()
    
    async def _generate_llm_response(self, prompt: str) -> str:
        """Simulate LLM response."""
        return "Quantum entanglement is a phenomenon in quantum physics where two or more particles become correlated in such a way that the quantum state of each particle cannot be described independently."

    async def _generate_nn_response(self, prompt: str) -> str:
        """Simulate NN response."""
        return "Quantum entanglement occurs when particles interact in ways that their quantum states become linked, regardless of the distance between them. This is what Einstein called 'spooky action at a distance'."

    def _calculate_confidence_delta(self, llm_response: str, nn_response: str) -> float:
        """Calculate confidence delta between LLM and NN responses."""
        # Use difflib for text similarity
        matcher = difflib.SequenceMatcher(None, llm_response, nn_response)
        similarity = matcher.ratio()
        return 1.0 - similarity  # Higher delta means more difference

    def _calculate_feedback_score(self, confidence_delta: float) -> float:
        """Calculate feedback score based on confidence delta."""
        # For demo purposes, we'll use a simple scoring function
        return max(0.0, 1.0 - confidence_delta)

    async def run_demo(self, prompt: str, concept_id: str):
        """Run the cognitive feedback loop demo."""
        console.print("\nRunning Cognitive Feedback Loop Demo")
        console.print(f"Prompt: {prompt}")
        console.print(f"Concept: {concept_id}\n")

        try:
            # Generate responses
            llm_response = await self._generate_llm_response(prompt)
            nn_response = await self._generate_nn_response(prompt)
            
            # Calculate metrics
            confidence_delta = self._calculate_confidence_delta(llm_response, nn_response)
            feedback_score = self._calculate_feedback_score(confidence_delta)

            # Log feedback
            async with httpx.AsyncClient() as client:
                feedback_data = {
                    "concept_id": concept_id,
                    "user_input": prompt,
                    "llm_response": llm_response,
                    "nn_response": nn_response,
                    "confidence_delta": confidence_delta,
                    "feedback_score": feedback_score
                }
                
                response = await client.post(
                    f"{self.config.feedback_logger_url}/feedback",
                    json=feedback_data
                )
                response.raise_for_status()

            # Display results
            console.print(Panel.fit(
                f"[bold]LLM Response:[/bold]\n{llm_response}\n\n"
                f"[bold]NN Response:[/bold]\n{nn_response}\n\n"
                f"[bold]Metrics:[/bold]\n"
                f"Confidence Delta: {confidence_delta:.2f}\n"
                f"Feedback Score: {feedback_score:.2f}",
                title="Demo Results"
            ))

            # Start training monitor
            console.print("\nStarting Training Monitor")
            await self._monitor_training(concept_id)

        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            console.print(f"\n[red]Error running demo:[/red] {str(e)}")
            sys.exit(1)

    async def _monitor_training(self, concept_id: str):
        """Monitor training progress."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Waiting for training updates...", total=None)
            
            try:
                redis_client = redis.from_url(self.config.redis_url)
                pubsub = redis_client.pubsub()
                await pubsub.subscribe("training_events")
                
                while True:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message["type"] == "message":
                        data = json.loads(message["data"])
                        if data["concept_id"] == concept_id:
                            progress.update(
                                task,
                                description=f"Training progress: {data['progress']:.1%}"
                            )
                            if data["status"] == "completed":
                                break
                    
                    await asyncio.sleep(0.1)
                
            except asyncio.TimeoutError:
                console.print("[yellow]Training monitor timed out[/yellow]")
            except Exception as e:
                console.print(f"[red]Error monitoring training:[/red] {str(e)}")
            finally:
                await pubsub.unsubscribe("training_events")

async def main():
    parser = argparse.ArgumentParser(description="Lumina Cognitive Feedback Loop Demo")
    parser.add_argument("prompt", help="The prompt to test")
    parser.add_argument("--concept", default="quantum_drift", help="The concept ID to use")
    args = parser.parse_args()
    
    config = DemoConfig()
    
    # Check services before running demo
    console.print("[bold cyan]Checking required services...[/bold cyan]")
    if not await config.check_services():
        sys.exit(1)
    
    demo = FeedbackLoopDemo(config)
    await demo.run_demo(args.prompt, args.concept)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        sys.exit(1) 