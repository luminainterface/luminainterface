#!/usr/bin/env python3
"""
Dream Mode Demo for Lumina V7

This script demonstrates the functionality of the V7 Dream Mode system by:
1. Initializing the dream controller and related components
2. Running a complete dream cycle
3. Displaying insights and patterns generated during the dream
4. Showing how to retrieve dream records from the archive

Usage:
    python dream_mode_demo.py [--duration MINUTES] [--intensity LEVEL]
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dream_demo")

# Import Dream Mode components
from src.v7.lumina_v7.core.dream_controller import get_dream_controller
from src.v7.lumina_v7.core.memory_consolidator import MemoryConsolidator
from src.v7.lumina_v7.core.pattern_synthesizer import PatternSynthesizer
from src.v7.lumina_v7.core.dream_archive import DreamArchive

def setup_demo_environment():
    """Set up the environment for the dream mode demo"""
    # Create data directory
    demo_data_dir = project_root / "data" / "v7" / "demo" / "dreams"
    demo_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Return configuration
    return {
        "dream_enabled": True,
        "dream_data_dir": str(demo_data_dir),
        "auto_dream": False,  # Disable auto-dream for demo
        "default_dream_duration": 5,  # 5 minutes for demo
        "default_dream_intensity": 0.7
    }

def print_section(title):
    """Print a section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_json(data):
    """Print data as formatted JSON"""
    print(json.dumps(data, indent=2))

def run_dream_cycle(duration=None, intensity=0.7):
    """Run a complete dream cycle and display results"""
    print_section("Dream Mode Demo - V7 Node Consciousness")
    print(f"Starting demo with duration={duration} minutes, intensity={intensity}")
    
    # Set up the demo environment
    config = setup_demo_environment()
    
    # Initialize the dream controller
    print("\nInitializing Dream Controller...")
    dream_controller = get_dream_controller(config=config)
    
    # Display initial state
    print("\nInitial dream state:")
    initial_state = dream_controller.get_dream_state()
    print_json(initial_state)
    
    # Enter dream state
    print("\nEntering dream state...")
    dream_controller.enter_dream_state(duration=duration, intensity=intensity)
    
    # Display dream state
    dream_state = dream_controller.get_dream_state()
    print("\nDream state after entering:")
    print_json(dream_state)
    
    # Wait for dream to complete
    print("\nDream is active. Waiting for dream cycle to complete...")
    
    # Calculate wait time
    wait_duration = duration * 60 if duration else 300  # 5 minutes default
    
    # Monitor dream state while waiting
    start_time = time.time()
    while dream_controller.dream_active and time.time() - start_time < wait_duration:
        # Print current state every 10 seconds
        current_state = dream_controller.current_state
        print(f"Current dream phase: {current_state} (elapsed: {int(time.time() - start_time)}s)")
        
        # Sleep for a bit
        time.sleep(10)
    
    # Ensure dream is complete
    if dream_controller.dream_active:
        print("\nForcing exit from dream state...")
        dream_controller.exit_dream_state()
    
    # Get final dream state
    final_state = dream_controller.get_dream_state()
    print("\nFinal dream state:")
    print_json(final_state)
    
    # Get completed dream
    dream_id = dream_controller.current_dream.get("id") if dream_controller.current_dream else None
    
    if dream_id:
        print(f"\nRetrieving dream record {dream_id}...")
        dream = dream_controller.get_dream(dream_id)
        
        # Display insights
        if dream and "insights" in dream and dream["insights"]:
            print_section("Dream Insights")
            for i, insight in enumerate(dream["insights"]):
                print(f"Insight {i+1}: {insight.get('text', 'No text')}")
                print(f"Type: {insight.get('type', 'unknown')}")
                print(f"Confidence: {insight.get('confidence', 0):.2f}")
                print()
        else:
            print("\nNo insights were generated during this dream.")
        
        # Display patterns
        if dream and "synthesis_results" in dream:
            pattern_count = sum(
                len(result.get("patterns", [])) 
                for result in dream["synthesis_results"] 
                if isinstance(result, dict)
            )
            
            if pattern_count > 0:
                print_section(f"Dream Patterns ({pattern_count} total)")
                
                # Get a sample of patterns
                patterns = []
                for result in dream["synthesis_results"]:
                    if isinstance(result, dict) and "patterns" in result:
                        patterns.extend(result["patterns"])
                
                # Display up to 3 patterns
                for i, pattern in enumerate(patterns[:3]):
                    print(f"Pattern {i+1} - Type: {pattern.get('type', 'unknown')}")
                    print(f"Description: {pattern.get('description', 'No description')}")
                    print(f"Confidence: {pattern.get('confidence', 0):.2f}")
                    print()
                
                if pattern_count > 3:
                    print(f"... and {pattern_count - 3} more patterns")
            else:
                print("\nNo patterns were synthesized during this dream.")
    
    # Display dream archive stats
    print_section("Dream Archive Statistics")
    archive_stats = dream_controller.dream_archive.get_stats()
    print(f"Total Dreams: {archive_stats.get('total_dreams', 0)}")
    print(f"Total Insights: {archive_stats.get('insight_count', 0)}")
    print(f"Average Dream Duration: {archive_stats.get('average_dream_duration', 0):.1f} minutes")
    
    # Display most common topics
    if "most_common_topics" in archive_stats and archive_stats["most_common_topics"]:
        print("\nMost Common Topics:")
        for topic, count in archive_stats["most_common_topics"].items():
            print(f"  - {topic}: {count}")
    
    print("\nDemo completed successfully!")

def main():
    """Main entry point for the demo"""
    parser = argparse.ArgumentParser(description="Dream Mode Demo for Lumina V7")
    parser.add_argument("--duration", type=float, default=5,
                        help="Dream duration in minutes (default: 5)")
    parser.add_argument("--intensity", type=float, default=0.7,
                        help="Dream intensity 0.0-1.0 (default: 0.7)")
    args = parser.parse_args()
    
    # Validate args
    duration = max(1, min(30, args.duration))  # Between 1-30 minutes
    intensity = max(0.1, min(1.0, args.intensity))  # Between 0.1-1.0
    
    # Run the demo
    run_dream_cycle(duration=duration, intensity=intensity)

if __name__ == "__main__":
    main() 
"""
Dream Mode Demo for Lumina V7

This script demonstrates the functionality of the V7 Dream Mode system by:
1. Initializing the dream controller and related components
2. Running a complete dream cycle
3. Displaying insights and patterns generated during the dream
4. Showing how to retrieve dream records from the archive

Usage:
    python dream_mode_demo.py [--duration MINUTES] [--intensity LEVEL]
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dream_demo")

# Import Dream Mode components
from src.v7.lumina_v7.core.dream_controller import get_dream_controller
from src.v7.lumina_v7.core.memory_consolidator import MemoryConsolidator
from src.v7.lumina_v7.core.pattern_synthesizer import PatternSynthesizer
from src.v7.lumina_v7.core.dream_archive import DreamArchive

def setup_demo_environment():
    """Set up the environment for the dream mode demo"""
    # Create data directory
    demo_data_dir = project_root / "data" / "v7" / "demo" / "dreams"
    demo_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Return configuration
    return {
        "dream_enabled": True,
        "dream_data_dir": str(demo_data_dir),
        "auto_dream": False,  # Disable auto-dream for demo
        "default_dream_duration": 5,  # 5 minutes for demo
        "default_dream_intensity": 0.7
    }

def print_section(title):
    """Print a section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_json(data):
    """Print data as formatted JSON"""
    print(json.dumps(data, indent=2))

def run_dream_cycle(duration=None, intensity=0.7):
    """Run a complete dream cycle and display results"""
    print_section("Dream Mode Demo - V7 Node Consciousness")
    print(f"Starting demo with duration={duration} minutes, intensity={intensity}")
    
    # Set up the demo environment
    config = setup_demo_environment()
    
    # Initialize the dream controller
    print("\nInitializing Dream Controller...")
    dream_controller = get_dream_controller(config=config)
    
    # Display initial state
    print("\nInitial dream state:")
    initial_state = dream_controller.get_dream_state()
    print_json(initial_state)
    
    # Enter dream state
    print("\nEntering dream state...")
    dream_controller.enter_dream_state(duration=duration, intensity=intensity)
    
    # Display dream state
    dream_state = dream_controller.get_dream_state()
    print("\nDream state after entering:")
    print_json(dream_state)
    
    # Wait for dream to complete
    print("\nDream is active. Waiting for dream cycle to complete...")
    
    # Calculate wait time
    wait_duration = duration * 60 if duration else 300  # 5 minutes default
    
    # Monitor dream state while waiting
    start_time = time.time()
    while dream_controller.dream_active and time.time() - start_time < wait_duration:
        # Print current state every 10 seconds
        current_state = dream_controller.current_state
        print(f"Current dream phase: {current_state} (elapsed: {int(time.time() - start_time)}s)")
        
        # Sleep for a bit
        time.sleep(10)
    
    # Ensure dream is complete
    if dream_controller.dream_active:
        print("\nForcing exit from dream state...")
        dream_controller.exit_dream_state()
    
    # Get final dream state
    final_state = dream_controller.get_dream_state()
    print("\nFinal dream state:")
    print_json(final_state)
    
    # Get completed dream
    dream_id = dream_controller.current_dream.get("id") if dream_controller.current_dream else None
    
    if dream_id:
        print(f"\nRetrieving dream record {dream_id}...")
        dream = dream_controller.get_dream(dream_id)
        
        # Display insights
        if dream and "insights" in dream and dream["insights"]:
            print_section("Dream Insights")
            for i, insight in enumerate(dream["insights"]):
                print(f"Insight {i+1}: {insight.get('text', 'No text')}")
                print(f"Type: {insight.get('type', 'unknown')}")
                print(f"Confidence: {insight.get('confidence', 0):.2f}")
                print()
        else:
            print("\nNo insights were generated during this dream.")
        
        # Display patterns
        if dream and "synthesis_results" in dream:
            pattern_count = sum(
                len(result.get("patterns", [])) 
                for result in dream["synthesis_results"] 
                if isinstance(result, dict)
            )
            
            if pattern_count > 0:
                print_section(f"Dream Patterns ({pattern_count} total)")
                
                # Get a sample of patterns
                patterns = []
                for result in dream["synthesis_results"]:
                    if isinstance(result, dict) and "patterns" in result:
                        patterns.extend(result["patterns"])
                
                # Display up to 3 patterns
                for i, pattern in enumerate(patterns[:3]):
                    print(f"Pattern {i+1} - Type: {pattern.get('type', 'unknown')}")
                    print(f"Description: {pattern.get('description', 'No description')}")
                    print(f"Confidence: {pattern.get('confidence', 0):.2f}")
                    print()
                
                if pattern_count > 3:
                    print(f"... and {pattern_count - 3} more patterns")
            else:
                print("\nNo patterns were synthesized during this dream.")
    
    # Display dream archive stats
    print_section("Dream Archive Statistics")
    archive_stats = dream_controller.dream_archive.get_stats()
    print(f"Total Dreams: {archive_stats.get('total_dreams', 0)}")
    print(f"Total Insights: {archive_stats.get('insight_count', 0)}")
    print(f"Average Dream Duration: {archive_stats.get('average_dream_duration', 0):.1f} minutes")
    
    # Display most common topics
    if "most_common_topics" in archive_stats and archive_stats["most_common_topics"]:
        print("\nMost Common Topics:")
        for topic, count in archive_stats["most_common_topics"].items():
            print(f"  - {topic}: {count}")
    
    print("\nDemo completed successfully!")

def main():
    """Main entry point for the demo"""
    parser = argparse.ArgumentParser(description="Dream Mode Demo for Lumina V7")
    parser.add_argument("--duration", type=float, default=5,
                        help="Dream duration in minutes (default: 5)")
    parser.add_argument("--intensity", type=float, default=0.7,
                        help="Dream intensity 0.0-1.0 (default: 0.7)")
    args = parser.parse_args()
    
    # Validate args
    duration = max(1, min(30, args.duration))  # Between 1-30 minutes
    intensity = max(0.1, min(1.0, args.intensity))  # Between 0.1-1.0
    
    # Run the demo
    run_dream_cycle(duration=duration, intensity=intensity)

if __name__ == "__main__":
    main() 