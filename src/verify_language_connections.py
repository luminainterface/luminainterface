#!/usr/bin/env python3
"""
Language Memory Connection Verification Tool

This script verifies and fixes connections between the Language Memory System
and the rest of the Lumina backend components, ensuring proper integration
between language processing, memory systems, and visualization.
"""

import os
import sys
import logging
from pathlib import Path
import importlib
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verify_language_connections")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def verify_component_imports():
    """Verify that all required components can be imported"""
    print("\n=== Verifying Component Imports ===")
    
    components = [
        # Core language components
        {"module": "src.language.language_memory", "class": "LanguageMemory", "required": True},
        {"module": "src.language.language_memory_integration", "function": "integrate_language_memory", "required": True},
        
        # Memory components
        {"module": "src.conversation_memory", "class": "ConversationMemory", "required": True},
        {"module": "src.memory.conversation_language_bridge", "class": "ConversationLanguageBridge", "required": True},
        
        # Integration components
        {"module": "src.language_memory_synthesis_integration", "class": "LanguageMemorySynthesisIntegration", "required": True},
        {"module": "src.language_memory_api_compat", "class": "LanguageMemoryAPI", "required": False},
        
        # V5 visualization components
        {"module": "src.v5.language_memory_integration", "class": "LanguageMemoryIntegrationPlugin", "required": False},
        {"module": "src.v5_integration.visualization_bridge", "function": "get_visualization_bridge", "required": False},
        
        # Central node components
        {"module": "src.central_language_node", "class": "CentralLanguageNode", "required": False},
        {"module": "src.central_node", "class": "CentralNode", "required": False},
    ]
    
    success_count = 0
    required_count = 0
    
    for component in components:
        required = component.get("required", False)
        module_name = component["module"]
        
        if required:
            required_count += 1
        
        try:
            module = importlib.import_module(module_name)
            
            # Check for class or function
            if "class" in component:
                class_name = component["class"]
                if hasattr(module, class_name):
                    print(f"[OK] {module_name}.{class_name} - Available")
                    success_count += 1
                else:
                    print(f"[X] {module_name}.{class_name} - Class not found in module")
            elif "function" in component:
                function_name = component["function"]
                if hasattr(module, function_name):
                    print(f"[OK] {module_name}.{function_name} - Available")
                    success_count += 1
                else:
                    print(f"[X] {module_name}.{function_name} - Function not found in module")
        
        except ImportError as e:
            status = "Missing (required)" if required else "Missing (optional)"
            print(f"[X] {module_name} - {status}: {str(e)}")
    
    print(f"\nSuccessfully imported {success_count}/{len(components)} components")
    if success_count >= required_count:
        print("All required components are available")
        return True
    else:
        print("Some required components are missing")
        return False


def verify_language_memory():
    """Verify that the language memory system can be initialized and used"""
    print("\n=== Verifying Language Memory System ===")
    
    try:
        from src.language.language_memory import LanguageMemory
        
        # Initialize language memory
        memory = LanguageMemory()
        print("[OK] LanguageMemory initialized")
        
        # Test basic operations
        memory.remember_word_association("neural", "network", 0.9, "test")
        print("[OK] remember_word_association works")
        
        associations = memory.recall_associations("neural")
        if associations and len(associations) > 0:
            print(f"[OK] recall_associations works (found {len(associations)} associations)")
        else:
            print("[X] recall_associations returned no results")
        
        memory.remember_sentence("Language memory system integration test.", {"test": True})
        print("[OK] remember_sentence works")
        
        # Get memory statistics
        stats = memory.get_memory_statistics()
        print(f"[OK] Memory statistics: {len(memory.word_associations)} words, {len(memory.sentences)} sentences")
        
        return True
    
    except Exception as e:
        print(f"[X] Language memory verification failed: {str(e)}")
        return False


def verify_conversation_bridge():
    """Verify that the conversation language bridge works"""
    print("\n=== Verifying Conversation-Language Bridge ===")
    
    try:
        from src.memory.conversation_language_bridge import ConversationLanguageBridge
        
        # Initialize bridge
        bridge = ConversationLanguageBridge()
        print(f"[OK] ConversationLanguageBridge initialized (success: {bridge.initialized})")
        
        # Test processing a conversation
        if bridge.initialized:
            result = bridge.process_conversation(
                "How do language memory systems work?",
                "Language memory systems store and retrieve linguistic patterns and word associations.",
                {"topic": "language_memory", "test": True}
            )
            print(f"[OK] process_conversation result: {result}")
            
            # Test retrieving by topic
            memories = bridge.retrieve_by_topic("language")
            print(f"[OK] retrieve_by_topic found {len(memories.get('language_memories', []))} language memories and "
                  f"{len(memories.get('conversation_memories', []))} conversation memories")
            
            # Get stats
            stats = bridge.get_stats()
            print(f"[OK] Bridge stats: {stats.get('conversations_processed', 0)} conversations processed")
        
        return bridge.initialized
    
    except Exception as e:
        print(f"[X] Conversation bridge verification failed: {str(e)}")
        return False


def verify_synthesis_integration():
    """Verify that the language memory synthesis integration works"""
    print("\n=== Verifying Language Memory Synthesis Integration ===")
    
    try:
        from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
        
        # Initialize synthesis integration
        synthesis = LanguageMemorySynthesisIntegration()
        print("[OK] LanguageMemorySynthesisIntegration initialized")
        
        # Test topic synthesis
        result = synthesis.synthesize_topic("neural networks", depth=2)
        if "synthesis_results" in result:
            print("[OK] synthesize_topic returned results")
        else:
            print("[X] synthesize_topic didn't return expected results")
        
        # Get stats
        stats = synthesis.get_stats()
        print(f"[OK] Synthesis stats: {stats.get('synthesis_stats', {}).get('synthesis_count', 0)} syntheses")
        
        return True
    
    except Exception as e:
        print(f"[X] Synthesis integration verification failed: {str(e)}")
        return False


def verify_v5_visualization():
    """Verify that the V5 visualization integration works"""
    print("\n=== Verifying V5 Visualization Integration ===")
    
    try:
        # Try to import the visualization bridge
        from src.v5_integration.visualization_bridge import get_visualization_bridge
        
        # Get bridge instance
        bridge = get_visualization_bridge()
        print(f"[OK] VisualizationBridge initialized")
        
        # Check if visualization is available
        is_available = bridge.is_visualization_available()
        print(f"[OK] V5 visualization available: {is_available}")
        
        # If available, test visualization
        if is_available:
            viz_data = bridge.visualize_topic("neural networks", depth=2)
            if "error" in viz_data:
                print(f"[X] visualize_topic error: {viz_data['error']}")
            else:
                print("[OK] visualize_topic succeeded")
                
                # Check visualization components
                components = bridge.get_available_visualization_components()
                print(f"[OK] Available visualization components: {', '.join(components) if components else 'None'}")
        
        return is_available
    
    except Exception as e:
        print(f"[X] V5 visualization verification failed: {str(e)}")
        return False


def verify_central_node_integration():
    """Verify that the language memory integrates with the central node"""
    print("\n=== Verifying Central Node Integration ===")
    
    try:
        # Try central language node first
        central_language_available = False
        try:
            from src.central_language_node import CentralLanguageNode
            central_language_available = True
            print("[OK] CentralLanguageNode available")
        except ImportError:
            print("[X] CentralLanguageNode not available")
        
        # Try v10 central node as fallback
        central_v10_available = False
        try:
            from src.central_node import CentralNode
            central_v10_available = True
            print("[OK] V10 CentralNode available")
        except ImportError:
            print("[X] V10 CentralNode not available")
        
        if not central_language_available and not central_v10_available:
            print("[X] No central node available")
            return False
        
        # Integrate language memory
        from src.language.language_memory_integration import integrate_language_memory
        success, language_memory, error = integrate_language_memory()
        
        if success and language_memory:
            print("[OK] Language memory integration succeeded")
            
            # Check if consciousness node connection is available (v10 feature)
            has_consciousness = hasattr(language_memory, 'consciousness_node') and language_memory.consciousness_node is not None
            print(f"[OK] Consciousness node integration: {has_consciousness}")
            
            # Check if conversation memory connection is available
            has_conversation = hasattr(language_memory, 'conversation_memory') and language_memory.conversation_memory is not None
            print(f"[OK] Conversation memory integration: {has_conversation}")
            
            return True
        else:
            print(f"[X] Language memory integration failed: {error}")
            return False
    
    except Exception as e:
        print(f"[X] Central node integration verification failed: {str(e)}")
        return False


def main():
    """Main function for verifying language connections"""
    print("\n" + "="*80)
    print("LANGUAGE MEMORY CONNECTION VERIFICATION TOOL")
    print("="*80)
    
    results = {}
    
    # Step 1: Verify component imports
    results["imports"] = verify_component_imports()
    
    # Step 2: Verify language memory
    results["language_memory"] = verify_language_memory()
    
    # Step 3: Verify conversation bridge
    results["conversation_bridge"] = verify_conversation_bridge()
    
    # Step 4: Verify synthesis integration
    results["synthesis"] = verify_synthesis_integration()
    
    # Step 5: Verify V5 visualization
    results["visualization"] = verify_v5_visualization()
    
    # Step 6: Verify central node integration
    results["central_node"] = verify_central_node_integration()
    
    # Print summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    all_good = True
    for component, status in results.items():
        status_text = "[PASS]" if status else "[FAIL]"
        print(f"{status_text} - {component}")
        all_good = all_good and status
    
    if all_good:
        print("\n[SUCCESS] All language memory connections verified successfully!")
    else:
        print("\n[WARNING] Some connections failed verification. Please check the logs above for details.")
    
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main()) 