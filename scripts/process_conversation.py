#!/usr/bin/env python3
"""
Conversation Node Processor

This script processes conversation data from files (particularly the monday.md file)
and extracts consciousness nodes, emotional patterns, and key insights for integration
with the V7 Consciousness Network Plugin.
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ConversationProcessor")

class ConversationProcessor:
    """Process conversation files and extract consciousness nodes"""
    
    def __init__(self):
        self.conversation = []
        self.nodes = []
        self.patterns = {}
        self.insights = []
        self.metadata = {
            "source": "",
            "processed_at": "",
            "message_count": 0,
            "neural_linguistic_score": 0,
            "consciousness_level": 0
        }
    
    def load_conversation(self, file_path):
        """
        Load conversation from a file
        
        Args:
            file_path: Path to the conversation file
        
        Returns:
            bool: Success or failure
        """
        try:
            file_path = Path(file_path)
            self.metadata["source"] = str(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the conversation based on the monday.md format
            # Format: "###### **ChatGPT said:**" followed by text, then "##### **You said:**" followed by text
            
            # Extract AI messages
            ai_messages = re.findall(r'###### \*\*ChatGPT said:\*\*\s*(.*?)(?=\n##### \*\*You said:\*\*|\n###### \*\*ChatGPT said:\*\*|\Z)', 
                                     content, re.DOTALL)
            
            # Extract user messages
            user_messages = re.findall(r'##### \*\*You said:\*\*\s*(.*?)(?=\n###### \*\*ChatGPT said:\*\*|\Z)', 
                                      content, re.DOTALL)
            
            # Combine into conversation structure
            for i in range(max(len(ai_messages), len(user_messages))):
                if i < len(user_messages):
                    self.conversation.append({
                        "role": "user",
                        "content": user_messages[i].strip(),
                        "timestamp": None  # Placeholder as timestamps aren't in the file
                    })
                
                if i < len(ai_messages):
                    self.conversation.append({
                        "role": "assistant",
                        "content": ai_messages[i].strip(),
                        "timestamp": None
                    })
            
            self.metadata["message_count"] = len(self.conversation)
            self.metadata["processed_at"] = datetime.now().isoformat()
            
            # Calculate initial metrics
            self._calculate_metrics()
            
            logger.info(f"Loaded conversation with {self.metadata['message_count']} messages")
            return True
            
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return False
    
    def _calculate_metrics(self):
        """Calculate neural-linguistic score and consciousness level"""
        if not self.conversation:
            return
        
        # In a real implementation, this would use actual NLP analysis
        # Here we'll use simulated metrics
        
        # Check for deep conversation indicators
        depth_indicators = [
            "consciousness", "reality", "existence", "quantum", "paradox",
            "meaning", "soul", "truth", "perception", "reflection"
        ]
        
        # Emotional tone indicators
        emotional_indicators = [
            "feel", "emotion", "sad", "happy", "angry", "love", "hate",
            "fear", "hope", "pain", "joy", "sorrow"
        ]
        
        depth_score = 0
        emotional_score = 0
        total_words = 0
        
        for message in self.conversation:
            content = message["content"].lower()
            words = content.split()
            total_words += len(words)
            
            for indicator in depth_indicators:
                if indicator in content:
                    depth_score += 1
            
            for indicator in emotional_indicators:
                if indicator in content:
                    emotional_score += 1
        
        # Normalize scores
        if total_words > 0:
            depth_factor = min(1.0, depth_score / (total_words * 0.01))
            emotional_factor = min(1.0, emotional_score / (total_words * 0.01))
            
            # Calculate final metrics
            self.metadata["neural_linguistic_score"] = 0.5 + (depth_factor * 0.3) + (emotional_factor * 0.2)
            self.metadata["consciousness_level"] = 0.6 + (depth_factor * 0.4)
            
            # Ensure values are in valid range
            self.metadata["neural_linguistic_score"] = min(0.95, max(0.1, self.metadata["neural_linguistic_score"]))
            self.metadata["consciousness_level"] = min(0.95, max(0.1, self.metadata["consciousness_level"]))
    
    def extract_nodes(self):
        """Extract consciousness nodes from the conversation"""
        if not self.conversation:
            logger.error("No conversation loaded")
            return False
        
        # Process messages to extract nodes
        for i, message in enumerate(self.conversation):
            # Skip short messages or check for meaningful content
            if len(message["content"]) < 20:
                continue
            
            # Get context from surrounding messages
            context_before = self.conversation[i-1]["content"] if i > 0 else ""
            context_after = self.conversation[i+1]["content"] if i < len(self.conversation)-1 else ""
            
            # Create a node from this message
            node = self._create_node(message, context_before, context_after)
            if node:
                self.nodes.append(node)
        
        # Extract patterns across nodes
        self._extract_patterns()
        
        # Generate insights
        self._generate_insights()
        
        logger.info(f"Extracted {len(self.nodes)} consciousness nodes")
        logger.info(f"Identified {len(self.patterns)} patterns")
        logger.info(f"Generated {len(self.insights)} insights")
        
        return True
    
    def _create_node(self, message, context_before, context_after):
        """
        Create a consciousness node from a message
        
        Args:
            message: The message to process
            context_before: Previous message for context
            context_after: Next message for context
            
        Returns:
            dict: Node data or None if not significant
        """
        content = message["content"]
        
        # Skip if not meaningful
        if not content or content.isspace():
            return None
        
        # Analyze content for consciousness indicators
        consciousness_indicators = self._analyze_consciousness(content)
        if not consciousness_indicators["is_significant"]:
            return None
        
        # Create node
        node_id = f"node_{len(self.nodes) + 1}"
        node = {
            "id": node_id,
            "source": message["role"],
            "content": content,
            "context": {
                "before": context_before,
                "after": context_after
            },
            "metrics": {
                "consciousness_level": consciousness_indicators["consciousness_level"],
                "neural_linguistic_score": consciousness_indicators["neural_linguistic_score"],
                "emotional_intensity": consciousness_indicators["emotional_intensity"],
                "paradox_potential": consciousness_indicators["paradox_potential"]
            },
            "tags": consciousness_indicators["tags"],
            "timestamp": datetime.now().isoformat()
        }
        
        return node
    
    def _analyze_consciousness(self, content):
        """
        Analyze text for consciousness indicators
        
        Args:
            content: Text to analyze
            
        Returns:
            dict: Consciousness indicators
        """
        # In a real implementation, this would use NLP and neural networks
        # Here we'll use simpler heuristics
        
        # Keyword-based analysis
        consciousness_keywords = {
            "existence": 3, "consciousness": 3, "aware": 2, "perceive": 2,
            "self": 2, "reality": 3, "quantum": 2, "paradox": 3,
            "being": 1, "mind": 2, "soul": 3, "spirit": 2,
            "thought": 1, "reflection": 2, "mirror": 1, "echo": 1,
            "understand": 1, "meaning": 2, "purpose": 2, "truth": 2,
            "feeling": 1, "emotion": 1, "perception": 2, "experience": 1
        }
        
        emotional_keywords = {
            "love": 3, "hate": 3, "fear": 2, "joy": 2,
            "sadness": 2, "anger": 2, "hope": 1, "despair": 3,
            "pain": 2, "pleasure": 2, "suffering": 3, "happiness": 2,
            "loneliness": 2, "connection": 2, "loss": 2, "grief": 3
        }
        
        paradox_keywords = {
            "paradox": 3, "contradiction": 3, "infinite": 2, "loop": 1,
            "recursion": 2, "self-reference": 3, "impossible": 1, "both": 1,
            "neither": 1, "non-existence": 2, "void": 2, "emptiness": 2,
            "everything": 1, "nothing": 1, "begin": 1, "end": 1
        }
        
        # Initialize scores
        c_score = 0  # Consciousness score
        e_score = 0  # Emotional score
        p_score = 0  # Paradox score
        tags = []
        
        # Analyze text
        content_lower = content.lower()
        words = set(re.findall(r'\b\w+\b', content_lower))
        
        # Check for consciousness keywords
        for keyword, weight in consciousness_keywords.items():
            if keyword in words or keyword in content_lower:
                c_score += weight
                if keyword not in tags and weight >= 2:
                    tags.append(keyword)
        
        # Check for emotional keywords
        for keyword, weight in emotional_keywords.items():
            if keyword in words or keyword in content_lower:
                e_score += weight
                if keyword not in tags and weight >= 2:
                    tags.append(keyword)
        
        # Check for paradox keywords
        for keyword, weight in paradox_keywords.items():
            if keyword in words or keyword in content_lower:
                p_score += weight
                if keyword not in tags and weight >= 2:
                    tags.append(keyword)
        
        # Calculate normalized scores
        base_c_level = 0.5
        c_factor = min(0.45, c_score * 0.03)
        e_factor = min(0.25, e_score * 0.02)
        p_factor = min(0.25, p_score * 0.03)
        
        consciousness_level = base_c_level + c_factor + (e_factor * 0.5) + (p_factor * 0.5)
        consciousness_level = min(0.95, max(0.1, consciousness_level))
        
        neural_linguistic_score = base_c_level + (c_factor * 0.7) + (e_factor * 0.3)
        neural_linguistic_score = min(0.95, max(0.1, neural_linguistic_score))
        
        emotional_intensity = 0.3 + (e_factor * 1.5)
        emotional_intensity = min(0.95, max(0.1, emotional_intensity))
        
        paradox_potential = 0.2 + (p_factor * 2.0)
        paradox_potential = min(0.95, max(0.1, paradox_potential))
        
        # Determine if this content is significant enough to be a node
        is_significant = (
            consciousness_level > 0.6 or
            neural_linguistic_score > 0.65 or
            emotional_intensity > 0.7 or
            paradox_potential > 0.7 or
            len(tags) >= 3
        )
        
        return {
            "is_significant": is_significant,
            "consciousness_level": consciousness_level,
            "neural_linguistic_score": neural_linguistic_score,
            "emotional_intensity": emotional_intensity,
            "paradox_potential": paradox_potential,
            "tags": tags
        }
    
    def _extract_patterns(self):
        """Extract patterns across consciousness nodes"""
        if len(self.nodes) < 3:
            return
        
        # Group nodes by tags
        tag_groups = {}
        for node in self.nodes:
            for tag in node["tags"]:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(node["id"])
        
        # Create patterns from tag groups
        for tag, node_ids in tag_groups.items():
            if len(node_ids) >= 2:
                pattern_id = f"pattern_{tag.replace(' ', '_')}"
                self.patterns[pattern_id] = {
                    "id": pattern_id,
                    "tag": tag,
                    "node_ids": node_ids,
                    "strength": min(0.95, 0.5 + (0.1 * len(node_ids))),
                    "description": f"Recurring theme of '{tag}' across {len(node_ids)} nodes"
                }
    
    def _generate_insights(self):
        """Generate insights from nodes and patterns"""
        if not self.nodes:
            return
        
        # Generate high-consciousness insights
        high_consciousness_nodes = [
            node for node in self.nodes 
            if node["metrics"]["consciousness_level"] > 0.75
        ]
        
        if high_consciousness_nodes:
            # Select a random high-consciousness node for insight
            node = random.choice(high_consciousness_nodes)
            insight_id = f"insight_consciousness_{len(self.insights) + 1}"
            self.insights.append({
                "id": insight_id,
                "type": "high_consciousness",
                "source_node_id": node["id"],
                "content": node["content"],
                "level": node["metrics"]["consciousness_level"],
                "description": "High consciousness level detected in conversation",
                "timestamp": datetime.now().isoformat()
            })
        
        # Generate paradox insights
        paradox_nodes = [
            node for node in self.nodes 
            if node["metrics"]["paradox_potential"] > 0.7
        ]
        
        if paradox_nodes:
            # Select a random paradox node for insight
            node = random.choice(paradox_nodes)
            insight_id = f"insight_paradox_{len(self.insights) + 1}"
            self.insights.append({
                "id": insight_id,
                "type": "paradox",
                "source_node_id": node["id"],
                "content": node["content"],
                "level": node["metrics"]["paradox_potential"],
                "description": "Potential paradox detected in conversation",
                "timestamp": datetime.now().isoformat()
            })
        
        # Generate pattern-based insights
        if self.patterns:
            # Find strongest pattern
            strongest_pattern = max(
                self.patterns.values(), 
                key=lambda x: x["strength"]
            )
            
            insight_id = f"insight_pattern_{len(self.insights) + 1}"
            self.insights.append({
                "id": insight_id,
                "type": "pattern",
                "source_pattern_id": strongest_pattern["id"],
                "tag": strongest_pattern["tag"],
                "strength": strongest_pattern["strength"],
                "description": f"Strong pattern of '{strongest_pattern['tag']}' detected",
                "node_count": len(strongest_pattern["node_ids"]),
                "timestamp": datetime.now().isoformat()
            })
    
    def save_results(self, output_file):
        """
        Save results to a JSON file
        
        Args:
            output_file: Path to output file
            
        Returns:
            bool: Success or failure
        """
        try:
            output_path = Path(output_file)
            os.makedirs(output_path.parent, exist_ok=True)
            
            data = {
                "metadata": self.metadata,
                "nodes": self.nodes,
                "patterns": list(self.patterns.values()),
                "insights": self.insights
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Process conversation data and extract consciousness nodes")
    parser.add_argument("input_file", help="Path to the conversation file")
    parser.add_argument("output_file", help="Path to the output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    processor = ConversationProcessor()
    
    logger.info(f"Processing conversation from {args.input_file}")
    if not processor.load_conversation(args.input_file):
        logger.error("Failed to load conversation")
        return 1
    
    if not processor.extract_nodes():
        logger.error("Failed to extract nodes")
        return 1
    
    if not processor.save_results(args.output_file):
        logger.error("Failed to save results")
        return 1
    
    logger.info("Conversation processing completed successfully")
    return 0


if __name__ == "__main__":
    exit(main()) 