from typing import Dict, Any, List
import logging
from .base_node import BaseNode
import numpy as np
from datetime import datetime
import json
import os

class LearningNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.active = False
        self.learning_rate = 0.01
        self.learning_state = {}
        self.knowledge_base = {}
        self.adaptation_threshold = 0.7
        self.knowledge_path = "data/knowledge"
        
    def initialize(self) -> bool:
        """Initialize the learning node"""
        try:
            # Create knowledge directory if it doesn't exist
            os.makedirs(self.knowledge_path, exist_ok=True)
            
            # Initialize learning state
            self.learning_state = {
                'current_focus': None,
                'learning_progress': 0.0,
                'adaptation_level': 0.0,
                'last_update': datetime.now().isoformat()
            }
            
            # Load existing knowledge
            self._load_knowledge()
            
            self.active = True
            logging.info("LearningNode initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize LearningNode: {str(e)}")
            return False
            
    def _load_knowledge(self):
        """Load existing knowledge from storage"""
        try:
            knowledge_files = [f for f in os.listdir(self.knowledge_path) 
                             if f.endswith('.json')]
            for file in knowledge_files:
                with open(os.path.join(self.knowledge_path, file), 'r') as f:
                    knowledge_data = json.load(f)
                    self.knowledge_base.update(knowledge_data)
        except Exception as e:
            logging.error(f"Error loading knowledge: {str(e)}")
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input for learning operations"""
        try:
            operation = input_data.get('operation')
            if operation == 'learn':
                return self._learn_from_data(input_data.get('data', {}))
            elif operation == 'adapt':
                return self._adapt_to_feedback(input_data.get('feedback', {}))
            elif operation == 'query':
                return self._query_knowledge(input_data.get('query', {}))
            else:
                return {'error': 'Invalid operation'}
        except Exception as e:
            logging.error(f"Error processing learning operation: {str(e)}")
            return {'error': str(e)}
            
    def _learn_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from new data"""
        try:
            # Update learning state
            self.learning_state['current_focus'] = data.get('topic')
            
            # Extract features and patterns
            features = self._extract_features(data)
            
            # Update knowledge base
            topic = data.get('topic', 'general')
            if topic not in self.knowledge_base:
                self.knowledge_base[topic] = {
                    'features': [],
                    'patterns': [],
                    'confidence': 0.0
                }
                
            # Add new features
            self.knowledge_base[topic]['features'].extend(features)
            
            # Update confidence
            current_confidence = self.knowledge_base[topic]['confidence']
            new_confidence = min(1.0, current_confidence + self.learning_rate)
            self.knowledge_base[topic]['confidence'] = new_confidence
            
            # Save updated knowledge
            self._save_knowledge(topic)
            
            # Update learning progress
            self.learning_state['learning_progress'] = new_confidence
            self.learning_state['last_update'] = datetime.now().isoformat()
            
            return {
                'status': 'success',
                'topic': topic,
                'confidence': new_confidence,
                'features_learned': len(features)
            }
        except Exception as e:
            logging.error(f"Error learning from data: {str(e)}")
            return {'error': str(e)}
            
    def _adapt_to_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt based on feedback"""
        try:
            topic = feedback.get('topic', 'general')
            if topic not in self.knowledge_base:
                return {'error': 'Topic not found in knowledge base'}
                
            # Calculate adaptation score
            feedback_score = feedback.get('score', 0.0)
            current_confidence = self.knowledge_base[topic]['confidence']
            
            if feedback_score > self.adaptation_threshold:
                # Positive feedback - reinforce knowledge
                new_confidence = min(1.0, current_confidence + self.learning_rate)
            else:
                # Negative feedback - adjust knowledge
                new_confidence = max(0.0, current_confidence - self.learning_rate)
                
            self.knowledge_base[topic]['confidence'] = new_confidence
            
            # Update adaptation level
            self.learning_state['adaptation_level'] = new_confidence
            self.learning_state['last_update'] = datetime.now().isoformat()
            
            # Save changes
            self._save_knowledge(topic)
            
            return {
                'status': 'success',
                'topic': topic,
                'new_confidence': new_confidence,
                'adaptation_level': self.learning_state['adaptation_level']
            }
        except Exception as e:
            logging.error(f"Error adapting to feedback: {str(e)}")
            return {'error': str(e)}
            
    def _query_knowledge(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Query existing knowledge"""
        try:
            topic = query.get('topic')
            if topic and topic in self.knowledge_base:
                return {
                    'status': 'success',
                    'topic': topic,
                    'knowledge': self.knowledge_base[topic]
                }
            elif not topic:
                # Return overview of all topics
                return {
                    'status': 'success',
                    'topics': list(self.knowledge_base.keys()),
                    'total_confidence': np.mean([
                        k['confidence'] for k in self.knowledge_base.values()
                    ])
                }
            else:
                return {'error': 'Topic not found'}
        except Exception as e:
            logging.error(f"Error querying knowledge: {str(e)}")
            return {'error': str(e)}
            
    def _extract_features(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract features from input data"""
        features = []
        try:
            # Extract numerical features
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    features.append({
                        'type': 'numerical',
                        'name': key,
                        'value': value
                    })
                elif isinstance(value, str):
                    features.append({
                        'type': 'categorical',
                        'name': key,
                        'value': value
                    })
                elif isinstance(value, list):
                    features.append({
                        'type': 'sequence',
                        'name': key,
                        'value': value
                    })
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
        return features
        
    def _save_knowledge(self, topic: str):
        """Save knowledge to file"""
        try:
            file_path = os.path.join(self.knowledge_path, f"{topic}.json")
            with open(file_path, 'w') as f:
                json.dump({topic: self.knowledge_base[topic]}, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving knowledge to file: {str(e)}")
            
    def get_status(self) -> str:
        """Get current status of the learning node"""
        if not self.active:
            return "inactive"
        return (f"active (topics: {len(self.knowledge_base)}, "
                f"learning progress: {self.learning_state['learning_progress']:.2f})")
        
    def is_active(self) -> bool:
        """Check if the node is active"""
        return self.active 