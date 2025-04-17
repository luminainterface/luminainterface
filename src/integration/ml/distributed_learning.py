#!/usr/bin/env python3
"""
Distributed Learning System

This module implements a distributed learning system where nodes can teach each other
through knowledge sharing, model distillation, and collaborative learning.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from .core import BaseModel, MLConfig
from .models.transformer import TransformerModel

logger = logging.getLogger(__name__)

@dataclass
class NodeConfig:
    """Configuration for learning nodes"""
    node_id: str
    model_type: str = "transformer"
    learning_rate: float = 0.001
    knowledge_threshold: float = 0.8
    teach_interval: int = 100  # steps between teaching attempts
    max_students: int = 5  # maximum number of simultaneous students
    distillation_temperature: float = 2.0
    collaboration_weight: float = 0.3

class LearningNode:
    """A node in the distributed learning system"""
    
    def __init__(
        self,
        config: NodeConfig,
        model: Optional[BaseModel] = None
    ):
        self.config = config
        self.model = model or self._create_model()
        self.knowledge_base = {}
        self.students: List[LearningNode] = []
        self.teachers: List[LearningNode] = []
        self.learning_history: List[Dict[str, float]] = []
        self.teaching_history: List[Dict[str, Any]] = []
        
    def _create_model(self) -> BaseModel:
        """Create model for this node"""
        if self.config.model_type == "transformer":
            return TransformerModel(MLConfig())
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
            
    async def learn(
        self,
        data: torch.Tensor,
        teacher: Optional['LearningNode'] = None
    ) -> Dict[str, float]:
        """Learn from data or teacher"""
        metrics = {}
        
        if teacher is not None:
            # Learn through knowledge distillation
            metrics = await self._learn_from_teacher(data, teacher)
        else:
            # Direct learning from data
            metrics = await self._learn_from_data(data)
            
        self.learning_history.append(metrics)
        return metrics
        
    async def teach(
        self,
        student: 'LearningNode',
        data: torch.Tensor
    ) -> Dict[str, float]:
        """Teach another node"""
        if len(self.students) >= self.config.max_students:
            raise ValueError("Maximum number of students reached")
            
        if student not in self.students:
            self.students.append(student)
            student.teachers.append(self)
            
        # Generate teacher predictions
        with torch.no_grad():
            teacher_logits = self.model(data)
            
        # Have student learn from teacher's knowledge
        metrics = await student.learn(data, teacher=self)
        
        # Record teaching interaction
        self.teaching_history.append({
            'student_id': student.config.node_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        return metrics
        
    async def _learn_from_teacher(
        self,
        data: torch.Tensor,
        teacher: 'LearningNode'
    ) -> Dict[str, float]:
        """Learn from teacher through knowledge distillation"""
        # Get teacher's predictions
        with torch.no_grad():
            teacher_logits = teacher.model(data)
            teacher_probs = F.softmax(
                teacher_logits / self.config.distillation_temperature,
                dim=-1
            )
            
        # Student forward pass
        student_logits = self.model(data)
        student_probs = F.softmax(
            student_logits / self.config.distillation_temperature,
            dim=-1
        )
        
        # Calculate distillation loss
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.config.distillation_temperature, dim=-1),
            teacher_probs,
            reduction='batchmean'
        ) * (self.config.distillation_temperature ** 2)
        
        # Update student model
        self.model.zero_grad()
        distillation_loss.backward()
        
        # Calculate metrics
        with torch.no_grad():
            accuracy = (student_probs.argmax(dim=-1) == teacher_probs.argmax(dim=-1)).float().mean()
            
        return {
            'loss': distillation_loss.item(),
            'accuracy': accuracy.item(),
            'teacher_id': teacher.config.node_id
        }
        
    async def _learn_from_data(
        self,
        data: torch.Tensor
    ) -> Dict[str, float]:
        """Direct learning from data"""
        # Forward pass
        outputs = self.model(data)
        loss = F.cross_entropy(outputs, data)
        
        # Backward pass and update
        self.model.zero_grad()
        loss.backward()
        
        # Calculate metrics
        with torch.no_grad():
            accuracy = (outputs.argmax(dim=-1) == data).float().mean()
            
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
        
    async def collaborate(
        self,
        peers: List['LearningNode'],
        data: torch.Tensor
    ) -> Dict[str, float]:
        """Collaborate with peer nodes"""
        # Collect predictions from all peers
        peer_predictions = []
        for peer in peers:
            with torch.no_grad():
                peer_logits = peer.model(data)
                peer_predictions.append(F.softmax(peer_logits, dim=-1))
                
        # Average peer predictions
        ensemble_predictions = torch.stack(peer_predictions).mean(dim=0)
        
        # Learn from ensemble knowledge
        student_logits = self.model(data)
        student_probs = F.softmax(student_logits, dim=-1)
        
        # Calculate collaboration loss
        collaboration_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            ensemble_predictions,
            reduction='batchmean'
        ) * self.config.collaboration_weight
        
        # Update model
        self.model.zero_grad()
        collaboration_loss.backward()
        
        # Calculate metrics
        with torch.no_grad():
            accuracy = (student_probs.argmax(dim=-1) == ensemble_predictions.argmax(dim=-1)).float().mean()
            
        return {
            'loss': collaboration_loss.item(),
            'accuracy': accuracy.item(),
            'num_peers': len(peers)
        }
        
    def get_knowledge_state(self) -> Dict[str, Any]:
        """Get current knowledge state"""
        return {
            'node_id': self.config.node_id,
            'model_type': self.config.model_type,
            'num_students': len(self.students),
            'num_teachers': len(self.teachers),
            'learning_history': self.learning_history[-10:],  # Last 10 learning events
            'teaching_history': self.teaching_history[-10:],  # Last 10 teaching events
            'knowledge_level': self._calculate_knowledge_level()
        }
        
    def _calculate_knowledge_level(self) -> float:
        """Calculate current knowledge level"""
        if not self.learning_history:
            return 0.0
            
        # Use recent learning history
        recent_history = self.learning_history[-100:]  # Last 100 learning events
        if not recent_history:
            return 0.0
            
        # Calculate average accuracy
        accuracies = [event['accuracy'] for event in recent_history]
        return sum(accuracies) / len(accuracies)

class DistributedLearningSystem:
    """System managing distributed learning nodes"""
    
    def __init__(self):
        self.nodes: Dict[str, LearningNode] = {}
        self.learning_tasks: List[asyncio.Task] = []
        
    def add_node(self, node: LearningNode) -> None:
        """Add node to system"""
        self.nodes[node.config.node_id] = node
        
    def remove_node(self, node_id: str) -> None:
        """Remove node from system"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            # Remove node from students and teachers lists
            for student in node.students:
                student.teachers.remove(node)
            for teacher in node.teachers:
                teacher.students.remove(node)
            del self.nodes[node_id]
            
    async def start_learning(self, data: torch.Tensor) -> None:
        """Start distributed learning process"""
        tasks = []
        for node in self.nodes.values():
            # Create learning task for each node
            task = asyncio.create_task(self._node_learning_loop(node, data))
            tasks.append(task)
            
        self.learning_tasks.extend(tasks)
        await asyncio.gather(*tasks)
        
    async def _node_learning_loop(
        self,
        node: LearningNode,
        data: torch.Tensor
    ) -> None:
        """Learning loop for a single node"""
        while True:
            try:
                # Direct learning
                await node.learn(data)
                
                # Teaching phase
                if len(node.students) < node.config.max_students:
                    # Find potential students
                    potential_students = [
                        n for n in self.nodes.values()
                        if n != node and n not in node.students
                        and len(n.teachers) < n.config.max_students
                    ]
                    
                    if potential_students:
                        student = potential_students[0]
                        await node.teach(student, data)
                        
                # Collaboration phase
                peers = [n for n in self.nodes.values() if n != node]
                if peers:
                    await node.collaborate(peers, data)
                    
                await asyncio.sleep(1)  # Prevent overwhelming the system
                
            except Exception as e:
                logger.error(f"Error in node {node.config.node_id}: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    def get_system_state(self) -> Dict[str, Any]:
        """Get current state of the learning system"""
        return {
            'num_nodes': len(self.nodes),
            'nodes': {
                node_id: node.get_knowledge_state()
                for node_id, node in self.nodes.items()
            },
            'total_learning_events': sum(
                len(node.learning_history)
                for node in self.nodes.values()
            ),
            'total_teaching_events': sum(
                len(node.teaching_history)
                for node in self.nodes.values()
            )
        }

# Create global instance
distributed_learning = DistributedLearningSystem() 