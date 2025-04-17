import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
import json

class GameTheoryNode(nn.Module):
    def __init__(self, dimension=512, num_dimensions=5):
        super().__init__()
        self.dimension = dimension
        self.num_dimensions = num_dimensions
        
        # Calculate input dimension
        self.input_dim = 2 * 8 * 8 * num_dimensions  # [player, x, y, dimension]
        
        # Game state representation
        self.game_state = nn.Parameter(
            torch.zeros(2, 8, 8, num_dimensions, dtype=torch.float32)
        )
        
        # Piece evolution tracker (using float32 instead of long)
        self.piece_levels = nn.Parameter(
            torch.zeros(2, 8, 8, dtype=torch.float32)
        )
        
        # Quantum state processor
        self.quantum_processor = nn.Sequential(
            nn.Linear(self.input_dim, dimension),
            nn.ReLU(),
            nn.Linear(dimension, dimension),
            nn.Tanh()
        )
        
        # Timeline processor
        self.timeline_processor = nn.Sequential(
            nn.Linear(self.input_dim, dimension * 2),
            nn.ReLU(),
            nn.Linear(dimension * 2, dimension),
            nn.Tanh()
        )
        
        # Merge probability calculator
        self.merge_processor = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.ReLU(),
            nn.Linear(dimension, 1),
            nn.Sigmoid()
        )
        
        # Energy system
        self.energy_levels = nn.Parameter(torch.ones(2, dtype=torch.float32))
        
    def forward(self, game_state, player_idx):
        # Ensure game_state has correct shape
        if len(game_state.shape) == 5:  # [batch, player, x, y, dimension]
            batch_size = game_state.size(0)
        else:
            batch_size = 1
            game_state = game_state.unsqueeze(0)
            
        # Process quantum states
        quantum_state = self.process_quantum_state(game_state)
        
        # Process timeline branches
        timeline_state = self.process_timelines(game_state)
        
        # Calculate merge probabilities
        merge_probs = self.calculate_merge_probabilities(quantum_state, timeline_state)
        
        # Update energy levels
        self.update_energy(player_idx)
        
        return {
            'quantum_state': quantum_state,
            'timeline_state': timeline_state,
            'merge_probabilities': merge_probs,
            'energy_level': self.energy_levels[player_idx]
        }
    
    def process_quantum_state(self, game_state):
        # Project game state into quantum space
        batch_size = game_state.size(0)
        flattened = game_state.reshape(batch_size, -1)
        quantum_state = self.quantum_processor(flattened)
        return quantum_state
    
    def process_timelines(self, game_state):
        # Process potential timeline branches
        batch_size = game_state.size(0)
        flattened = game_state.reshape(batch_size, -1)
        timeline_state = self.timeline_processor(flattened)
        return timeline_state
    
    def calculate_merge_probabilities(self, quantum_state, timeline_state):
        # Calculate probability of successful merges
        combined = torch.cat([quantum_state, timeline_state], dim=-1)
        merge_probs = self.merge_processor(combined)
        return merge_probs
    
    def update_energy(self, player_idx, cost=0.1):
        # Update energy levels after moves
        with torch.no_grad():
            self.energy_levels[player_idx] = torch.clamp(
                self.energy_levels[player_idx] - cost,
                min=0.0,
                max=1.0
            )
    
    def merge_pieces(self, pos1: Tuple[int, int], pos2: Tuple[int, int], player_idx: int):
        # Attempt to merge pieces at given positions
        if not self._can_merge(pos1, pos2, player_idx):
            return False
            
        with torch.no_grad():
            level = self.piece_levels[player_idx, pos1[0], pos1[1]]
            self.piece_levels[player_idx, pos1[0], pos1[1]] = level + 1
            self.piece_levels[player_idx, pos2[0], pos2[1]] = 0
            
            # Regenerate energy on successful merge
            self.energy_levels[player_idx] = torch.clamp(
                self.energy_levels[player_idx] + 0.2,
                max=1.0
            )
            
        return True
    
    def _can_merge(self, pos1: Tuple[int, int], pos2: Tuple[int, int], player_idx: int):
        # Check if pieces can be merged
        level1 = self.piece_levels[player_idx, pos1[0], pos1[1]]
        level2 = self.piece_levels[player_idx, pos2[0], pos2[1]]
        return level1 > 0 and torch.abs(level1 - level2) < 1e-6  # Compare float values with tolerance
    
    def get_state(self):
        return {
            'game_state': self.game_state.detach().cpu().numpy(),
            'piece_levels': self.piece_levels.detach().cpu().numpy(),
            'energy_levels': self.energy_levels.detach().cpu().numpy()
        }
    
    def load_state(self, state_dict):
        self.game_state.data = torch.tensor(state_dict['game_state'], dtype=torch.float32)
        self.piece_levels.data = torch.tensor(state_dict['piece_levels'], dtype=torch.float32)
        self.energy_levels.data = torch.tensor(state_dict['energy_levels'], dtype=torch.float32) 