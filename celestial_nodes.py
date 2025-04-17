import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from enum import Enum

class CelestialBodyType(Enum):
    PLANET = 0
    STAR = 1
    MOON = 2
    ASTEROID = 3
    COMET = 4

class AstrologicalNode(nn.Module):
    def __init__(self, dimension: int = 256):
        super().__init__()
        self.dimension = dimension
        
        # Zodiac sign embeddings (12 signs)
        self.zodiac_embeddings = nn.Parameter(
            torch.randn(12, dimension) / np.sqrt(dimension)  # Normalized initialization
        )
        
        # House embeddings (12 houses)
        self.house_embeddings = nn.Parameter(
            torch.randn(12, dimension) / np.sqrt(dimension)  # Normalized initialization
        )
        
        # Aspect calculator
        self.aspect_network = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
            nn.Tanh()
        )
        
        # Planetary influence
        self.planetary_influence = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Softmax(dim=-1)
        )
        
        # Elemental balance (Fire, Earth, Air, Water)
        self.elemental_balance = nn.Sequential(
            nn.Linear(dimension, 4),
            nn.Softmax(dim=-1)
        )
        
        # Aspect strength calculator
        self.aspect_strength = nn.Sequential(
            nn.Linear(dimension, 1),
            nn.Sigmoid()
        )

    def calculate_aspect(self, 
                        body1: torch.Tensor, 
                        body2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate astrological aspects between two celestial bodies
        
        Args:
            body1: First celestial body tensor [batch_size, dimension]
            body2: Second celestial body tensor [batch_size, dimension]
            
        Returns:
            Dictionary containing aspect, strength, and elemental balance
            
        Raises:
            ValueError: If input tensors have incorrect dimensions
        """
        if body1.dim() != 2 or body2.dim() != 2:
            raise ValueError("Input tensors must be 2-dimensional [batch_size, dimension]")
            
        if body1.size(1) != self.dimension or body2.size(1) != self.dimension:
            raise ValueError(f"Input tensors must have dimension {self.dimension}")
            
        combined = torch.cat([body1, body2], dim=-1)
        aspect = self.aspect_network(combined)
        strength = self.aspect_strength(aspect)
        
        return {
            "aspect": aspect,
            "strength": strength,
            "elemental_balance": self.elemental_balance(aspect)
        }
    
    def get_house_influence(self, 
                          body: torch.Tensor, 
                          house: int) -> torch.Tensor:
        """Calculate influence of a celestial body in a specific house
        
        Args:
            body: Celestial body tensor [batch_size, dimension]
            house: House number (0-11)
            
        Returns:
            Influence tensor [batch_size, dimension]
            
        Raises:
            ValueError: If input tensor has incorrect dimensions or house is invalid
        """
        if body.dim() != 2:
            raise ValueError("Input tensor must be 2-dimensional [batch_size, dimension]")
            
        if body.size(1) != self.dimension:
            raise ValueError(f"Input tensor must have dimension {self.dimension}")
            
        if not 0 <= house < 12:
            raise ValueError("House must be between 0 and 11")
            
        house_embedding = self.house_embeddings[house]
        return self.planetary_influence(body * house_embedding)
    
    def calculate_zodiac_position(self, 
                                body: torch.Tensor, 
                                sign: int) -> Dict[str, torch.Tensor]:
        """Calculate position and influence in a zodiac sign
        
        Args:
            body: Celestial body tensor [batch_size, dimension]
            sign: Zodiac sign number (0-11)
            
        Returns:
            Dictionary containing position and elemental balance
            
        Raises:
            ValueError: If input tensor has incorrect dimensions or sign is invalid
        """
        if body.dim() != 2:
            raise ValueError("Input tensor must be 2-dimensional [batch_size, dimension]")
            
        if body.size(1) != self.dimension:
            raise ValueError(f"Input tensor must have dimension {self.dimension}")
            
        if not 0 <= sign < 12:
            raise ValueError("Sign must be between 0 and 11")
            
        sign_embedding = self.zodiac_embeddings[sign]
        influence = self.planetary_influence(body * sign_embedding)
        
        return {
            "position": influence,
            "elemental_balance": self.elemental_balance(influence)
        }

class AstronomicalNode(nn.Module):
    def __init__(self, dimension: int = 256):
        super().__init__()
        self.dimension = dimension
        
        # Physical property encoders
        self.mass_encoder = nn.Sequential(
            nn.Linear(1, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension)
        )
        
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, dimension),  # 3D velocity vector
            nn.GELU(),
            nn.Linear(dimension, dimension)
        )
        
        # Orbital mechanics calculator
        self.orbital_mechanics = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
            nn.Tanh()
        )
        
        # Spectral analysis
        self.spectral_analysis = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, 10)  # 10 spectral bands
        )
        
        # Gravitational influence
        self.gravitational_influence = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Softmax(dim=-1)
        )
        
        # Celestial body classifier
        self.body_classifier = nn.Sequential(
            nn.Linear(dimension, len(CelestialBodyType)),
            nn.Softmax(dim=-1)
        )

    def calculate_orbital_parameters(self, 
                                   body1: torch.Tensor, 
                                   body2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate orbital parameters between two celestial bodies
        
        Args:
            body1: First celestial body tensor [batch_size, dimension]
            body2: Second celestial body tensor [batch_size, dimension]
            
        Returns:
            Dictionary containing orbital parameters and gravitational influence
        """
        if body1.dim() != 2 or body2.dim() != 2:
            raise ValueError("Input tensors must be 2-dimensional [batch_size, dimension]")
            
        if body1.size(1) != self.dimension or body2.size(1) != self.dimension:
            raise ValueError(f"Input tensors must have dimension {self.dimension}")
            
        combined = torch.cat([body1, body2], dim=-1)
        orbital = self.orbital_mechanics(combined)
        
        return {
            "orbital_parameters": orbital,
            "gravitational_influence": self.gravitational_influence(orbital)
        }
    
    def analyze_spectral_data(self, body: torch.Tensor) -> torch.Tensor:
        """Analyze spectral data of a celestial body
        
        Args:
            body: Celestial body tensor [batch_size, dimension]
            
        Returns:
            Spectral analysis tensor [batch_size, 10]
        """
        if body.dim() != 2:
            raise ValueError("Input tensor must be 2-dimensional [batch_size, dimension]")
            
        if body.size(1) != self.dimension:
            raise ValueError(f"Input tensor must have dimension {self.dimension}")
            
        return self.spectral_analysis(body)
    
    def classify_celestial_body(self, body: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Classify the type of celestial body
        
        Args:
            body: Celestial body tensor [batch_size, dimension]
            
        Returns:
            Dictionary containing classification probabilities and predicted body type
        """
        if body.dim() != 2:
            raise ValueError("Input tensor must be 2-dimensional [batch_size, dimension]")
            
        if body.size(1) != self.dimension:
            raise ValueError(f"Input tensor must have dimension {self.dimension}")
            
        classification = self.body_classifier(body)
        
        return {
            "classification": classification,
            "body_type": CelestialBodyType(classification.argmax(dim=-1).item())
        }
    
    @staticmethod
    def calculate_gravitational_force(mass1: float, 
                                    mass2: float, 
                                    distance: float) -> float:
        """Calculate gravitational force between two bodies
        
        Args:
            mass1: Mass of first body in kg
            mass2: Mass of second body in kg
            distance: Distance between bodies in meters
            
        Returns:
            Gravitational force in Newtons
            
        Raises:
            ValueError: If any input is negative or distance is zero
        """
        if mass1 < 0 or mass2 < 0:
            raise ValueError("Masses cannot be negative")
        if distance <= 0:
            raise ValueError("Distance must be positive")
            
        G = 6.67430e-11  # Gravitational constant
        return G * mass1 * mass2 / (distance ** 2)

class CelestialHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.dimension = 512
        self.embedding = nn.Embedding(256, 64)  # Character embedding
        self.encoder = nn.Sequential(
            nn.Linear(64, self.dimension),
            nn.GELU(),
            nn.Linear(self.dimension, self.dimension)
        )
        self.astrological_processor = nn.Sequential(
            nn.Linear(self.dimension, self.dimension),
            nn.Tanh(),
            nn.Linear(self.dimension, self.dimension)
        )
        self.astronomical_processor = nn.Sequential(
            nn.Linear(self.dimension, self.dimension),
            nn.Tanh(),
            nn.Linear(self.dimension, self.dimension)
        )
        
    def forward(self, input_text: str) -> Dict[str, torch.Tensor]:
        """Process input text through both astrological and astronomical systems
        
        Args:
            input_text: Input text to process
            
        Returns:
            Dictionary containing astrological, astronomical and fused analyses
        """
        # Convert text to character indices
        char_indices = torch.tensor([ord(c) % 256 for c in input_text], dtype=torch.long)
        
        # Get character embeddings
        char_embeddings = self.embedding(char_indices)  # Shape: [seq_len, 64]
        
        # Average the embeddings to get a fixed-size representation
        text_embedding = char_embeddings.mean(dim=0)  # Shape: [64]
        
        # Encode to full dimension
        input_tensor = self.encoder(text_embedding).unsqueeze(0)  # Shape: [1, dimension]
            
        # Process through both systems
        astrological = self.astrological_processor(input_tensor)
        astronomical = self.astronomical_processor(input_tensor)
        
        return {
            "astrological_analysis": astrological,
            "astronomical_analysis": astronomical,
            "fused": (astrological + astronomical) / 2
        } 