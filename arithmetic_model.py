import torch
import torch.nn as nn
from base_model import BaseModel

class ArithmeticModel(BaseModel):
    def __init__(self, input_size: int = 16, hidden_size: int = 128):
        super().__init__(input_size, hidden_size)
        
        # Arithmetic-specific layers
        self.arithmetic_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)  # Single output for arithmetic value
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better arithmetic performance"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get shared features
        features = super().forward(x)
        
        # Arithmetic prediction
        arithmetic_out = self.arithmetic_head(features)
        
        return arithmetic_out

class ArithmeticUtils:
    @staticmethod
    def extract_arithmetic_features(text: str) -> torch.Tensor:
        """Extract arithmetic-specific features"""
        features = []
        
        # Count operators
        features.append(text.count('+'))
        features.append(text.count('-'))
        features.append(text.count('*'))
        features.append(text.count('/'))
        
        # Count numbers
        features.append(len([c for c in text if c in 'IVXLCDM']))
        
        # Position features
        if '=' in text:
            features.append(text.find('='))
        else:
            features.append(-1)
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        
        return features
    
    @staticmethod
    def calculate_expected_result(text: str) -> int:
        """Calculate expected result from arithmetic expression"""
        try:
            # Extract numbers and operator
            parts = text.split('=')
            if len(parts) != 2:
                return 0
            
            expr = parts[0].strip()
            numbers = []
            operator = None
            
            # Parse expression
            for op in ['+', '-', '*', '/']:
                if op in expr:
                    operator = op
                    numbers = [n.strip() for n in expr.split(op)]
                    break
            
            if not operator or len(numbers) != 2:
                return 0
            
            # Convert Roman to Arabic
            def from_roman(s: str) -> int:
                roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
                result = 0
                prev = 0
                for c in reversed(s):
                    curr = roman[c]
                    result += curr if curr >= prev else -curr
                    prev = curr
                return result
            
            num1 = from_roman(numbers[0])
            num2 = from_roman(numbers[1])
            
            # Calculate result
            if operator == '+':
                return num1 + num2
            elif operator == '-':
                return num1 - num2
            elif operator == '*':
                return num1 * num2
            elif operator == '/':
                return num1 // num2 if num2 != 0 else 0
            
            return 0
        except Exception:
            return 0 