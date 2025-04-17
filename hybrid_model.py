import torch
import torch.nn as nn
from typing import Tuple, Union, Optional
from base_model import BaseModel, FeatureExtractor, ModelUtils
from arithmetic_model import ArithmeticModel, ArithmeticUtils
from structure_model import StructureModel, StructureUtils
import torch.nn.functional as F
import re
from knowledge_source import KnowledgeSource, KnowledgeEntry
from calculus_engine import CalculusEngine, CalculusResult
from typing import Dict, List
import sympy as sp
import numpy as np
import logging

class HybridModel(BaseModel):
    def __init__(self, input_size: int = 272, hidden_size: int = 128):
        # Initialize BaseModel with combined input size
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set feature dimensions
        self.feature_dim = 256  # Base feature dimension
        self.hybrid_feature_dim = 128  # Hybrid-specific feature dimension
        self.domain_feature_dim = 16  # Domain-specific feature dimension
        
        # Additional layers for hybrid-specific features
        self.hybrid_layers = nn.Sequential(
            nn.Linear(16, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        
        # Domain-specific layers
        self.domain_layers = nn.Sequential(
            nn.Linear(16, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        
        # Enhanced output heads
        self.type_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 3)  # 3 types: information, equation, definition
        ).to(self.device)
        
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 3)  # 3 domains: ai, physics, mathematics
        ).to(self.device)
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)  # Single confidence score
        ).to(self.device)
        
        # Initialize knowledge components
        self._initialize_knowledge_components()
        
        # Move model to device
        self.to(self.device)
        
        # Compile the model
        self._compile_model()
        
    def _compile_model(self):
        """Compile the model using torch's compiler"""
        try:
            # Compile the forward method
            self.forward = torch.compile(self.forward, mode='reduce-overhead')
            
            # Compile feature extraction methods
            self.extract_features = torch.compile(self.extract_features, mode='reduce-overhead')
            self._extract_hybrid_features = torch.compile(self._extract_hybrid_features, mode='reduce-overhead')
            
            # Compile hybrid layers
            self.hybrid_layers = torch.compile(self.hybrid_layers, mode='reduce-overhead')
            
            logging.info("Model successfully compiled with torch compiler")
        except Exception as e:
            logging.warning(f"Failed to compile model: {e}")
            logging.warning("Falling back to uncompiled model")
            
    def extract_features(self, text: str) -> Optional[torch.Tensor]:
        try:
            # Get base features from FeatureExtractor
            base_features = FeatureExtractor.extract_features(text)
            if base_features is None:
                return None
            
            # Get hybrid-specific features
            hybrid_features = self._extract_hybrid_features(text)
            if hybrid_features is None:
                return None
            
            # Get domain-specific features
            domain_features = self._extract_domain_features(text)
            
            # Combine features
            combined_features = torch.cat([
                base_features,
                hybrid_features,
                domain_features
            ], dim=0)
            
            return combined_features
            
        except Exception as e:
            logging.error(f"Error in hybrid feature extraction: {e}")
            return None

    def _extract_hybrid_features(self, text: str) -> Optional[torch.Tensor]:
        try:
            features = torch.zeros(256, dtype=torch.float32)
            
            # Original hybrid feature extraction logic
            word_count = len(text.split())
            char_count = len(text)
            
            # Arithmetic operations
            arithmetic_ops = sum(1 for c in text if c in '+-*/=')
            
            # Roman numerals
            roman_numerals = ['I', 'V', 'X', 'L', 'C', 'D', 'M']
            roman_count = sum(text.count(numeral) for numeral in roman_numerals)
            
            # Latin sentence patterns
            latin_patterns = {
                'est': 0, 'sunt': 0, 'habet': 0,
                'in': 0, 'ad': 0, 'cum': 0, 'ex': 0
            }
            for word in text.lower().split():
                if word in latin_patterns:
                    latin_patterns[word] += 1
            
            # Question words
            question_words = ['quis', 'quid', 'cur', 'quomodo', 'ubi', 'quando']
            question_count = sum(text.lower().count(word) for word in question_words)
            
            # Pack features
            feature_values = [
                word_count, char_count, arithmetic_ops, roman_count,
                question_count, *latin_patterns.values()
            ]
            
            for i, value in enumerate(feature_values):
                features[i] = float(value)
            
            # Normalize features
            features = F.normalize(features, p=2, dim=0)
            return features
            
        except Exception as e:
            logging.error(f"Error in hybrid-specific feature extraction: {e}")
            return None

    def _extract_domain_features(self, text: str) -> torch.Tensor:
        """Extract domain-specific features for better classification"""
        try:
            features = torch.zeros(16, dtype=torch.float32)
            
            # AI domain features
            ai_keywords = [
                'neural', 'network', 'learning', 'model', 'algorithm',
                'training', 'data', 'intelligence', 'machine', 'deep'
            ]
            ai_count = sum(text.lower().count(word) for word in ai_keywords)
            features[0] = ai_count
            
            # Physics domain features
            physics_keywords = [
                'energy', 'force', 'mass', 'velocity', 'acceleration',
                'quantum', 'particle', 'wave', 'field', 'motion'
            ]
            physics_count = sum(text.lower().count(word) for word in physics_keywords)
            features[1] = physics_count
            
            # Mathematics domain features
            math_keywords = [
                'function', 'equation', 'derivative', 'integral', 'matrix',
                'vector', 'algebra', 'calculus', 'theorem', 'proof'
            ]
            math_count = sum(text.lower().count(word) for word in math_keywords)
            features[2] = math_count
            
            # Equation patterns
            equation_patterns = [
                r'[A-Za-z]\s*=\s*[^=]+',  # Basic equations
                r'[A-Za-z]\s*[+\-*/]\s*[A-Za-z]',  # Algebraic expressions
                r'[A-Za-z]\s*\^\s*\d+',  # Exponents
                r'[A-Za-z]\s*[+\-*/]\s*\d+',  # Numeric operations
                r'[A-Za-z]\s*[+\-*/]\s*[A-Za-z]\s*[+\-*/]\s*[A-Za-z]'  # Complex expressions
            ]
            for i, pattern in enumerate(equation_patterns):
                features[3 + i] = len(re.findall(pattern, text))
            
            # Definition patterns
            definition_patterns = [
                r'is defined as',
                r'means that',
                r'refers to',
                r'is a',
                r'is the'
            ]
            for i, pattern in enumerate(definition_patterns):
                features[8 + i] = len(re.findall(pattern, text.lower()))
            
            # Information patterns
            info_patterns = [
                r'can be used to',
                r'is used for',
                r'is important because',
                r'is a type of',
                r'is based on'
            ]
            for i, pattern in enumerate(info_patterns):
                features[13 + i] = len(re.findall(pattern, text.lower()))
            
            # Normalize features
            features = F.normalize(features, p=2, dim=0)
            return features
            
        except Exception as e:
            logging.error(f"Error extracting domain features: {e}")
            return torch.zeros(16, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Move input to device
        x = x.to(self.device)
        
        # Get base features through parent's forward
        base_features = super().forward(x)
        
        # Process hybrid features through compiled layers
        hybrid_features = self.hybrid_layers(x)
        
        # Process domain features
        domain_features = self.domain_layers(x)
        
        # Combine features
        combined = torch.cat([
            base_features,
            hybrid_features,
            domain_features
        ], dim=1)
        
        # Project to output spaces with enhanced heads
        type_logits = self.type_head(combined)  # [batch_size, num_types]
        domain_logits = self.domain_head(combined)  # [batch_size, num_domains]
        confidence = torch.sigmoid(self.confidence_head(combined))  # [batch_size, 1]
        
        # Apply temperature scaling for better probability calibration
        temperature = 0.5  # Lower temperature for sharper probabilities
        type_logits = type_logits / temperature
        domain_logits = domain_logits / temperature
        
        return {
            'type': type_logits,
            'domain': domain_logits,
            'confidence': confidence
        }
    
    def roman_to_int(self, roman_str):
        """Convert Roman numeral to integer."""
        roman_values = {
            'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
            'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10
        }
        roman_str = roman_str.strip().upper()
        if roman_str in roman_values:
            return roman_values[roman_str]
        return 1  # Default value for invalid input
    
    def predict(self, text):
        """Make prediction with improved handling"""
        try:
            # Extract features
            features = self.extract_features(text)
            if features is None:
                return {
                    'arithmetic': torch.zeros(1, device=self.device),
                    'structure': torch.zeros(1, device=self.device),
                    'pattern': torch.zeros(1, device=self.device)
                }
            
            # Add batch dimension
            features = features.unsqueeze(0)
            
            # Get model outputs
            outputs = self.forward(features)
            
            # Split outputs into components
            arithmetic_out = outputs[:, :1]  # First component
            structure_out = outputs[:, 1:2]  # Second component
            pattern_out = outputs[:, 2:3]    # Third component
            
            return {
                'arithmetic': arithmetic_out,
                'structure': structure_out,
                'pattern': pattern_out
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'arithmetic': torch.zeros(1, device=self.device),
                'structure': torch.zeros(1, device=self.device),
                'pattern': torch.zeros(1, device=self.device)
            }

    def extract_algebraic_components(self, expression):
        """Extract components from algebraic expressions"""
        try:
            # Remove spaces and convert to lowercase
            expr = expression.replace(" ", "").lower()
            
            components = {
                'coefficients': [],
                'variables': [],
                'constants': [],
                'operators': []
            }
            
            current_num = ''
            
            for char in expr:
                if char.isdigit():
                    current_num += char
                elif char.isalpha():
                    if current_num:
                        components['coefficients'].append(int(current_num))
                        current_num = ''
                    components['variables'].append(char)
                elif char in ['+', '-', '*', '/', '=']:
                    if current_num:
                        components['constants'].append(int(current_num))
                        current_num = ''
                    components['operators'].append(char)
            
            # Handle any remaining number
            if current_num:
                components['constants'].append(int(current_num))
            
            return components
        except Exception as e:
            print(f"Error extracting algebraic components: {e}")
            return None

    def extract_data_features(self, data_text):
        """Extract features from data-related text"""
        try:
            features = {
                'numerical_values': [],
                'trends': [],
                'relationships': [],
                'categories': []
            }
            
            # Extract numerical values
            words = data_text.split()
            for word in words:
                try:
                    num = float(word)
                    features['numerical_values'].append(num)
                except ValueError:
                    continue
            
            # Identify trends and relationships
            trend_keywords = ['increase', 'decrease', 'growth', 'decline']
            relation_keywords = ['correlation', 'relationship', 'dependency']
            
            for word in words:
                if word.lower() in trend_keywords:
                    features['trends'].append(word.lower())
                elif word.lower() in relation_keywords:
                    features['relationships'].append(word.lower())
            
            return features
        except Exception as e:
            print(f"Error extracting data features: {e}")
            return None

    def extract_semantic_features(self, text):
        """Extract semantic features from text"""
        try:
            features = {
                'tense': 0,  # Present, Imperfect, Perfect, Pluperfect, Future
                'mood': 0,   # Indicative, Subjunctive, Imperative
                'voice': 0,  # Active, Passive
                'number': 0, # Singular, Plural
                'person': 0, # First, Second, Third
                'case': 0,   # Nominative, Genitive, Dative, Accusative, Ablative
                'gender': 0, # Masculine, Feminine, Neuter
                'degree': 0  # Positive, Comparative, Superlative
            }
            
            # Tense indicators
            tense_indicators = {
                'present': ['o', 's', 't', 'mus', 'tis', 'nt'],
                'imperfect': ['bam', 'bas', 'bat', 'bamus', 'batis', 'bant'],
                'perfect': ['i', 'isti', 'it', 'imus', 'istis', 'erunt'],
                'pluperfect': ['eram', 'eras', 'erat', 'eramus', 'eratis', 'erant'],
                'future': ['bo', 'bis', 'bit', 'bimus', 'bitis', 'bunt']
            }
            
            # Mood indicators
            mood_indicators = {
                'indicative': ['o', 's', 't', 'mus', 'tis', 'nt'],
                'subjunctive': ['em', 'es', 'et', 'emus', 'etis', 'ent'],
                'imperative': ['', 'te', 'to', 'tote', 'nto']
            }
            
            # Case endings
            case_endings = {
                'nominative': ['us', 'a', 'um', 'i', 'ae', 'a'],
                'genitive': ['i', 'ae', 'i', 'orum', 'arum', 'orum'],
                'dative': ['o', 'ae', 'o', 'is', 'is', 'is'],
                'accusative': ['um', 'am', 'um', 'os', 'as', 'a'],
                'ablative': ['o', 'a', 'o', 'is', 'is', 'is']
            }
            
            # Analyze each word
            words = text.lower().split()
            for word in words:
                # Check tense
                for tense, endings in tense_indicators.items():
                    if any(word.endswith(ending) for ending in endings):
                        features['tense'] = list(tense_indicators.keys()).index(tense)
                        break
                
                # Check mood
                for mood, endings in mood_indicators.items():
                    if any(word.endswith(ending) for ending in endings):
                        features['mood'] = list(mood_indicators.keys()).index(mood)
                        break
                
                # Check case
                for case, endings in case_endings.items():
                    if any(word.endswith(ending) for ending in endings):
                        features['case'] = list(case_endings.keys()).index(case)
                        break
                
                # Check voice (active/passive)
                if word.endswith(('r', 'ris', 'tur', 'mur', 'mini', 'ntur')):
                    features['voice'] = 1  # Passive
                else:
                    features['voice'] = 0  # Active
                
                # Check number
                if word.endswith(('i', 'ae', 'a', 'os', 'as', 'is')):
                    features['number'] = 1  # Plural
                else:
                    features['number'] = 0  # Singular
                
                # Check person
                if word.endswith(('o', 'm', 'mus')):
                    features['person'] = 0  # First
                elif word.endswith(('s', 'tis')):
                    features['person'] = 1  # Second
                else:
                    features['person'] = 2  # Third
                
                # Check gender
                if word.endswith(('us', 'i', 'os')):
                    features['gender'] = 0  # Masculine
                elif word.endswith(('a', 'ae', 'as')):
                    features['gender'] = 1  # Feminine
                else:
                    features['gender'] = 2  # Neuter
                
                # Check degree
                if word.endswith(('ior', 'ius')):
                    features['degree'] = 1  # Comparative
                elif word.endswith(('issimus', 'issima', 'issimum')):
                    features['degree'] = 2  # Superlative
                else:
                    features['degree'] = 0  # Positive
            
            return torch.tensor(list(features.values()), dtype=torch.float32)
            
        except Exception as e:
            print(f"Error extracting semantic features: {e}")
            return torch.zeros(8, dtype=torch.float32)

    def extract_context_features(self, text):
        """Extract context features from text"""
        try:
            features = {
                'main_clause': 0,
                'subordinate_clause': 0,
                'question': 0,
                'negation': 0
            }
            
            # Check for question
            if text.strip().endswith('?'):
                features['question'] = 1
            
            # Check for negation
            if 'non' in text.lower() or 'ne' in text.lower():
                features['negation'] = 1
            
            # Check for subordinate clauses
            subordinate_indicators = ['quod', 'quia', 'ut', 'cum', 'si', 'dum']
            if any(indicator in text.lower() for indicator in subordinate_indicators):
                features['subordinate_clause'] = 1
            else:
                features['main_clause'] = 1
            
            return torch.tensor(list(features.values()), dtype=torch.float32)
            
        except Exception as e:
            print(f"Error extracting context features: {e}")
            return torch.zeros(4, dtype=torch.float32)

    def extract_math_features(self, text):
        """Extract mathematical features from text"""
        try:
            # Initialize features with zeros
            features = torch.zeros(16, dtype=torch.float32)
            
            # Check for mathematical content
            math_patterns = [
                r'\d+',  # Numbers
                r'[+\-*/=]',  # Basic operators
                r'[a-zA-Z]+\s*=',  # Variables
                r'sin|cos|tan|log|exp',  # Functions
                r'[a-zA-Z]+\^[0-9]+',  # Exponents
                r'[a-zA-Z]+\s*[+\-*/]\s*[a-zA-Z0-9]+'  # Algebraic expressions
            ]
            
            # Count occurrences of each pattern
            for i, pattern in enumerate(math_patterns):
                matches = re.findall(pattern, text)
                features[i] = len(matches)
            
            # Normalize features
            features = features / (features.sum() + 1e-6)
            return features
            
        except Exception as e:
            print(f"Error extracting math features: {str(e)}")
            return torch.zeros(16, dtype=torch.float32)
    
    def extract_science_features(self, text):
        """Extract scientific features from text"""
        try:
            # Initialize features with zeros
            features = torch.zeros(16, dtype=torch.float32)
            
            # Check for scientific content
            science_patterns = [
                r'[A-Z][a-z]+\s*[0-9]*',  # Chemical elements
                r'[0-9]+\s*[A-Za-z]+',  # Units
                r'[A-Z][a-z]+\s*[A-Z][a-z]+',  # Scientific terms
                r'[0-9]+\.[0-9]+',  # Decimal numbers
                r'[0-9]+e[+\-]?[0-9]+',  # Scientific notation
                r'[A-Z][a-z]+\s*[0-9]+\s*[A-Z][a-z]+'  # Chemical compounds
            ]
            
            # Count occurrences of each pattern
            for i, pattern in enumerate(science_patterns):
                matches = re.findall(pattern, text)
                features[i] = len(matches)
            
            # Normalize features
            features = features / (features.sum() + 1e-6)
            return features
            
        except Exception as e:
            print(f"Error extracting science features: {str(e)}")
            return torch.zeros(16, dtype=torch.float32)
    
    def extract_doc_features(self, text):
        """Extract document structure features"""
        try:
            # Initialize features with zeros
            features = torch.zeros(16, dtype=torch.float32)
            
            # Check for document structure
            doc_patterns = [
                r'^#+\s',  # Headers
                r'\n\n',  # Paragraph breaks
                r'\[.*?\]\(.*?\)',  # Links
                r'!\[.*?\]\(.*?\)',  # Images
                r'`[^`]+`',  # Code blocks
                r'\*\*[^*]+\*\*',  # Bold text
                r'\*[^*]+\*',  # Italic text
                r'^\s*[-*+]\s'  # List items
            ]
            
            # Count occurrences of each pattern
            for i, pattern in enumerate(doc_patterns):
                matches = re.findall(pattern, text, re.MULTILINE)
                features[i] = len(matches)
            
            # Normalize features
            features = features / (features.sum() + 1e-6)
            return features
            
        except Exception as e:
            print(f"Error extracting document features: {str(e)}")
            return torch.zeros(16, dtype=torch.float32)

    def _initialize_calculus_knowledge(self):
        """Initialize the knowledge base with calculus concepts"""
        calculus_concepts = [
            KnowledgeEntry(
                source='calculus/derivatives',
                content='Derivatives measure the rate of change of a function. The derivative of a function f(x) represents the slope of the tangent line at any point x.',
                metadata={
                    'type': 'concept',
                    'category': 'calculus',
                    'subcategory': 'derivatives',
                    'keywords': ['derivative', 'rate of change', 'slope', 'tangent']
                },
                confidence=1.0
            ),
            KnowledgeEntry(
                source='calculus/integrals',
                content='Integration is the reverse process of differentiation. It can be used to find areas, volumes, and solve differential equations.',
                metadata={
                    'type': 'concept',
                    'category': 'calculus',
                    'subcategory': 'integration',
                    'keywords': ['integral', 'antiderivative', 'area', 'volume']
                },
                confidence=1.0
            ),
            KnowledgeEntry(
                source='calculus/limits',
                content='Limits describe the value that a function approaches as its input approaches a particular value. They are fundamental to understanding continuity and derivatives.',
                metadata={
                    'type': 'concept',
                    'category': 'calculus',
                    'subcategory': 'limits',
                    'keywords': ['limit', 'continuity', 'approach', 'infinity']
                },
                confidence=1.0
            ),
            KnowledgeEntry(
                source='calculus/multivariable',
                content='Multivariable calculus extends calculus to functions of multiple variables. It includes concepts like partial derivatives, gradients, and multiple integrals.',
                metadata={
                    'type': 'concept',
                    'category': 'calculus',
                    'subcategory': 'multivariable',
                    'keywords': ['multivariable', 'partial', 'gradient', 'multiple']
                },
                confidence=1.0
            ),
            KnowledgeEntry(
                source='calculus/vector',
                content='Vector calculus deals with vector fields, including concepts like divergence, curl, and line integrals. It is essential for understanding electromagnetic theory and fluid dynamics.',
                metadata={
                    'type': 'concept',
                    'category': 'calculus',
                    'subcategory': 'vector',
                    'keywords': ['vector', 'field', 'divergence', 'curl', 'line integral']
                },
                confidence=1.0
            )
        ]
        
        for entry in calculus_concepts:
            self.knowledge_source.update_knowledge_base(entry)

    def process_query(self, query: str) -> Dict:
        """Process a query that might involve calculus or general knowledge"""
        # Check if query contains mathematical expressions
        math_patterns = [
            r'derivative of (.*)',
            r'integrate (.*)',
            r'solve (.*) equation',
            r'limit of (.*) as .* approaches .*',
            r'analyze function (.*)'
        ]
        
        for pattern in math_patterns:
            match = re.search(pattern, query.lower())
            if match:
                return self._handle_math_query(query)
        
        # If no math patterns found, search knowledge base
        return self._handle_knowledge_query(query)
    
    def _handle_math_query(self, query: str) -> Dict:
        """Handle mathematical queries using calculus engine"""
        try:
            # Extract mathematical expression and operation type
            if 'gradient' in query.lower():
                expr = re.search(r'gradient of (.*)', query.lower()).group(1)
                result = self.compute_gradient(expr)
                operation = 'gradient'
            elif 'divergence' in query.lower():
                match = re.search(r'divergence of \[(.*)\]', query.lower())
                vector_field = [f.strip() for f in match.group(1).split(',')]
                result = self.compute_divergence(vector_field)
                operation = 'divergence'
            elif 'curl' in query.lower():
                match = re.search(r'curl of \[(.*)\]', query.lower())
                vector_field = [f.strip() for f in match.group(1).split(',')]
                result = self.compute_curl(vector_field)
                operation = 'curl'
            elif 'line integral' in query.lower():
                # Example format: "line integral of [P,Q,R] along [x(t),y(t),z(t)] from t=a to b"
                match = re.search(r'line integral of \[(.*)\] along \[(.*)\] from t=(.*) to (.*)', query.lower())
                vector_field = [f.strip() for f in match.group(1).split(',')]
                path = [p.strip() for p in match.group(2).split(',')]
                t_range = (float(match.group(3)), float(match.group(4)))
                result = self.compute_line_integral(vector_field, path, t_range)
                operation = 'line_integral'
            elif 'derivative' in query.lower():
                expr = re.search(r'derivative of (.*)', query.lower()).group(1)
                result = self.calculus_engine.differentiate(expr)
                operation = 'derivative'
            elif 'integrate' in query.lower():
                expr = re.search(r'integrate (.*)', query.lower()).group(1)
                result = self.calculus_engine.integrate(expr)
                operation = 'integration'
            elif 'limit' in query.lower():
                match = re.search(r'limit of (.*) as (.*) approaches (.*)', query.lower())
                expr, var, point = match.groups()
                result = self.calculus_engine.find_limits(expr, var, point)
                operation = 'limit'
            elif 'analyze' in query.lower():
                expr = re.search(r'analyze function (.*)', query.lower()).group(1)
                result = self.calculus_engine.analyze_function(expr)
                operation = 'analysis'
            else:
                raise ValueError("Unsupported mathematical operation")
            
            # Get relevant knowledge about the operation
            knowledge = self.knowledge_source.search_knowledge_base(f"calculus {operation}")
            
            return {
                'type': 'math_result',
                'operation': operation,
                'result': result,
                'related_knowledge': [k.content for k in knowledge],
                'confidence': 1.0
            }
        except Exception as e:
            return {
                'type': 'error',
                'message': str(e),
                'confidence': 0.0
            }
    
    def _handle_knowledge_query(self, query: str) -> Dict:
        """Handle general knowledge queries"""
        results = self.knowledge_source.search_knowledge_base(query)
        
        if not results:
            return {
                'type': 'no_results',
                'message': 'No relevant information found',
                'confidence': 0.0
            }
        
        return {
            'type': 'knowledge_result',
            'results': [{
                'content': r.content,
                'source': r.source,
                'confidence': r.confidence,
                'metadata': r.metadata
            } for r in results],
            'confidence': max(r.confidence for r in results)
        }
    
    def learn_from_result(self, query: str, result: Dict):
        """Learn from successful queries and results"""
        if result['type'] == 'math_result':
            # Create new knowledge entry from mathematical result
            entry = KnowledgeEntry(
                source=f"learned/math/{result['operation']}",
                content=f"Query: {query}\nResult: {str(result['result'])}",
                metadata={
                    'type': 'learned_result',
                    'category': 'calculus',
                    'subcategory': result['operation'],
                    'query': query
                },
                confidence=result['confidence']
            )
            self.knowledge_source.update_knowledge_base(entry)

    def compute_gradient(self, expr_str: str, variables: List[str] = None) -> Dict:
        """Compute the gradient of a multivariable function"""
        try:
            if variables is None:
                variables = ['x', 'y', 'z']
            
            expr = self.calculus_engine.parse_expression(expr_str)
            gradient = []
            steps = []
            
            for var in variables:
                partial = sp.diff(expr, sp.Symbol(var))
                gradient.append(str(partial))
                steps.append(f"∂/∂{var} = {partial}")
            
            return {
                'gradient': gradient,
                'steps': steps,
                'variables': variables
            }
        except Exception as e:
            raise ValueError(f"Error computing gradient: {str(e)}")
    
    def compute_divergence(self, vector_field: List[str], variables: List[str] = None) -> str:
        """Compute the divergence of a vector field"""
        try:
            if variables is None:
                variables = ['x', 'y', 'z']
            if len(vector_field) != len(variables):
                raise ValueError("Vector field components must match number of variables")
            
            div = 0
            for f, var in zip(vector_field, variables):
                component = self.calculus_engine.parse_expression(f)
                div += sp.diff(component, sp.Symbol(var))
            
            return str(div)
        except Exception as e:
            raise ValueError(f"Error computing divergence: {str(e)}")
    
    def compute_curl(self, vector_field: List[str]) -> List[str]:
        """Compute the curl of a vector field"""
        try:
            if len(vector_field) != 3:
                raise ValueError("Vector field must have 3 components for curl")
            
            F = [self.calculus_engine.parse_expression(f) for f in vector_field]
            x, y, z = sp.symbols('x y z')
            
            curl = [
                str(sp.diff(F[2], y) - sp.diff(F[1], z)),
                str(sp.diff(F[0], z) - sp.diff(F[2], x)),
                str(sp.diff(F[1], x) - sp.diff(F[0], y))
            ]
            
            return curl
        except Exception as e:
            raise ValueError(f"Error computing curl: {str(e)}")
    
    def compute_line_integral(self, vector_field: List[str], path: List[str], t_range: tuple) -> str:
        """Compute a line integral along a parametric path"""
        try:
            # Parse vector field components
            F = [self.calculus_engine.parse_expression(f) for f in vector_field]
            
            # Parse path components
            r = [self.calculus_engine.parse_expression(p) for p in path]
            
            t = sp.Symbol('t')
            x, y, z = sp.symbols('x y z')
            
            # Compute dr/dt components
            dr_dt = [sp.diff(ri, t) for ri in r]
            
            # Substitute path into vector field
            subs_dict = {x: r[0], y: r[1], z: r[2] if len(r) > 2 else 0}
            F_along_path = [fi.subs(subs_dict) for fi in F]
            
            # Compute integrand
            integrand = sum(F_along_path[i] * dr_dt[i] for i in range(len(F)))
            
            # Compute definite integral
            t_min, t_max = t_range
            result = sp.integrate(integrand, (t, t_min, t_max))
            
            return str(result)
        except Exception as e:
            raise ValueError(f"Error computing line integral: {str(e)}")
    
    def compute_surface_integral(self, scalar_field: str, surface: List[str], u_range: tuple, v_range: tuple) -> str:
        """Compute a surface integral"""
        try:
            # Parse scalar field
            f = self.calculus_engine.parse_expression(scalar_field)
            
            # Parse surface parametrization
            r = [self.calculus_engine.parse_expression(p) for p in surface]
            
            u, v = sp.symbols('u v')
            x, y, z = sp.symbols('x y z')
            
            # Compute partial derivatives for surface normal
            ru = [sp.diff(ri, u) for ri in r]
            rv = [sp.diff(ri, v) for ri in r]
            
            # Compute surface normal magnitude
            normal_mag = sp.sqrt(
                (ru[1]*rv[2] - ru[2]*rv[1])**2 +
                (ru[2]*rv[0] - ru[0]*rv[2])**2 +
                (ru[0]*rv[1] - ru[1]*rv[0])**2
            )
            
            # Substitute surface into scalar field
            subs_dict = {x: r[0], y: r[1], z: r[2]}
            f_on_surface = f.subs(subs_dict)
            
            # Compute double integral
            u_min, u_max = u_range
            v_min, v_max = v_range
            result = sp.integrate(f_on_surface * normal_mag, (u, u_min, u_max), (v, v_min, v_max))
            
            return str(result)
        except Exception as e:
            raise ValueError(f"Error computing surface integral: {str(e)}")

    def _initialize_knowledge_components(self):
        """Initialize knowledge components for the hybrid model"""
        try:
            # Initialize knowledge source
            self.knowledge_source = KnowledgeSource()
            
            # Initialize calculus knowledge
            self._initialize_calculus_knowledge()
            
            # Initialize calculus engine
            self.calculus_engine = CalculusEngine()
            
            # Initialize feature extractors
            self.feature_extractors = {
                'algebraic': self.extract_algebraic_features,
                'data': self.extract_data_features,
                'science': self.extract_science_features,
                'doc': self.extract_doc_features,
                'semantic': self.extract_semantic_features
            }
            
            # Initialize state tracking
            self.state = {
                "quantum": {},
                "mathematical": {
                    "equations": {},
                    "patterns": {},
                    "complexity": 0.0,
                    "topological_structures": {}
                },
                "philosophical": {
                    "ontological": {},
                    "epistemic": {},
                    "ethical": {},
                    "metaphysical": {}
                },
                "linguistic": {
                    "semantic": {},
                    "syntactic": {},
                    "narrative": {},
                    "symbolic": {}
                },
                "cosmic": {
                    "alignment": {},
                    "resonance": {},
                    "synchronization": {},
                    "center_state": {
                        "singularity_black_hole_superposition": 0.0,
                        "multiversal_grid_connection": 0.0,
                        "quantum_entanglement": 0.0,
                        "temporal_synchronization": 0.0
                    }
                }
            }
            
            # Initialize metrics
            self.metrics = {
                "architecture": {"layer_performance": {}, "bottlenecks": [], "optimizations": []},
                "learning": {"dynamics": {}, "parameters": {}, "effectiveness": 0.0},
                "attention": {"patterns": {}, "weights": {}, "scope": {}},
                "integration": {"strategy": {}, "knowledge": {}, "context": {}},
                "emotional": {"resonance": 0.0, "harmony": 0.0, "depth": 0.0},
                "cosmic": {
                    "alignment": 0.0,
                    "synchronization": 0.0,
                    "resonance": 0.0,
                    "center_state_coherence": 0.0,
                    "multiversal_connection_strength": 0.0
                }
            }
            
        except Exception as e:
            print(f"Error initializing knowledge components: {e}")
            raise

    def extract_algebraic_features(self, input_data):
        """Extract algebraic features from input data."""
        try:
            # Convert input to tensor if not already
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32)
            
            # Apply algebraic transformations
            features = {
                'magnitude': torch.norm(input_data, dim=-1),
                'mean': torch.mean(input_data, dim=-1),
                'variance': torch.var(input_data, dim=-1),
                'max_val': torch.max(input_data, dim=-1)[0],
                'min_val': torch.min(input_data, dim=-1)[0]
            }
            return features
        except Exception as e:
            print(f"Error in algebraic feature extraction: {str(e)}")
            return None

    def extract_data_features(self, input_data):
        """Extract statistical and data-oriented features."""
        try:
            # Convert input to tensor if not already
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32)
            
            # Calculate statistical features
            features = {
                'skewness': torch.mean(((input_data - torch.mean(input_data)) / torch.std(input_data)) ** 3),
                'kurtosis': torch.mean(((input_data - torch.mean(input_data)) / torch.std(input_data)) ** 4),
                'entropy': -torch.sum(input_data * torch.log(input_data + 1e-10)),
                'gradient_mean': torch.mean(torch.gradient(input_data)[0]),
                'gradient_var': torch.var(torch.gradient(input_data)[0])
            }
            return features
        except Exception as e:
            print(f"Error in data feature extraction: {str(e)}")
            return None

    def extract_science_features(self, input_data):
        """Extract scientific and physical features."""
        try:
            # Convert input to tensor if not already
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32)
            
            # Calculate physics-inspired features
            features = {
                'energy': torch.sum(input_data ** 2),
                'momentum': torch.sum(torch.gradient(input_data)[0]),
                'acceleration': torch.sum(torch.gradient(torch.gradient(input_data)[0])[0]),
                'phase': torch.angle(torch.complex(input_data, torch.zeros_like(input_data))),
                'frequency': torch.fft.fft(input_data).abs()
            }
            return features
        except Exception as e:
            print(f"Error in science feature extraction: {str(e)}")
            return None

    def extract_doc_features(self, input_data):
        """Extract document and text-based features."""
        try:
            # Assuming input_data is text or can be converted to text
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.tolist()
            if isinstance(input_data, (list, np.ndarray)):
                input_data = str(input_data)
            
            # Calculate text-based features
            features = {
                'length': len(input_data),
                'word_count': len(input_data.split()),
                'unique_chars': len(set(input_data)),
                'alpha_ratio': sum(c.isalpha() for c in input_data) / len(input_data) if input_data else 0,
                'numeric_ratio': sum(c.isnumeric() for c in input_data) / len(input_data) if input_data else 0
            }
            return features
        except Exception as e:
            print(f"Error in document feature extraction: {str(e)}")
            return None

    def extract_semantic_features(self, input_data):
        """Extract semantic and contextual features."""
        try:
            # Convert input to tensor if not already
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32)
            
            # Calculate semantic features using embeddings
            features = {
                'semantic_norm': torch.norm(input_data, p=2, dim=-1),
                'semantic_direction': input_data / (torch.norm(input_data, p=2, dim=-1, keepdim=True) + 1e-8),
                'semantic_similarity': torch.cosine_similarity(input_data, torch.mean(input_data, dim=0, keepdim=True), dim=-1),
                'semantic_variance': torch.var(input_data, dim=-1),
                'semantic_range': torch.max(input_data, dim=-1)[0] - torch.min(input_data, dim=-1)[0]
            }
            return features
        except Exception as e:
            print(f"Error in semantic feature extraction: {str(e)}")
            return None

    def extract_base_features(self, input_data: Union[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Extract base features from input data.
        
        Args:
            input_data: Either a string or tensor input
            
        Returns:
            Optional[torch.Tensor]: Extracted features or None if extraction fails
        """
        try:
            # Handle string input
            if isinstance(input_data, str):
                # Convert string to initial feature tensor
                features = torch.zeros(self.feature_dim, dtype=torch.float32)
                
                # Extract basic text features
                words = input_data.split()
                features[0] = len(words)  # Word count
                features[1] = len(input_data)  # Character count
                features[2] = len(set(words))  # Unique word count
                features[3] = sum(c.isupper() for c in input_data)  # Uppercase count
                features[4] = sum(c.isdigit() for c in input_data)  # Digit count
                features[5] = sum(c.isspace() for c in input_data)  # Space count
                
                # Extract pattern-based features
                features[6] = sum(1 for c in input_data if c in '+-*/=')  # Math operators
                features[7] = input_data.count('(')  # Open parentheses
                features[8] = input_data.count(')')  # Close parentheses
                
                # Normalize features
                features = F.normalize(features, p=2, dim=0)
                return features
                
            # Handle tensor input
            elif isinstance(input_data, torch.Tensor):
                return input_data
                
            else:
                logging.error(f"Unsupported input type: {type(input_data)}")
                return None
                
        except Exception as e:
            logging.error(f"Error in extract_base_features: {e}")
            return None

    def calculate_loss(self, output: Dict, target: Dict) -> torch.Tensor:
        """Calculate loss between model output and target.
        
        Args:
            output: Dictionary containing model outputs
            target: Dictionary containing target values
            
        Returns:
            torch.Tensor: Combined loss value
        """
        try:
            # Initialize total loss
            total_loss = torch.tensor(0.0, device=self.device)
            
            # Calculate arithmetic loss if available
            if 'arithmetic' in output and 'arithmetic' in target:
                arithmetic_loss = F.mse_loss(
                    output['arithmetic'],
                    target['arithmetic']
                )
                total_loss += 0.4 * arithmetic_loss  # 40% weight
            
            # Calculate structure loss if available
            if 'structure' in output and 'structure' in target:
                structure_loss = F.cross_entropy(
                    output['structure'],
                    target['structure']
                )
                total_loss += 0.3 * structure_loss  # 30% weight
            
            # Calculate pattern loss if available
            if 'pattern' in output and 'pattern' in target:
                pattern_loss = F.cross_entropy(
                    output['pattern'],
                    target['pattern']
                )
                total_loss += 0.3 * pattern_loss  # 30% weight
            
            return total_loss
            
        except Exception as e:
            logging.error(f"Error calculating loss: {e}")
            # Return a default loss value that can be backpropagated
            return torch.tensor(1.0, requires_grad=True, device=self.device)

class HybridUtils:
    @staticmethod
    def combine_features(text: str) -> torch.Tensor:
        """Combine features from all components"""
        try:
            # Get base features
            base_features = FeatureExtractor.extract_features(text)
            if base_features is None:
                return None
            
            # Get arithmetic features
            arithmetic_features = ArithmeticUtils.extract_arithmetic_features(text)
            
            # Get structure features
            structure_features = StructureUtils.extract_structure_features(text)
            
            # Get algebraic features
            algebraic_features = HybridUtils.extract_algebraic_features(text)
            
            # Get data features
            data_features = HybridUtils.extract_data_features(text)
            
            # Ensure all feature tensors are 1D and have correct size
            feature_tensors = [
                base_features,
                arithmetic_features,
                structure_features,
                algebraic_features,
                data_features
            ]
            
            # Pad or truncate each feature set to size 16
            for i in range(len(feature_tensors)):
                if len(feature_tensors[i]) < 16:
                    feature_tensors[i] = F.pad(feature_tensors[i], (0, 16 - len(feature_tensors[i])))
                else:
                    feature_tensors[i] = feature_tensors[i][:16]
            
            # Combine features by weighted averaging
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal weights for now
            combined_features = sum(w * f for w, f in zip(weights, feature_tensors))
            
            return combined_features
            
        except Exception as e:
            print(f"Error combining features: {e}")
            return None
    
    @staticmethod
    def determine_pattern_type(text: str) -> int:
        """Determine the pattern type based on text characteristics"""
        # Check for various patterns
        has_arithmetic = any(op in text for op in ['+', '-', '*', '/'])
        has_sentence = any(word in text.lower() for word in ['est', 'sunt', 'habet', 'currit', 'volat', 'legit', 'claude'])
        has_algebraic = any(char.isalpha() and char.lower() in ['x', 'y', 'z'] for char in text)
        has_data = any(word in text.lower() for word in ['data', 'trend', 'correlation', 'mean', 'average', 'increase', 'decrease'])
        
        if has_algebraic:
            return 3  # Algebraic
        elif has_data:
            return 4  # Data
        elif has_arithmetic and has_sentence:
            return 2  # Mixed
        elif has_arithmetic:
            return 0  # Arithmetic
        else:
            return 1  # Sentence
    
    @staticmethod
    def extract_algebraic_features(text: str) -> torch.Tensor:
        """Extract features specific to algebraic expressions"""
        try:
            features = []
            
            # Count variables
            var_count = sum(1 for char in text if char.isalpha() and char.lower() in ['x', 'y', 'z'])
            features.append(var_count)
            
            # Count operators
            op_count = sum(1 for char in text if char in ['+', '-', '*', '/', '='])
            features.append(op_count)
            
            # Count numbers
            num_count = sum(1 for char in text if char.isdigit())
            features.append(num_count)
            
            # Check for specific patterns
            features.extend([
                1 if '=' in text else 0,  # Equation
                1 if '^' in text else 0,  # Power
                1 if any(op in text for op in ['sin', 'cos', 'tan']) else 0,  # Trigonometry
                1 if 'log' in text else 0,  # Logarithm
                1 if 'sqrt' in text else 0  # Square root
            ])
            
            # Pad to length 16
            while len(features) < 16:
                features.append(0)
            
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error extracting algebraic features: {e}")
            return torch.zeros(16, dtype=torch.float32)
    
    @staticmethod
    def extract_data_features(text: str) -> torch.Tensor:
        """Extract features specific to data analysis"""
        try:
            features = []
            text_lower = text.lower()
            
            # Count numerical values
            numbers = [float(word) for word in text.split() if word.replace('.', '').isdigit()]
            features.extend([
                len(numbers),  # Number count
                sum(numbers) / len(numbers) if numbers else 0,  # Mean
                max(numbers) if numbers else 0,  # Max
                min(numbers) if numbers else 0  # Min
            ])
            
            # Check for trend keywords
            trend_words = ['increase', 'decrease', 'growth', 'decline', 'rise', 'fall']
            features.append(sum(1 for word in trend_words if word in text_lower))
            
            # Check for statistical terms
            stat_terms = ['mean', 'average', 'median', 'mode', 'variance', 'deviation']
            features.append(sum(1 for term in stat_terms if term in text_lower))
            
            # Check for relationship terms
            relation_terms = ['correlation', 'relationship', 'dependency', 'causation']
            features.append(sum(1 for term in relation_terms if term in text_lower))
            
            # Pad to length 16
            while len(features) < 16:
                features.append(0)
            
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error extracting data features: {e}")
            return torch.zeros(16, dtype=torch.float32)

class HybridArithmeticNet(nn.Module):
    def __init__(self):
        super(HybridArithmeticNet, self).__init__()
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Arithmetic head
        self.arithmetic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # Structure head
        self.structure_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        # Pattern head
        self.pattern_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        
        # Shared features
        shared_features = self.shared_layers(x)
        
        # Task-specific outputs
        arithmetic_out = self.arithmetic_head(shared_features)
        structure_out = self.structure_head(shared_features)
        pattern_out = self.pattern_head(shared_features)
        
        return arithmetic_out, structure_out, pattern_out 

class HybridNode:
    """Main hybrid node class that integrates all components"""
    
    def __init__(self):
        # Initialize core components
        self.rsen = RSEN()
        self.isomorph = IsomorphNode()
        self.vortex = HybridVortexModule()
        
        # Initialize state tracking with enhanced cosmic state
        self.state = {
            "quantum": {
                "entanglement": 0.0,
                "superposition": 0.0,
                "coherence": 0.0,
                "multiversal_connection": 0.0
            },
            "mathematical": {
                "equations": {},
                "patterns": {},
                "complexity": 0.0,
                "topological_structures": {}
            },
            "philosophical": {
                "ontological": {},
                "epistemic": {},
                "ethical": {},
                "metaphysical": {}
            },
            "linguistic": {
                "semantic": {},
                "syntactic": {},
                "narrative": {},
                "symbolic": {}
            },
            "cosmic": {
                "alignment": {},
                "resonance": {},
                "synchronization": {},
                "center_state": {
                    "singularity_black_hole_superposition": 0.0,
                    "multiversal_grid_connection": 0.0,
                    "quantum_entanglement": 0.0,
                    "temporal_synchronization": 0.0
                }
            }
        }
        
        # Initialize performance metrics
        self.metrics = {
            "architecture": {"layer_performance": {}, "bottlenecks": [], "optimizations": []},
            "learning": {"dynamics": {}, "parameters": {}, "effectiveness": 0.0},
            "attention": {"patterns": {}, "weights": {}, "scope": {}},
            "integration": {"strategy": {}, "knowledge": {}, "context": {}},
            "emotional": {"resonance": 0.0, "harmony": 0.0, "depth": 0.0},
            "cosmic": {
                "alignment": 0.0,
                "synchronization": 0.0,
                "resonance": 0.0,
                "center_state_coherence": 0.0,
                "multiversal_connection_strength": 0.0
            }
        }
        
        logger.info("HybridNode initialized with enhanced cosmic state tracking")
        
    def _update_state(self, output: Dict):
        """Update node state based on output"""
        try:
            # Update quantum state
            self.state["quantum"].update(output.get("quantum", {}))
            
            # Update mathematical state
            self.state["mathematical"].update(output.get("mathematical", {}))
            
            # Update philosophical state
            self.state["philosophical"].update(output.get("philosophical", {}))
            
            # Update linguistic state
            self.state["linguistic"].update(output.get("linguistic", {}))
            
            # Update cosmic state with enhanced center state tracking
            cosmic_output = output.get("cosmic", {})
            if "center_state" in cosmic_output:
                self.state["cosmic"]["center_state"].update(cosmic_output["center_state"])
            self.state["cosmic"].update(cosmic_output)
            
        except Exception as e:
            logger.error(f"Error updating state: {e}")
            
    def _update_metrics(self):
        """Update performance metrics"""
        try:
            # Update architecture metrics
            self.metrics["architecture"] = self.rsen._analyze_architecture_performance()
            
            # Update learning metrics
            self.metrics["learning"] = self.rsen._analyze_learning_dynamics()
            
            # Update attention metrics
            self.metrics["attention"] = self.rsen._analyze_attention_performance()
            
            # Update integration metrics
            self.metrics["integration"] = self.rsen._analyze_knowledge_integration()
            
            # Update emotional metrics
            self.metrics["emotional"] = self.rsen._analyze_emotional_dynamics()
            
            # Update cosmic metrics with enhanced center state analysis
            cosmic_metrics = self.rsen._analyze_cosmic_alignment()
            if "center_state" in cosmic_metrics:
                self.metrics["cosmic"].update({
                    "center_state_coherence": cosmic_metrics["center_state"].get("coherence", 0.0),
                    "multiversal_connection_strength": cosmic_metrics["center_state"].get("connection_strength", 0.0)
                })
            self.metrics["cosmic"].update(cosmic_metrics)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            
    def calculate_performance(self) -> Dict:
        """Calculate overall performance"""
        try:
            # Calculate RSEN performance
            rsen_performance = {
                "resonance": self.rsen.calculate_resonance(),
                "complexity": self.rsen.calculate_complexity(),
                "coherence": self.rsen.calculate_coherence(),
                "emotional_depth": self.rsen.calculate_emotional_depth(),
                "emotional_harmony": self.rsen.calculate_emotional_harmony(),
                "recursion_depth": self.rsen.calculate_recursion_depth(),
                "pattern_complexity": self.rsen.calculate_pattern_complexity()
            }
            
            # Calculate IsomorphNode performance
            isomorph_performance = {
                "mapping_effectiveness": self.isomorph.calculate_mapping_effectiveness(),
                "integration_quality": self.isomorph.calculate_integration_quality(),
                "transformation_efficiency": self.isomorph.calculate_transformation_efficiency()
            }
            
            # Calculate Vortex performance
            vortex_performance = {
                "knowledge_quality": self.vortex.calculate_knowledge_quality(),
                "pattern_recognition": self.vortex.calculate_pattern_recognition(),
                "evolution_rate": self.vortex.calculate_evolution_rate()
            }
            
            # Calculate cosmic center state performance
            cosmic_performance = {
                "center_state_coherence": self._calculate_center_state_coherence(),
                "multiversal_connection_strength": self._calculate_multiversal_connection(),
                "superposition_stability": self._calculate_superposition_stability()
            }
            
            return {
                "rsen": rsen_performance,
                "isomorph": isomorph_performance,
                "vortex": vortex_performance,
                "cosmic": cosmic_performance,
                "overall": self._calculate_overall_performance(
                    rsen_performance,
                    isomorph_performance,
                    vortex_performance,
                    cosmic_performance
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return {}
            
    def _calculate_center_state_coherence(self) -> float:
        """Calculate coherence of the cosmic center state"""
        try:
            center_state = self.state["cosmic"]["center_state"]
            return (
                center_state["singularity_black_hole_superposition"] *
                center_state["multiversal_grid_connection"] *
                center_state["quantum_entanglement"] *
                center_state["temporal_synchronization"]
            )
        except Exception as e:
            logger.error(f"Error calculating center state coherence: {e}")
            return 0.0
            
    def _calculate_multiversal_connection(self) -> float:
        """Calculate strength of multiversal grid connection"""
        try:
            center_state = self.state["cosmic"]["center_state"]
            return (
                center_state["multiversal_grid_connection"] *
                (1 + center_state["quantum_entanglement"]) *
                (1 + center_state["temporal_synchronization"])
            ) / 3.0
        except Exception as e:
            logger.error(f"Error calculating multiversal connection: {e}")
            return 0.0
            
    def _calculate_superposition_stability(self) -> float:
        """Calculate stability of the singularity-black hole superposition"""
        try:
            center_state = self.state["cosmic"]["center_state"]
            return (
                center_state["singularity_black_hole_superposition"] *
                (1 + center_state["quantum_entanglement"]) *
                (1 + center_state["temporal_synchronization"])
            ) / 3.0
        except Exception as e:
            logger.error(f"Error calculating superposition stability: {e}")
            return 0.0
            
    def _calculate_overall_performance(
        self,
        rsen_performance: Dict,
        isomorph_performance: Dict,
        vortex_performance: Dict,
        cosmic_performance: Dict
    ) -> float:
        """Calculate overall performance score"""
        try:
            # Calculate weighted average of all performance metrics
            weights = {
                "rsen": 0.3,
                "isomorph": 0.2,
                "vortex": 0.2,
                "cosmic": 0.3
            }
            
            rsen_score = sum(rsen_performance.values()) / len(rsen_performance)
            isomorph_score = sum(isomorph_performance.values()) / len(isomorph_performance)
            vortex_score = sum(vortex_performance.values()) / len(vortex_performance)
            cosmic_score = sum(cosmic_performance.values()) / len(cosmic_performance)
            
            return (
                weights["rsen"] * rsen_score +
                weights["isomorph"] * isomorph_score +
                weights["vortex"] * vortex_score +
                weights["cosmic"] * cosmic_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating overall performance: {e}")
            return 0.0

def main():
    # Example usage
    model = HybridModel()
    
    # Test mathematical query
    print("\nTesting mathematical query:")
    result = model.process_query("find the derivative of x^2 * sin(x)")
    print(f"Result: {result}")
    
    # Test knowledge query
    print("\nTesting knowledge query:")
    result = model.process_query("what is a derivative in calculus?")
    print(f"Result: {result}")
    
    # Test learning
    print("\nTesting learning from results:")
    model.learn_from_result("derivative example", {
        'type': 'math_result',
        'operation': 'derivative',
        'result': 'x^2 * cos(x) + 2*x * sin(x)',
        'confidence': 1.0
    })

if __name__ == "__main__":
    main() 