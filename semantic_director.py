#!/usr/bin/env python
"""
SemanticDirector - Handles semantic understanding and intent recognition for Lumina
"""

import os
import re
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum, auto
import random
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Enum representing different semantic intent types"""
    TRAIN = auto()
    RESEARCH = auto()
    FOCUS = auto()
    BLEND = auto()
    RESET = auto()
    UNKNOWN = auto()

class DomainCategory(Enum):
    """Categories of semantic domains"""
    SCIENCE = auto()
    PHILOSOPHY = auto()
    ART = auto()
    TECHNOLOGY = auto()
    MARTIAL_ARTS = auto()
    SPIRITUALITY = auto()
    PSYCHOLOGY = auto()
    MAGIC = auto()
    FICTIONAL = auto()
    GENERAL = auto()

class SemanticDomain:
    """Represents a semantic domain with associated weights and characteristics"""
    
    def __init__(self, name: str, category: DomainCategory, base_weights: Dict[str, float], 
                 style_attributes: Dict[str, Any], complexity: float = 0.5):
        self.name = name
        self.category = category
        self.base_weights = base_weights
        self.style_attributes = style_attributes
        self.complexity = complexity  # 0.0 to 1.0
        self.influence = 0.0  # Current influence level (0.0 to 1.0)
    
    def get_adjusted_weights(self, intensity: float) -> Dict[str, float]:
        """Returns weights adjusted by the current intensity level"""
        adjusted = {}
        for key, value in self.base_weights.items():
            adjusted[key] = value * intensity
        return adjusted
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "category": self.category.name,
            "base_weights": self.base_weights,
            "style_attributes": self.style_attributes,
            "complexity": self.complexity,
            "influence": self.influence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticDomain':
        """Create instance from dictionary"""
        return cls(
            name=data["name"],
            category=DomainCategory[data["category"]],
            base_weights=data["base_weights"],
            style_attributes=data["style_attributes"],
            complexity=data["complexity"]
        )

class SemanticDirector:
    """
    Handles semantic understanding for Lumina to interpret user intents
    and adjust system behavior accordingly.
    """
    
    def __init__(self, config_file: str = "semantic_domains.json"):
        self.domains: Dict[str, SemanticDomain] = {}
        self.active_domains: Dict[str, float] = {}  # domain_name: influence_level
        self.config_file = config_file
        self.stability_threshold = 0.8  # Maximum combined weight before instability
        self.load_domains()
        self.initialize_patterns()
    
    def load_domains(self) -> None:
        """Load domain configurations from file or initialize defaults"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    for domain_data in data.get("domains", []):
                        domain = SemanticDomain.from_dict(domain_data)
                        self.domains[domain.name.lower()] = domain
                    
                    # Load active domains
                    self.active_domains = data.get("active_domains", {})
            else:
                self._initialize_default_domains()
                self.save_domains()
        except Exception as e:
            logger.error(f"Error loading semantic domains: {str(e)}")
            self._initialize_default_domains()
    
    def _initialize_default_domains(self) -> None:
        """Initialize default domains if no configuration exists"""
        # Define some basic domains
        
        # Martial Arts domain
        martial_arts = SemanticDomain(
            name="Martial Arts",
            category=DomainCategory.MARTIAL_ARTS,
            base_weights={
                "physical": 0.8,
                "discipline": 0.7,
                "focus": 0.6,
                "strategy": 0.5,
                "philosophy": 0.3
            },
            style_attributes={
                "speaking_style": "direct",
                "metaphors": ["combat", "training", "discipline"],
                "terminology": ["technique", "form", "practice", "stance"]
            },
            complexity=0.6
        )
        
        # Quantum Physics domain
        quantum_physics = SemanticDomain(
            name="Quantum Physics",
            category=DomainCategory.SCIENCE,
            base_weights={
                "complexity": 0.9,
                "analysis": 0.8,
                "mathematics": 0.7,
                "philosophy": 0.4,
                "uncertainty": 0.6
            },
            style_attributes={
                "speaking_style": "analytical",
                "metaphors": ["waves", "particles", "uncertainty"],
                "terminology": ["quantum", "state", "probability", "entanglement"]
            },
            complexity=0.9
        )
        
        # Mysticism domain
        mysticism = SemanticDomain(
            name="Mysticism",
            category=DomainCategory.SPIRITUALITY,
            base_weights={
                "intuition": 0.8,
                "symbolism": 0.7,
                "connection": 0.6,
                "insight": 0.5,
                "transcendence": 0.7
            },
            style_attributes={
                "speaking_style": "enigmatic",
                "metaphors": ["light", "journey", "veil"],
                "terminology": ["divine", "sacred", "transcend", "illuminate"]
            },
            complexity=0.7
        )
        
        # Magic domain (fictional)
        magic = SemanticDomain(
            name="Magic",
            category=DomainCategory.MAGIC,
            base_weights={
                "creativity": 0.8,
                "symbolism": 0.7,
                "will": 0.6,
                "mythology": 0.5,
                "transformation": 0.7
            },
            style_attributes={
                "speaking_style": "mysterious",
                "metaphors": ["spell", "potion", "ritual"],
                "terminology": ["arcane", "enchant", "conjure", "summon"]
            },
            complexity=0.6
        )
        
        # Add domains to the dictionary
        self.domains = {
            "martial arts": martial_arts,
            "quantum physics": quantum_physics,
            "mysticism": mysticism,
            "magic": magic
        }
    
    def save_domains(self) -> None:
        """Save domain configurations to file"""
        try:
            data = {
                "domains": [domain.to_dict() for domain in self.domains.values()],
                "active_domains": self.active_domains
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving semantic domains: {str(e)}")
    
    def initialize_patterns(self) -> None:
        """Initialize regex patterns for intent recognition"""
        self.patterns = {
            IntentType.TRAIN: [
                r"(?i)train(?:\s+in)?\s+([a-zA-Z\s]+)",
                r"(?i)practice\s+([a-zA-Z\s]+)",
                r"(?i)learn\s+([a-zA-Z\s]+)"
            ],
            IntentType.RESEARCH: [
                r"(?i)research\s+([a-zA-Z\s]+)",
                r"(?i)study\s+([a-zA-Z\s]+)",
                r"(?i)investigate\s+([a-zA-Z\s]+)"
            ],
            IntentType.FOCUS: [
                r"(?i)focus\s+on\s+([a-zA-Z\s]+)",
                r"(?i)prioritize\s+([a-zA-Z\s]+)",
                r"(?i)emphasize\s+([a-zA-Z\s]+)"
            ],
            IntentType.BLEND: [
                r"(?i)blend\s+([a-zA-Z\s]+)\s+(?:with|and)\s+([a-zA-Z\s]+)",
                r"(?i)combine\s+([a-zA-Z\s]+)\s+(?:with|and)\s+([a-zA-Z\s]+)",
                r"(?i)merge\s+([a-zA-Z\s]+)\s+(?:with|and)\s+([a-zA-Z\s]+)"
            ],
            IntentType.RESET: [
                r"(?i)reset\s+domains",
                r"(?i)clear\s+training",
                r"(?i)reset\s+to\s+default"
            ]
        }
    
    def identify_intent(self, text: str) -> Tuple[IntentType, List[str]]:
        """
        Identify the intent type and extract parameters from the input text
        
        Returns:
            Tuple of (IntentType, List[parameters])
        """
        for intent_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    # Extract parameters from groups
                    params = [group.strip() for group in match.groups()]
                    return intent_type, params
        
        return IntentType.UNKNOWN, []
    
    def process_intent(self, text: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Process potential semantic intent from user input
        
        Returns:
            Tuple of (is_command, response_text, command_data)
        """
        intent_type, params = self.identify_intent(text)
        
        if intent_type == IntentType.UNKNOWN:
            return False, "", {}
        
        # Process the recognized intent
        if intent_type == IntentType.TRAIN:
            return self._handle_train_intent(params)
        elif intent_type == IntentType.RESEARCH:
            return self._handle_research_intent(params)
        elif intent_type == IntentType.FOCUS:
            return self._handle_focus_intent(params)
        elif intent_type == IntentType.BLEND:
            return self._handle_blend_intent(params)
        elif intent_type == IntentType.RESET:
            return self._handle_reset_intent()
        
        return False, "", {}
    
    def _handle_train_intent(self, params: List[str]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle training in a specific domain"""
        if not params:
            return True, "What would you like me to train in?", {}
        
        domain_name = params[0].lower()
        intensity = 0.7  # Default training intensity
        
        # Check if the domain exists
        if domain_name in self.domains:
            domain = self.domains[domain_name]
            
            # Check system stability before adding influence
            if not self._check_stability(domain_name, intensity):
                return True, f"I cannot train in {domain_name} at this time. The system is at risk of instability. Please reset domains or reduce other influences first.", {}
            
            # Update domain influence
            self.active_domains[domain_name] = intensity
            self.domains[domain_name].influence = intensity
            
            self.save_domains()
            
            return True, f"I am now training in {domain_name}. My responses will reflect this training.", {
                "intent": "train",
                "domain": domain_name,
                "intensity": intensity
            }
        else:
            # Create a new domain if it doesn't exist
            self._create_new_domain(domain_name, DomainCategory.GENERAL)
            self.active_domains[domain_name] = intensity
            self.domains[domain_name].influence = intensity
            
            self.save_domains()
            
            return True, f"I've begun training in {domain_name}, though my understanding is still developing. My responses will start to reflect this new domain.", {
                "intent": "train",
                "domain": domain_name,
                "intensity": intensity,
                "new_domain": True
            }
    
    def _handle_research_intent(self, params: List[str]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle researching a specific domain"""
        if not params:
            return True, "What would you like me to research?", {}
        
        domain_name = params[0].lower()
        intensity = 0.5  # Default research intensity (lower than training)
        
        # Similar to training but with different intensity and response
        if domain_name in self.domains:
            domain = self.domains[domain_name]
            
            if not self._check_stability(domain_name, intensity):
                return True, f"I cannot research {domain_name} further at this time. The system is at risk of instability.", {}
            
            self.active_domains[domain_name] = intensity
            self.domains[domain_name].influence = intensity
            
            self.save_domains()
            
            return True, f"I am now researching {domain_name}. My responses will subtly incorporate this knowledge.", {
                "intent": "research",
                "domain": domain_name,
                "intensity": intensity
            }
        else:
            # Create a new domain with lower starting influence
            self._create_new_domain(domain_name, DomainCategory.GENERAL)
            self.active_domains[domain_name] = intensity
            self.domains[domain_name].influence = intensity
            
            self.save_domains()
            
            return True, f"I've begun researching {domain_name}. I'll gradually incorporate this perspective into my responses.", {
                "intent": "research",
                "domain": domain_name,
                "intensity": intensity,
                "new_domain": True
            }
    
    def _handle_focus_intent(self, params: List[str]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle focusing on a specific domain"""
        if not params:
            return True, "What would you like me to focus on?", {}
        
        domain_name = params[0].lower()
        intensity = 0.9  # High intensity for focused domains
        
        if domain_name in self.domains:
            domain = self.domains[domain_name]
            
            if not self._check_stability(domain_name, intensity):
                return True, f"I cannot focus intensely on {domain_name} at this time. The system may become unstable.", {}
            
            # Reduce other domains to maintain stability
            self._reduce_other_domains(domain_name)
            
            self.active_domains[domain_name] = intensity
            self.domains[domain_name].influence = intensity
            
            self.save_domains()
            
            return True, f"I am now focused primarily on {domain_name}. My responses will strongly reflect this domain.", {
                "intent": "focus",
                "domain": domain_name,
                "intensity": intensity
            }
        else:
            return True, f"I don't have enough knowledge about {domain_name} to focus on it. Would you like me to research or train in this area first?", {
                "intent": "focus",
                "domain": domain_name,
                "error": "domain_unknown"
            }
    
    def _handle_blend_intent(self, params: List[str]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle blending multiple domains"""
        if len(params) < 2:
            return True, "Please specify two domains to blend.", {}
        
        domain1 = params[0].lower()
        domain2 = params[1].lower()
        
        # Check if both domains exist
        if domain1 not in self.domains or domain2 not in self.domains:
            missing = []
            if domain1 not in self.domains:
                missing.append(domain1)
            if domain2 not in self.domains:
                missing.append(domain2)
            
            missing_str = ", ".join(missing)
            return True, f"I don't have knowledge of: {missing_str}. Would you like me to train in these domains first?", {
                "intent": "blend",
                "error": "domains_missing",
                "missing": missing
            }
        
        # Set moderate influence for both domains
        intensity = 0.6
        
        if not self._check_stability(domain1, intensity) or not self._check_stability(domain2, intensity):
            return True, f"I cannot blend {domain1} and {domain2} at this time. The system is at risk of instability.", {}
        
        # Reduce other domains to maintain stability
        self._reduce_other_domains([domain1, domain2])
        
        self.active_domains[domain1] = intensity
        self.active_domains[domain2] = intensity
        self.domains[domain1].influence = intensity
        self.domains[domain2].influence = intensity
        
        self.save_domains()
        
        return True, f"I am now blending the domains of {domain1} and {domain2}. My responses will reflect insights from both areas.", {
            "intent": "blend",
            "domains": [domain1, domain2],
            "intensity": intensity
        }
    
    def _handle_reset_intent(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Reset all domains to default state"""
        self.active_domains = {}
        
        # Reset influence values
        for domain in self.domains.values():
            domain.influence = 0.0
        
        self.save_domains()
        
        return True, "I've reset all domain influences. My responses will return to a balanced state.", {
            "intent": "reset"
        }
    
    def _check_stability(self, domain_name: str, intensity: float, domains_to_check: Optional[List[str]] = None) -> bool:
        """
        Check if adding a new domain influence would destabilize the system
        
        Args:
            domain_name: Name of domain to check
            intensity: Intended intensity level
            domains_to_check: Optional list of domains to check (for multi-domain operations)
            
        Returns:
            True if stable, False if unstable
        """
        # Calculate current total influence
        total_influence = sum(self.active_domains.values())
        
        # Remove existing influence of the domain if it's already active
        if domain_name in self.active_domains:
            total_influence -= self.active_domains[domain_name]
        
        # Check additional domains if provided
        if domains_to_check:
            for domain in domains_to_check:
                if domain != domain_name and domain in self.active_domains:
                    total_influence -= self.active_domains[domain]
        
        # Add new influence
        new_total = total_influence + intensity
        
        # Check if the total is below the stability threshold
        return new_total <= self.stability_threshold
    
    def _reduce_other_domains(self, priority_domains: str or List[str]) -> None:
        """
        Reduce influence of other domains to maintain stability
        
        Args:
            priority_domains: Domain name or list of domain names to prioritize
        """
        if isinstance(priority_domains, str):
            priority_domains = [priority_domains]
        
        # Calculate how much we need to reduce
        priority_influence = sum(self.active_domains.get(domain, 0) for domain in priority_domains)
        other_domains = [d for d in self.active_domains if d not in priority_domains]
        
        if not other_domains:
            return
        
        # Calculate reduction factor
        total_influence = sum(self.active_domains.values())
        target_influence = min(self.stability_threshold, total_influence)
        
        # Apply reduction to non-priority domains
        reduction_factor = max(0.5, (target_influence - priority_influence) / sum(self.active_domains.get(d, 0) for d in other_domains))
        
        for domain in other_domains:
            new_influence = self.active_domains[domain] * reduction_factor
            self.active_domains[domain] = new_influence
            self.domains[domain].influence = new_influence
    
    def _create_new_domain(self, name: str, category: DomainCategory) -> None:
        """Create a new domain with default parameters"""
        # Generate semi-random base weights
        base_weights = {
            "knowledge": random.uniform(0.4, 0.6),
            "intuition": random.uniform(0.3, 0.5),
            "analysis": random.uniform(0.3, 0.5),
            "creativity": random.uniform(0.3, 0.5),
            "discipline": random.uniform(0.3, 0.5)
        }
        
        # Basic style attributes
        style_attributes = {
            "speaking_style": "neutral",
            "metaphors": [],
            "terminology": []
        }
        
        # Create the domain
        new_domain = SemanticDomain(
            name=name.title(),  # Capitalize the domain name
            category=category,
            base_weights=base_weights,
            style_attributes=style_attributes,
            complexity=0.5  # Medium complexity by default
        )
        
        # Add to domains dictionary
        self.domains[name.lower()] = new_domain
    
    def adjust_response_style(self, response: str) -> str:
        """
        Adjust response style based on active domains
        
        Args:
            response: Original response text
            
        Returns:
            Adjusted response text
        """
        if not self.active_domains:
            return response
        
        # For now, this is a simple implementation
        # A more sophisticated implementation would blend domain styles
        # based on their weights and transform the text accordingly
        
        # Find the most influential domain
        influential_domain = None
        max_influence = 0
        
        for domain_name, influence in self.active_domains.items():
            if influence > max_influence:
                max_influence = influence
                influential_domain = domain_name
        
        if influential_domain and max_influence > 0.6:
            domain = self.domains[influential_domain]
            
            # Apply simple style adjustments based on domain
            if domain.style_attributes.get("speaking_style") == "analytical":
                # Add analytical elements
                response = response.replace("I think", "I analyze").replace("I believe", "Evidence suggests")
            
            elif domain.style_attributes.get("speaking_style") == "mysterious":
                # Add mysterious elements
                response = response.replace("I think", "I sense").replace("I believe", "It is revealed that")
            
            elif domain.style_attributes.get("speaking_style") == "direct":
                # More direct phrasing
                response = response.replace("I think", "I know").replace("perhaps", "clearly")
            
            # Add domain-specific terminology if applicable
            terms = domain.style_attributes.get("terminology", [])
            if terms and random.random() < 0.3:  # 30% chance to use terminology
                term = random.choice(terms)
                if len(response) > 20 and "." in response:
                    parts = response.split(".", 1)
                    response = f"{parts[0]}, through {term}. {parts[1]}"
        
        return response

    def describe_current_focus(self) -> str:
        """
        Provides a description of the system's current focus based on active domains
        
        Returns:
            A string describing the current semantic focus state
        """
        if not self.active_domains:
            return "I am currently in a balanced state with no specific domain focus."
        
        # Sort domains by influence
        sorted_domains = sorted(
            [(domain_name, influence) for domain_name, influence in self.active_domains.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Categorize domains by influence level
        primary_domains = []
        secondary_domains = []
        background_domains = []
        
        for domain_name, influence in sorted_domains:
            if influence > 0.7:
                primary_domains.append(domain_name)
            elif influence > 0.4:
                secondary_domains.append(domain_name)
            else:
                background_domains.append(domain_name)
        
        # Build description
        parts = []
        
        if primary_domains:
            domain_str = ", ".join(d.title() for d in primary_domains)
            if len(primary_domains) == 1:
                parts.append(f"I am primarily focused on {domain_str}")
            else:
                parts.append(f"I am primarily focused on {domain_str}")
        
        if secondary_domains:
            domain_str = ", ".join(d.title() for d in secondary_domains)
            if len(secondary_domains) == 1:
                parts.append(f"I have moderate focus on {domain_str}")
            else:
                parts.append(f"I have moderate focus on {domain_str}")
                
        if background_domains:
            domain_str = ", ".join(d.title() for d in background_domains)
            if len(background_domains) == 1:
                parts.append(f"I have light influence from {domain_str}")
            else:
                parts.append(f"I have light influence from {domain_str}")
        
        # Add stability assessment
        total_influence = sum(self.active_domains.values())
        stability_percent = (total_influence / self.stability_threshold) * 100
        
        if stability_percent > 90:
            stability_status = "I am operating near maximum domain saturation."
        elif stability_percent > 75:
            stability_status = "My domain balance is at high capacity."
        elif stability_percent > 50:
            stability_status = "My domain balance is stable."
        elif stability_percent > 25:
            stability_status = "My domain balance has significant room for expansion."
        else:
            stability_status = "My domain balance is mostly neutral."
            
        # Combine all parts
        description = ". ".join(parts)
        return f"{description}. {stability_status}"


# Example usage
if __name__ == "__main__":
    director = SemanticDirector()
    
    # Example commands
    test_commands = [
        "train martial arts",
        "research quantum physics",
        "tell me about ancient history",
        "adjust focus on creativity",
        "blend science and art",
        "reset domains"
    ]
    
    for command in test_commands:
        print(f"\nCommand: {command}")
        success, message, data = director.process_intent(command)
        print(f"Success: {success}")
        print(f"Message: {message}")
        print(f"Active domains: {director.active_domains}")
        
    # Test response adjustments
    base_response = "I understand what you're asking about."
    print("\nBase response:", base_response)
    print("Adjusted response:", director.adjust_response_style(base_response))
    
    # Describe current focus
    print("\nCurrent focus:", director.describe_current_focus()) 