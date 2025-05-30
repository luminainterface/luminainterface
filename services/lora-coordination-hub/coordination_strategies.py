#!/usr/bin/env python3
"""
Coordination Strategies
=======================

Strategy management for coordinating multiple LoRA systems based on
query analysis and neural thought engine recommendations.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CoordinationMode(Enum):
    AUTO = "auto"
    TARGETED = "targeted"
    FULL = "full"
    MINIMAL = "minimal"

@dataclass
class CoordinationStrategy:
    """Defines how LoRA systems should be coordinated for a specific query"""
    name: str
    selected_systems: List[str]
    system_roles: Dict[str, str]
    synthesis_approach: str
    reasoning: str
    effectiveness_score: float
    parallel_processing: bool = True
    fallback_systems: List[str] = None

class CoordinationStrategyManager:
    """Manages coordination strategies for different query types and scenarios"""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
        self.performance_history = {}
        
    def _initialize_strategies(self) -> Dict[str, CoordinationStrategy]:
        """Initialize predefined coordination strategies"""
        return {
            # Creative and narrative queries
            "creative_coordination": CoordinationStrategy(
                name="creative_coordination",
                selected_systems=["jarvis_chat", "enhanced_prompt_lora", "npu_enhanced_lora"],
                system_roles={
                    "jarvis_chat": "primary_processor",
                    "enhanced_prompt_lora": "concept_enhancer",
                    "npu_enhanced_lora": "quality_optimizer"
                },
                synthesis_approach="creative_weighted",
                reasoning="Creative queries benefit from Jarvis personality, concept enhancement, and quality optimization",
                effectiveness_score=0.85,
                fallback_systems=["enhanced_prompt_lora", "jarvis_chat"]
            ),
            
            # Technical and programming queries
            "technical_coordination": CoordinationStrategy(
                name="technical_coordination",
                selected_systems=["optimal_lora_router", "npu_enhanced_lora", "enhanced_prompt_lora"],
                system_roles={
                    "optimal_lora_router": "primary_processor",
                    "npu_enhanced_lora": "performance_optimizer",
                    "enhanced_prompt_lora": "concept_validator"
                },
                synthesis_approach="technical_weighted",
                reasoning="Technical queries require specialized LoRA routing, NPU acceleration, and concept validation",
                effectiveness_score=0.90,
                fallback_systems=["npu_enhanced_lora", "enhanced_prompt_lora"]
            ),
            
            # Complex analytical queries
            "analytical_coordination": CoordinationStrategy(
                name="analytical_coordination",
                selected_systems=["npu_enhanced_lora", "enhanced_prompt_lora", "optimal_lora_router", "npu_adapter_selector"],
                system_roles={
                    "npu_enhanced_lora": "primary_processor",
                    "enhanced_prompt_lora": "concept_analyzer",
                    "optimal_lora_router": "specialized_enhancer",
                    "npu_adapter_selector": "adapter_optimizer"
                },
                synthesis_approach="analytical_weighted",
                reasoning="Complex queries need multi-system analysis with NPU acceleration and specialized adapters",
                effectiveness_score=0.88,
                fallback_systems=["npu_enhanced_lora", "enhanced_prompt_lora"]
            ),
            
            # Full system coordination for maximum quality
            "full_coordination": CoordinationStrategy(
                name="full_coordination",
                selected_systems=["optimal_lora_router", "enhanced_prompt_lora", "npu_enhanced_lora", "npu_adapter_selector", "jarvis_chat"],
                system_roles={
                    "optimal_lora_router": "specialized_processor",
                    "enhanced_prompt_lora": "concept_enhancer",
                    "npu_enhanced_lora": "primary_processor",
                    "npu_adapter_selector": "adapter_selector",
                    "jarvis_chat": "personality_provider"
                },
                synthesis_approach="neural_weighted",
                reasoning="Maximum quality coordination using all available systems with neural synthesis",
                effectiveness_score=0.92,
                parallel_processing=False,  # Sequential for quality
                fallback_systems=["npu_enhanced_lora", "enhanced_prompt_lora", "jarvis_chat"]
            ),
            
            # Standard general-purpose coordination
            "standard_coordination": CoordinationStrategy(
                name="standard_coordination",
                selected_systems=["npu_enhanced_lora", "enhanced_prompt_lora"],
                system_roles={
                    "npu_enhanced_lora": "primary_processor",
                    "enhanced_prompt_lora": "quality_enhancer"
                },
                synthesis_approach="balanced_weighted",
                reasoning="Balanced coordination for general queries with good performance and quality",
                effectiveness_score=0.75,
                fallback_systems=["enhanced_prompt_lora"]
            ),
            
            # Minimal coordination for simple queries
            "minimal_coordination": CoordinationStrategy(
                name="minimal_coordination",
                selected_systems=["enhanced_prompt_lora"],
                system_roles={
                    "enhanced_prompt_lora": "primary_processor"
                },
                synthesis_approach="direct",
                reasoning="Single system processing for simple queries to optimize performance",
                effectiveness_score=0.65,
                fallback_systems=["npu_enhanced_lora"]
            ),
            
            # Adapter-focused coordination
            "adapter_coordination": CoordinationStrategy(
                name="adapter_coordination",
                selected_systems=["npu_adapter_selector", "optimal_lora_router", "npu_enhanced_lora"],
                system_roles={
                    "npu_adapter_selector": "adapter_selector",
                    "optimal_lora_router": "lora_processor",
                    "npu_enhanced_lora": "quality_optimizer"
                },
                synthesis_approach="adapter_weighted",
                reasoning="Specialized coordination focusing on optimal adapter selection and LoRA processing",
                effectiveness_score=0.82,
                fallback_systems=["optimal_lora_router", "npu_enhanced_lora"]
            ),
            
            # Performance-optimized coordination
            "performance_coordination": CoordinationStrategy(
                name="performance_coordination",
                selected_systems=["npu_enhanced_lora", "optimal_lora_router"],
                system_roles={
                    "npu_enhanced_lora": "primary_processor",
                    "optimal_lora_router": "performance_optimizer"
                },
                synthesis_approach="performance_weighted",
                reasoning="Speed-optimized coordination using NPU acceleration and efficient LoRA routing",
                effectiveness_score=0.78,
                fallback_systems=["npu_enhanced_lora"]
            )
        }
    
    async def select_strategy(self, neural_analysis: Dict[str, Any], available_systems: List[str], 
                            preferred_systems: Optional[List[str]] = None, 
                            coordination_mode: str = "auto") -> CoordinationStrategy:
        """
        Select optimal coordination strategy based on neural analysis and system availability
        
        Args:
            neural_analysis: Analysis from Neural Thought Engine
            available_systems: List of currently available LoRA systems
            preferred_systems: User-preferred systems (optional)
            coordination_mode: Coordination mode preference
            
        Returns:
            Selected coordination strategy
        """
        logger.info(f"Selecting coordination strategy for mode: {coordination_mode}")
        
        # Extract key factors from neural analysis
        query_complexity = neural_analysis.get("query_complexity", 0.5)
        detected_domains = neural_analysis.get("detected_domains", [])
        coordination_strategy_hint = neural_analysis.get("coordination_strategy", "auto")
        confidence = neural_analysis.get("confidence", 0.5)
        
        # Apply coordination mode logic
        if coordination_mode == "minimal":
            strategy = self.strategies["minimal_coordination"]
        elif coordination_mode == "full":
            strategy = self.strategies["full_coordination"]
        elif coordination_mode == "targeted" and preferred_systems:
            strategy = await self._create_targeted_strategy(preferred_systems, neural_analysis)
        else:
            # Auto mode - select based on analysis
            strategy = await self._select_auto_strategy(
                query_complexity, detected_domains, coordination_strategy_hint, confidence
            )
        
        # Filter strategy based on available systems
        filtered_strategy = await self._filter_strategy_by_availability(strategy, available_systems)
        
        # Update strategy performance tracking
        await self._update_strategy_performance(filtered_strategy.name, neural_analysis)
        
        logger.info(f"Selected strategy: {filtered_strategy.name} with systems: {filtered_strategy.selected_systems}")
        return filtered_strategy
    
    async def _select_auto_strategy(self, complexity: float, domains: List[str], 
                                  hint: str, confidence: float) -> CoordinationStrategy:
        """Select strategy automatically based on query characteristics"""
        
        # Neural thought engine hint takes priority
        if hint in self.strategies:
            return self.strategies[hint]
        
        # Domain-based selection
        domain_str = " ".join(domains).lower()
        
        if any(word in domain_str for word in ["creative", "story", "narrative", "character", "fiction"]):
            return self.strategies["creative_coordination"]
        
        elif any(word in domain_str for word in ["programming", "code", "technical", "algorithm", "function"]):
            return self.strategies["technical_coordination"]
        
        elif any(word in domain_str for word in ["analysis", "research", "complex", "detailed", "academic"]):
            return self.strategies["analytical_coordination"]
        
        # Complexity-based selection
        elif complexity > 0.8:
            return self.strategies["full_coordination"]
        
        elif complexity > 0.6:
            return self.strategies["standard_coordination"]
        
        elif complexity < 0.3:
            return self.strategies["minimal_coordination"]
        
        # Confidence-based selection
        elif confidence > 0.8:
            return self.strategies["performance_coordination"]
        
        else:
            return self.strategies["standard_coordination"]
    
    async def _create_targeted_strategy(self, preferred_systems: List[str], 
                                      neural_analysis: Dict[str, Any]) -> CoordinationStrategy:
        """Create a custom strategy based on preferred systems"""
        
        # Assign roles based on system capabilities
        roles = {}
        synthesis_approach = "balanced_weighted"
        
        if "npu_enhanced_lora" in preferred_systems:
            roles["npu_enhanced_lora"] = "primary_processor"
            synthesis_approach = "npu_weighted"
        
        if "enhanced_prompt_lora" in preferred_systems:
            roles["enhanced_prompt_lora"] = "concept_enhancer" if "npu_enhanced_lora" in preferred_systems else "primary_processor"
        
        if "optimal_lora_router" in preferred_systems:
            roles["optimal_lora_router"] = "specialized_processor"
        
        if "jarvis_chat" in preferred_systems:
            roles["jarvis_chat"] = "personality_provider"
        
        if "npu_adapter_selector" in preferred_systems:
            roles["npu_adapter_selector"] = "adapter_optimizer"
        
        # Fill remaining roles
        role_priority = ["quality_enhancer", "performance_optimizer", "fallback_processor"]
        assigned_roles = set(roles.values())
        
        for system in preferred_systems:
            if system not in roles:
                for role in role_priority:
                    if role not in assigned_roles:
                        roles[system] = role
                        assigned_roles.add(role)
                        break
                else:
                    roles[system] = "general_processor"
        
        return CoordinationStrategy(
            name="targeted_custom",
            selected_systems=preferred_systems,
            system_roles=roles,
            synthesis_approach=synthesis_approach,
            reasoning=f"Custom strategy targeting user-preferred systems: {', '.join(preferred_systems)}",
            effectiveness_score=0.70,  # Default for custom strategies
            fallback_systems=preferred_systems[:2] if len(preferred_systems) > 1 else preferred_systems
        )
    
    async def _filter_strategy_by_availability(self, strategy: CoordinationStrategy, 
                                             available_systems: List[str]) -> CoordinationStrategy:
        """Filter strategy to only include available systems"""
        
        # Find available systems from strategy
        available_selected = [sys for sys in strategy.selected_systems if sys in available_systems]
        
        if not available_selected:
            # No systems available - create emergency fallback
            if available_systems:
                logger.warning(f"No strategy systems available, using emergency fallback with: {available_systems[0]}")
                return CoordinationStrategy(
                    name="emergency_fallback",
                    selected_systems=[available_systems[0]],
                    system_roles={available_systems[0]: "emergency_processor"},
                    synthesis_approach="direct",
                    reasoning="Emergency fallback due to system unavailability",
                    effectiveness_score=0.3
                )
            else:
                # Complete system failure
                return CoordinationStrategy(
                    name="system_failure",
                    selected_systems=[],
                    system_roles={},
                    synthesis_approach="none",
                    reasoning="No systems available",
                    effectiveness_score=0.0
                )
        
        # If we have fewer systems than planned, adjust roles
        if len(available_selected) < len(strategy.selected_systems):
            # Filter roles to match available systems
            filtered_roles = {sys: role for sys, role in strategy.system_roles.items() 
                            if sys in available_selected}
            
            # If primary processor is missing, reassign
            primary_processors = [sys for sys, role in filtered_roles.items() 
                                if role == "primary_processor"]
            
            if not primary_processors and available_selected:
                # Assign primary processor to best available system
                best_system = self._select_best_primary_system(available_selected)
                filtered_roles[best_system] = "primary_processor"
            
            return CoordinationStrategy(
                name=f"{strategy.name}_filtered",
                selected_systems=available_selected,
                system_roles=filtered_roles,
                synthesis_approach=strategy.synthesis_approach,
                reasoning=f"{strategy.reasoning} (Filtered for system availability)",
                effectiveness_score=strategy.effectiveness_score * 0.9,  # Slight penalty for filtering
                parallel_processing=strategy.parallel_processing,
                fallback_systems=[sys for sys in strategy.fallback_systems or [] if sys in available_systems]
            )
        
        # All systems available
        return strategy
    
    def _select_best_primary_system(self, available_systems: List[str]) -> str:
        """Select the best system to serve as primary processor"""
        
        # Priority order for primary processing
        priority_order = [
            "npu_enhanced_lora",
            "enhanced_prompt_lora", 
            "optimal_lora_router",
            "jarvis_chat",
            "npu_adapter_selector"
        ]
        
        for system in priority_order:
            if system in available_systems:
                return system
        
        # Return first available if none match priority
        return available_systems[0] if available_systems else ""
    
    async def _update_strategy_performance(self, strategy_name: str, neural_analysis: Dict[str, Any]):
        """Update performance tracking for strategies"""
        
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = {
                "usage_count": 0,
                "average_complexity": 0.0,
                "success_rate": 0.0,
                "average_confidence": 0.0
            }
        
        history = self.performance_history[strategy_name]
        count = history["usage_count"]
        
        # Update running averages
        complexity = neural_analysis.get("query_complexity", 0.5)
        confidence = neural_analysis.get("confidence", 0.5)
        
        history["average_complexity"] = (history["average_complexity"] * count + complexity) / (count + 1)
        history["average_confidence"] = (history["average_confidence"] * count + confidence) / (count + 1)
        history["usage_count"] += 1
        
        logger.debug(f"Updated performance tracking for {strategy_name}: {history}")
    
    async def get_strategy_recommendations(self, query_characteristics: Dict[str, Any]) -> List[str]:
        """Get strategy recommendations based on query characteristics"""
        
        recommendations = []
        complexity = query_characteristics.get("complexity", 0.5)
        domains = query_characteristics.get("domains", [])
        performance_priority = query_characteristics.get("performance_priority", False)
        
        # Add recommendations based on characteristics
        if complexity > 0.8:
            recommendations.append("full_coordination")
            recommendations.append("analytical_coordination")
        
        elif complexity < 0.3:
            recommendations.append("minimal_coordination")
            recommendations.append("performance_coordination")
        
        if performance_priority:
            recommendations.append("performance_coordination")
            recommendations.append("minimal_coordination")
        
        # Domain-specific recommendations
        domain_str = " ".join(domains).lower()
        if "creative" in domain_str:
            recommendations.append("creative_coordination")
        if "technical" in domain_str:
            recommendations.append("technical_coordination")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("standard_coordination")
        
        return recommendations
    
    def get_all_strategies(self) -> Dict[str, CoordinationStrategy]:
        """Get all available coordination strategies"""
        return self.strategies.copy()
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance history for all strategies"""
        return self.performance_history.copy() 