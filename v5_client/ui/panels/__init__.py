"""
V5 PySide6 Client UI Panels Package

UI panel components for the V5 Fractal Echo Visualization System.
"""

def get_available_panels():
    """
    Get a list of all available panels
    
    Returns:
        Dictionary mapping panel IDs to panel classes
    """
    from .fractal_pattern_panel import FractalPatternPanel
    from .memory_synthesis_panel import MemorySynthesisPanel
    from .node_consciousness_panel import NodeConsciousnessPanel
    from .conversation_panel import ConversationPanel
    
    return {
        "fractal_pattern": FractalPatternPanel,
        "memory_synthesis": MemorySynthesisPanel,
        "node_consciousness": NodeConsciousnessPanel,
        "conversation": ConversationPanel
    } 