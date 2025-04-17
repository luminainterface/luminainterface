"""
V5 Panels Package

This package contains the visualization panels for the V5 Fractal Echo Visualization system.
"""

# Import panels for easier access
try:
    from .fractal_pattern_panel import FractalPatternPanel
    from .node_consciousness_panel import NodeConsciousnessPanel
    from .memory_synthesis_panel import MemorySynthesisPanel
    from .network_visualization_panel import NetworkVisualizationPanel
    from .metrics_panel import MetricsPanel
    from .conversation_panel import ConversationPanel
except ImportError as e:
    # Log error but continue
    import logging
    logging.getLogger(__name__).error(f"Error importing panels: {e}")

# Import other panels as they are implemented 