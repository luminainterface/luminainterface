# Import visualization components
from consciousness.visualization import ConsciousnessVisualizer
# Import language visualizer
from language.language_visualizer import get_language_visualizer, set_llm_weight

def initialize_components():
    # Initialize consciousness components if needed
    if "consciousness" in config.active_modules:
        logging.info("Initializing consciousness components...")
        # Initialize any required consciousness components
        
        # Initialize consciousness visualizer
        consciousness_visualizer = ConsciousnessVisualizer(
            output_dir=os.path.join(config.data_dir, "visualizations", "consciousness")
        )
        components["consciousness_visualizer"] = consciousness_visualizer
    
    # Initialize language components if needed
    if "language" in config.active_modules:
        logging.info("Initializing language components...")
        # Initialize central language node if available
        central_node = None
        if "central_language_node" in components:
            central_node = components["central_language_node"]
        
        # Initialize language visualizer
        language_visualizer = get_language_visualizer(
            central_node=central_node,
            output_dir=os.path.join(config.data_dir, "visualizations", "language")
        )
        components["language_visualizer"] = language_visualizer
    
    return components

def connect_monitoring_dashboard(components, monitoring_module):
    """Connect components to the monitoring dashboard for visualization and metrics."""
    if not monitoring_module or not hasattr(monitoring_module, "register_visualization_source"):
        logging.warning("Monitoring module not available or missing registration function")
        return
    
    # Register dream mode visualization if available
    if "dream_controller" in components:
        dream_controller = components["dream_controller"]
        monitoring_module.register_visualization_source(
            "dream-mode", 
            dream_controller.get_visualization_data
        )
        logging.info("Registered Dream Mode visualization with monitoring dashboard")
    
    # Register consciousness visualization if available
    if "consciousness_visualizer" in components:
        consciousness_vis = components["consciousness_visualizer"]
        monitoring_module.register_visualization_source(
            "consciousness", 
            consciousness_vis.get_dashboard_data
        )
        logging.info("Registered Consciousness visualization with monitoring dashboard")
        
    # Register language module visualization if available
    if "language_visualizer" in components:
        language_vis = components["language_visualizer"]
        monitoring_module.register_visualization_source(
            "language", 
            language_vis.get_dashboard_data
        )
        # Register weight setter function
        monitoring_module.register_visualization_source(
            "language_weight_setter",
            set_llm_weight
        )
        logging.info("Registered Language Module visualization with monitoring dashboard")

def run_v7():
    """Main entry point for LUMINA V7."""
    try:
        components = initialize_components()
        
        # Initialize monitoring and connect visualizations
        if "monitoring" in sys.modules:
            import monitoring.dashboard as monitoring_module
            connect_monitoring_dashboard(components, monitoring_module)
    except Exception as e:
        logging.error(f"Error in run_v7: {e}")
        sys.exit(1) 