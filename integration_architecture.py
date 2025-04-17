#!/usr/bin/env python3
"""
Integration Architecture Diagram Generator
This script generates a visual representation of the integration 
between V5 Fractal Echo Visualization system and the Language Memory System.
"""

import os
import sys
from graphviz import Digraph

def create_integration_diagram(output_file="v5_language_integration", view=True):
    """
    Creates a diagram showing the integration between V5 components and the Language Memory System.
    
    Args:
        output_file (str): Name of the output file (without extension)
        view (bool): Whether to open the diagram after creation
    """
    # Create a new directed graph
    dot = Digraph(comment='V5-Language Memory Integration Architecture', 
                  format='png', 
                  engine='dot')
    
    # Set graph attributes
    dot.attr(rankdir='TB', 
             size='8,10', 
             ratio='fill',
             fontname='Arial',
             label='V5-Language Memory Integration Architecture',
             labelloc='t',
             fontsize='20')
    
    # Node styles
    dot.attr('node', shape='box', style='filled,rounded', 
             fontname='Arial', fontsize='12', margin='0.3,0.1')
    
    # Define color schemes for different component types
    colors = {
        'v5': '#a6cee3',         # Light blue for V5 components
        'language': '#b2df8a',   # Light green for Language components
        'bridge': '#fb9a99',     # Light red for Bridge components
        'ui': '#fdbf6f',         # Light orange for UI components
        'core': '#cab2d6',       # Light purple for Core components
        'data': '#ffff99'        # Light yellow for Data stores
    }
    
    # Create clusters/subgraphs
    
    # V5 Visualization System
    with dot.subgraph(name='cluster_v5') as c:
        c.attr(label='V5 Fractal Echo Visualization System', style='filled', color='#e6f3ff', fontsize='16')
        
        c.node('fractal_processor', 'Fractal Pattern Processor', fillcolor=colors['v5'])
        c.node('node_consciousness', 'Node Consciousness Metrics', fillcolor=colors['v5'])
        c.node('v5_ui', 'V5 Visualization Interface', fillcolor=colors['ui'])
        c.node('pattern_storage', 'Pattern Storage', shape='cylinder', fillcolor=colors['data'])
        
        c.edge('fractal_processor', 'node_consciousness')
        c.edge('node_consciousness', 'v5_ui')
        c.edge('fractal_processor', 'pattern_storage')
        c.edge('pattern_storage', 'v5_ui', style='dashed')
    
    # Language Memory System
    with dot.subgraph(name='cluster_language') as c:
        c.attr(label='Language Memory System', style='filled', color='#e6ffe6', fontsize='16')
        
        c.node('neural_linguistic', 'Neural Linguistic Processor', fillcolor=colors['language'])
        c.node('memory_api', 'Memory API', fillcolor=colors['language'])
        c.node('memory_storage', 'Memory Storage', shape='cylinder', fillcolor=colors['data'])
        c.node('topic_extractor', 'Topic Extraction Engine', fillcolor=colors['language'])
        
        c.edge('neural_linguistic', 'memory_api')
        c.edge('memory_api', 'memory_storage', dir='both')
        c.edge('topic_extractor', 'memory_api')
        c.edge('memory_storage', 'topic_extractor', style='dashed')
    
    # Bridge Components
    with dot.subgraph(name='cluster_bridge') as c:
        c.attr(label='Integration Bridge Components', style='filled', color='#fff2f2', fontsize='16')
        
        c.node('memory_v5_bridge', 'Language Memory V5 Bridge', fillcolor=colors['bridge'])
        c.node('chat_memory_interface', 'Chat Memory Interface', fillcolor=colors['bridge'])
        c.node('node_socket', 'Node Socket Manager', fillcolor=colors['bridge'])
        
        c.edge('memory_v5_bridge', 'node_socket')
        c.edge('chat_memory_interface', 'node_socket')
    
    # User Interface Components
    with dot.subgraph(name='cluster_ui') as c:
        c.attr(label='User Interface Components', style='filled', color='#fff8e6', fontsize='16')
        
        c.node('conversation_panel', 'Weighted Conversation Panel', fillcolor=colors['ui'])
        c.node('ui_controller', 'UI Controller', fillcolor=colors['ui'])
        
        c.edge('conversation_panel', 'ui_controller')
    
    # Add edges between clusters
    dot.edge('memory_v5_bridge', 'memory_api', constraint='true')
    dot.edge('memory_v5_bridge', 'fractal_processor', constraint='true')
    dot.edge('chat_memory_interface', 'memory_api', constraint='true')
    dot.edge('chat_memory_interface', 'conversation_panel', constraint='true')
    dot.edge('node_socket', 'v5_ui', constraint='true')
    dot.edge('fractal_processor', 'neural_linguistic', style='dotted', constraint='false', 
             label='pattern exchange')
    dot.edge('ui_controller', 'v5_ui', constraint='true')
    
    # Create directory for output if it doesn't exist
    output_dir = "diagrams"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Render the diagram
    output_path = os.path.join(output_dir, output_file)
    dot.render(output_path, view=view)
    print(f"Diagram created at: {output_path}.png")
    
    return output_path + ".png"

def create_data_flow_diagram(output_file="v5_language_data_flow", view=True):
    """
    Creates a diagram showing the data flow between components
    
    Args:
        output_file (str): Name of the output file (without extension)
        view (bool): Whether to open the diagram after creation
    """
    # Create a new directed graph
    dot = Digraph(comment='V5-Language Memory Data Flow', 
                  format='png', 
                  engine='dot')
    
    # Set graph attributes
    dot.attr(rankdir='LR', 
             size='10,8', 
             ratio='fill',
             fontname='Arial',
             label='V5-Language Memory Data Flow',
             labelloc='t',
             fontsize='20')
    
    # Node styles
    dot.attr('node', shape='box', style='filled,rounded', 
             fontname='Arial', fontsize='12', margin='0.3,0.1')
    
    # Edge styles
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # Define color schemes for different data types
    colors = {
        'component': '#a6cee3',  # Light blue for components
        'memory': '#b2df8a',     # Light green for memory data
        'pattern': '#fb9a99',    # Light red for pattern data
        'message': '#fdbf6f',    # Light orange for messages
        'config': '#cab2d6',     # Light purple for configuration
    }
    
    # Create nodes
    dot.node('user', 'User', shape='ellipse', fillcolor='#ffffff')
    dot.node('conversation', 'Conversation Panel', fillcolor=colors['component'])
    dot.node('chat_interface', 'Chat Memory Interface', fillcolor=colors['component'])
    dot.node('memory_bridge', 'Language Memory V5 Bridge', fillcolor=colors['component'])
    dot.node('memory_system', 'Language Memory System', fillcolor=colors['component'])
    dot.node('v5_system', 'V5 Visualization System', fillcolor=colors['component'])
    
    # Create edges with data flow labels
    dot.edge('user', 'conversation', label='User Input')
    
    dot.edge('conversation', 'chat_interface', 
             label='Message with\nNeural Weight')
    
    dot.edge('chat_interface', 'memory_system', 
             label='Store/Retrieve\nMemories')
    
    dot.edge('memory_system', 'chat_interface', 
             label='Memory Results\nwith Context')
    
    dot.edge('chat_interface', 'conversation', 
             label='Memory-Enhanced\nResponse')
    
    dot.edge('memory_system', 'memory_bridge', 
             label='Memory Topics\nand Contents')
    
    dot.edge('memory_bridge', 'v5_system', 
             label='Fractal Pattern\nGeneration Data')
    
    dot.edge('v5_system', 'memory_bridge', 
             label='Node Consciousness\nMetrics')
    
    dot.edge('memory_bridge', 'memory_system', 
             label='Visual Pattern\nMapping Data')
    
    dot.edge('v5_system', 'user', 
             label='Visual\nFeedback', constraint='false')
    
    # Create directory for output if it doesn't exist
    output_dir = "diagrams"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Render the diagram
    output_path = os.path.join(output_dir, output_file)
    dot.render(output_path, view=view)
    print(f"Data flow diagram created at: {output_path}.png")
    
    return output_path + ".png"

if __name__ == "__main__":
    # Check if graphviz is installed
    try:
        import graphviz
        print("Generating V5-Language Memory integration diagrams...")
        
        # Parse command line arguments
        view_diagram = True
        if len(sys.argv) > 1 and sys.argv[1] == "--no-view":
            view_diagram = False
            
        # Create both diagrams
        architecture_diagram = create_integration_diagram(view=view_diagram)
        data_flow_diagram = create_data_flow_diagram(view=view_diagram)
        
        print("\nDiagram generation complete!")
        print(f"Architecture diagram: {architecture_diagram}")
        print(f"Data flow diagram: {data_flow_diagram}")
        print("\nTo generate these diagrams without opening them:")
        print("  python integration_architecture.py --no-view")
        
    except ImportError:
        print("Error: This script requires graphviz to be installed.")
        print("Please install it using:")
        print("  pip install graphviz")
        print("You may also need to install the Graphviz software:")
        print("  - Windows: Install from https://graphviz.org/download/")
        print("  - Linux: sudo apt-get install graphviz")
        print("  - macOS: brew install graphviz")
        sys.exit(1) 