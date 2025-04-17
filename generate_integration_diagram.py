import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import os

def create_integration_diagram():
    """Create integration architecture diagram using matplotlib."""
    # Create a figure
    plt.figure(figsize=(12, 9))
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for V5 system
    v5_nodes = [
        ('v5_core', 'V5 Core Engine'),
        ('fractal_processor', 'Fractal Pattern Processor'),
        ('node_metrics', 'Node Consciousness Metrics'),
        ('v5_socket_mgr', 'Socket Manager'),
        ('v5_plugins', 'Plugin System')
    ]
    
    # Add nodes for Language Memory system
    lang_nodes = [
        ('lang_core', 'Language Memory Core'),
        ('memory_db', 'Memory Database'),
        ('topic_synth', 'Topic Synthesis Engine'),
        ('api_provider', 'Memory API Provider')
    ]
    
    # Add special nodes
    special_nodes = [
        ('bridge', 'Language Memory V5 Bridge'),
        ('neural_core', 'Neural Network Core'),
        ('data_flow', 'Message Queue System')
    ]
    
    # Add transition nodes
    transition_nodes = [
        ('v1_adapter', 'V1-V2 Interface Adapter'),
        ('v3_adapter', 'V3-V4 Interface Adapter'),
        ('compat_layer', 'Compatibility Layer')
    ]
    
    # Add all nodes to graph
    for node_id, label in v5_nodes + lang_nodes + special_nodes + transition_nodes:
        G.add_node(node_id, label=label)
    
    # Add edges for V5 system
    G.add_edge('v5_core', 'fractal_processor')
    G.add_edge('v5_core', 'node_metrics')
    G.add_edge('v5_core', 'v5_socket_mgr')
    G.add_edge('v5_socket_mgr', 'v5_plugins')
    
    # Add edges for Language Memory system
    G.add_edge('lang_core', 'memory_db')
    G.add_edge('lang_core', 'topic_synth')
    G.add_edge('lang_core', 'api_provider')
    
    # Add bridge connections
    G.add_edge('api_provider', 'bridge', label='Memory Data\nTopics\nAssociations')
    G.add_edge('bridge', 'v5_socket_mgr', label='Fractal Params\nVisualization Data')
    
    # Add neural core connections
    G.add_edge('neural_core', 'lang_core', label='Neural State')
    G.add_edge('neural_core', 'v5_core', label='Neural State')
    
    # Add message queue connections
    G.add_edge('bridge', 'data_flow', label='Bidirectional\nMessage Flow')
    G.add_edge('v5_socket_mgr', 'data_flow')
    G.add_edge('api_provider', 'data_flow')
    
    # Add transition layer connections
    G.add_edge('v1_adapter', 'compat_layer')
    G.add_edge('v3_adapter', 'compat_layer')
    G.add_edge('compat_layer', 'bridge', label='Version\nCompatibility')
    
    # Define node positions (manually for better layout)
    pos = {
        # V5 system nodes (top right)
        'v5_core': (0.7, 0.8),
        'fractal_processor': (0.9, 0.7),
        'node_metrics': (0.7, 0.7),
        'v5_socket_mgr': (0.5, 0.7),
        'v5_plugins': (0.5, 0.6),
        
        # Language Memory system nodes (top left)
        'lang_core': (0.3, 0.8),
        'memory_db': (0.1, 0.7),
        'topic_synth': (0.3, 0.7),
        'api_provider': (0.5, 0.8),
        
        # Special nodes
        'bridge': (0.5, 0.5),
        'neural_core': (0.5, 0.9),
        'data_flow': (0.5, 0.4),
        
        # Transition nodes (bottom)
        'v1_adapter': (0.3, 0.3),
        'v3_adapter': (0.7, 0.3),
        'compat_layer': (0.5, 0.3)
    }
    
    # Draw the graph
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n, _ in v5_nodes], 
                          node_color='lightblue', 
                          node_size=2000, 
                          alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n, _ in lang_nodes], 
                          node_color='lavender', 
                          node_size=2000, 
                          alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=['bridge'], 
                          node_color='lightyellow', 
                          node_size=3000, 
                          alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=['neural_core'], 
                          node_color='lightgreen', 
                          node_size=2500, 
                          alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=['data_flow'], 
                          node_color='khaki', 
                          node_size=2500, 
                          alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n, _ in transition_nodes], 
                          node_color='bisque', 
                          node_size=2000, 
                          alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray', 
                          arrows=True,
                          arrowsize=15,
                          width=1.5)
    
    # Draw labels
    node_labels = {node: data['label'] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, 
                           labels=node_labels, 
                           font_size=9, 
                           font_family='sans-serif')
    
    # Draw edge labels
    edge_labels = {(u, v): data['label'] for u, v, data in G.edges(data=True) if 'label' in data}
    nx.draw_networkx_edge_labels(G, pos, 
                                edge_labels=edge_labels, 
                                font_size=8)
    
    # Add system boxes
    # V5 System
    v5_box = patches.Rectangle((0.45, 0.55), 0.5, 0.3, 
                              fill=True, alpha=0.1, 
                              color='blue', zorder=-1)
    plt.gca().add_patch(v5_box)
    plt.text(0.7, 0.85, 'V5 Fractal Echo\nVisualization System', 
            fontsize=12, ha='center', fontweight='bold')
    
    # Language Memory System
    lang_box = patches.Rectangle((0.05, 0.65), 0.45, 0.2, 
                               fill=True, alpha=0.1, 
                               color='purple', zorder=-1)
    plt.gca().add_patch(lang_box)
    plt.text(0.25, 0.85, 'Language Memory System', 
            fontsize=12, ha='center', fontweight='bold')
    
    # Version Transition Layer
    transition_box = patches.Rectangle((0.25, 0.25), 0.5, 0.1, 
                                     fill=True, alpha=0.1, 
                                     color='orange', zorder=-1)
    plt.gca().add_patch(transition_box)
    plt.text(0.5, 0.36, 'Version Transition Layer', 
            fontsize=12, ha='center', fontweight='bold')
    
    # Add title
    plt.title('Integration Architecture Diagram', fontsize=16, fontweight='bold')
    
    # Remove axes
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('integration_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram created: integration_architecture.png")

def create_message_flow_diagram():
    """Create a message flow diagram using matplotlib."""
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add system nodes
    systems = [
        ('lang_mem', 'Language Memory System', 0.2, 0.5),
        ('bridge', 'Memory-V5 Bridge', 0.5, 0.5),
        ('v5_vis', 'V5 Visualization System', 0.8, 0.5)
    ]
    
    # Add message nodes
    messages = [
        ('msg1', 'Get Available Topics', 0.35, 0.8),
        ('msg2', 'Search Memories', 0.35, 0.65),
        ('msg3', 'Generate Fractal Pattern', 0.65, 0.8),
        ('msg4', 'Store Memory', 0.65, 0.65),
        ('msg5', 'Get Node Consciousness Data', 0.65, 0.35)
    ]
    
    # Add nodes to graph
    for node_id, label, _, _ in systems + messages:
        G.add_node(node_id, label=label)
    
    # Add edges
    edges = [
        ('lang_mem', 'msg1', 'sends'),
        ('msg1', 'bridge', 'Request'),
        ('bridge', 'lang_mem', 'API Call'),
        
        ('lang_mem', 'msg2', 'sends'),
        ('msg2', 'bridge', 'Request'),
        
        ('bridge', 'msg3', 'processes'),
        ('msg3', 'v5_vis', 'Generate'),
        
        ('v5_vis', 'msg4', 'sends'),
        ('msg4', 'bridge', 'Store'),
        ('bridge', 'lang_mem', 'API Call'),
        
        ('v5_vis', 'msg5', 'sends'),
        ('msg5', 'bridge', 'Request')
    ]
    
    for src, dest, label in edges:
        G.add_edge(src, dest, label=label)
    
    # Define positions
    pos = {node_id: (x, y) for node_id, _, x, y in systems + messages}
    
    # Draw system nodes
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[n for n, _, _, _ in systems],
                          node_color=['lavender', 'lightyellow', 'lightblue'],
                          node_size=3000,
                          node_shape='s',
                          alpha=0.8)
    
    # Draw message nodes
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[n for n, _, _, _ in messages],
                          node_color='aliceblue',
                          node_size=2500,
                          node_shape='o',
                          alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=15,
                          width=1.5,
                          connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    node_labels = {node_id: label for node_id, label, _, _ in systems + messages}
    nx.draw_networkx_labels(G, pos,
                           labels=node_labels,
                           font_size=10,
                           font_family='sans-serif')
    
    # Draw edge labels
    edge_labels = {(u, v): data['label'] for u, v, data in G.edges(data=True) if 'label' in data}
    nx.draw_networkx_edge_labels(G, pos,
                                edge_labels=edge_labels,
                                font_size=8,
                                label_pos=0.3)
    
    # Add title
    plt.title('Message Flow Between Systems', fontsize=16, fontweight='bold')
    
    # Remove axes
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('message_flow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram created: message_flow.png")

def create_version_transition_diagram():
    """Create a version transition diagram using matplotlib."""
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add version nodes
    versions = [
        ('v1', 'Version 1-2\nText Interface', 0.2, 0.8),
        ('v3', 'Version 3-4\nBasic GUI', 0.2, 0.2),
        ('v5', 'Version 5\nFractal Echo Visualization', 0.6, 0.5),
        ('v10', 'Version 10\nFull Consciousness', 0.9, 0.5)
    ]
    
    # Add transition components
    components = [
        ('socket', 'Node Socket Architecture', 0.4, 0.5),
        ('bridge', 'Language Memory V5 Bridge', 0.5, 0.5),
        ('compat', 'Compatibility Layer', 0.4, 0.35)
    ]
    
    # Add shared nodes
    shared = [
        ('shared1', 'Shared Memory Format', 0.3, 0.6),
        ('shared2', 'State Preservation', 0.35, 0.5),
        ('shared3', 'Command Parity', 0.3, 0.4)
    ]
    
    # Add nodes to graph
    for node_id, label, _, _ in versions + components + shared:
        G.add_node(node_id, label=label)
    
    # Add edges
    edges = [
        ('v1', 'socket', 'Socket Connection'),
        ('v3', 'socket', 'Socket Connection'),
        ('socket', 'compat', 'Data Translation'),
        ('compat', 'bridge', 'Unified Interface'),
        ('bridge', 'v5', 'Integration'),
        ('v5', 'v10', 'Evolution'),
        
        ('socket', 'shared1', ''),
        ('socket', 'shared2', ''),
        ('socket', 'shared3', '')
    ]
    
    for src, dest, label in edges:
        G.add_edge(src, dest, label=label)
    
    # Define positions
    pos = {node_id: (x, y) for node_id, _, x, y in versions + components + shared}
    
    # Draw version nodes
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[n for n, _, _, _ in versions],
                          node_color=['gold', 'lightsalmon', 'lightskyblue', 'lightgreen'],
                          node_size=2500,
                          node_shape='s',
                          alpha=0.8)
    
    # Draw component nodes
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[n for n, _, _, _ in components],
                          node_color=['aliceblue', 'lightyellow', 'beige'],
                          node_size=2000,
                          node_shape='s',
                          alpha=0.8)
    
    # Draw shared nodes
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[n for n, _, _, _ in shared],
                          node_color='lavender',
                          node_size=1500,
                          node_shape='o',
                          alpha=0.8)
    
    # Draw main edges
    solid_edges = [(u, v) for u, v, label in edges if label]
    nx.draw_networkx_edges(G, pos,
                          edgelist=solid_edges,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=15,
                          width=1.5)
    
    # Draw dashed edges
    dashed_edges = [(u, v) for u, v, label in edges if not label]
    nx.draw_networkx_edges(G, pos,
                          edgelist=dashed_edges,
                          edge_color='gray',
                          style='dashed',
                          arrows=True,
                          arrowsize=10,
                          width=1.0)
    
    # Draw labels
    node_labels = {node_id: label for node_id, label, _, _ in versions + components + shared}
    nx.draw_networkx_labels(G, pos,
                           labels=node_labels,
                           font_size=9,
                           font_family='sans-serif')
    
    # Draw edge labels
    edge_labels = {(u, v): label for u, v, label in edges if label}
    nx.draw_networkx_edge_labels(G, pos,
                                edge_labels=edge_labels,
                                font_size=8)
    
    # Add title
    plt.title('Version Transition Architecture', fontsize=16, fontweight='bold')
    
    # Remove axes
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('version_transition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram created: version_transition.png")

if __name__ == "__main__":
    try:
        print("Generating Integration Architecture Diagram...")
        create_integration_diagram()
        
        print("\nGenerating Message Flow Diagram...")
        create_message_flow_diagram()
        
        print("\nGenerating Version Transition Diagram...")
        create_version_transition_diagram()
        
        print("\nAll diagrams generated successfully!")
    except Exception as e:
        print(f"Error generating diagrams: {e}") 