import asyncio
from PySide6.QtWidgets import QApplication
from .node_manager import NodeManager
from .wiki_processor_node import WikiProcessorNode

async def main():
    """Example of using the node system with WikiProcessorNode"""
    
    # Create node manager
    manager = NodeManager()
    
    # Register node types
    manager.register_node_type("wiki_processor", WikiProcessorNode)
    
    # Create nodes
    wiki_node = manager.create_node("wiki_processor")
    if not wiki_node:
        print("Failed to create wiki processor node")
        return
        
    # Set up signal handlers
    def on_node_updated(node_id):
        node = manager.get_node(node_id)
        if node:
            # Print node outputs
            for port_name, port in node.output_ports.items():
                if port.value is not None:
                    print(f"{port_name}: {port.value}")
                    
    def on_error(error_msg):
        print(f"Error: {error_msg}")
        
    manager.node_updated.connect(on_node_updated)
    manager.error_occurred.connect(on_error)
    
    # Set input values
    wiki_node.input_ports["text"].value = "Python programming language"
    wiki_node.input_ports["auto_update"].value = True
    wiki_node.input_ports["update_interval"].value = 30
    
    # Execute nodes
    await manager.execute()
    
    # Wait for some updates
    await asyncio.sleep(60)
    
    # Clean up
    manager.stop()
    manager.clear()

if __name__ == "__main__":
    app = QApplication([])
    asyncio.run(main())
    app.exec() 