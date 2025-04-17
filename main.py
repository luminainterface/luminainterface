from src.central_node import CentralNode

def main():
    # Create and initialize central node
    central_node = CentralNode()
    
    # Print available components
    print("\nAvailable Components:")
    for category, components in central_node.list_available_components().items():
        print(f"\n{category.capitalize()}:")
        for component in components:
            print(f"  - {component}")
    
    # Print system status
    print("\nSystem Status:")
    for key, value in central_node.get_system_status().items():
        print(f"{key}: {value}")
            
    # Test the flow pipeline with sample data
    print("\nTesting Flow Pipeline:")
    input_data = {
        'symbol': 'infinity',
        'emotion': 'wonder',
        'breath': 'deep',
        'paradox': 'existence'
    }
    output = central_node.process_complete_flow(input_data)
    print("\nOutput:")
    for key, value in output.items():
        print(f"  - {key}: {value}")

if __name__ == "__main__":
    main() 