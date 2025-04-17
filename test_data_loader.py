import logging
from data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_recursive_data_loading():
    """Test the data loader's ability to recursively load data from subdirectories"""
    print("\n===== Testing Recursive Data Loading =====")
    
    # Create data loader
    loader = DataLoader(data_dir="training_data")
    
    # Test 1: Find all available files (including in subdirectories)
    print("\nTest 1: Find all files recursively")
    files = loader.get_available_files(recursive=True)
    print(f"Found {len(files)} files in total")
    for i, file in enumerate(files):
        print(f"  {i+1}. {file}")
    
    # Test 2: Find files only in top directory (non-recursive)
    print("\nTest 2: Find files in top directory only")
    top_files = loader.get_available_files(recursive=False)
    print(f"Found {len(top_files)} files in top directory")
    for i, file in enumerate(top_files):
        print(f"  {i+1}. {file}")
    
    # Test 3: Load all data from all directories
    print("\nTest 3: Load all data from all directories")
    all_data = loader.load_all_data(recursive=True)
    print(f"Loaded {len(all_data)} data items from all directories")
    
    # Group data by source file for better understanding
    data_by_source = {}
    for item in all_data:
        if item is None:
            continue
        source = item.get('_source_file', 'unknown')
        if source not in data_by_source:
            data_by_source[source] = []
        data_by_source[source].append(item)
    
    # Print summary of data by source
    print("\nData by source file:")
    for source, items in data_by_source.items():
        print(f"  {source}: {len(items)} items")
    
    # Test 4: Load specific file types
    print("\nTest 4: Load specific file types")
    json_files = loader.get_available_files(extensions=['.json'], recursive=True)
    print(f"Found {len(json_files)} JSON files")
    
    csv_files = loader.get_available_files(extensions=['.csv'], recursive=True)
    print(f"Found {len(csv_files)} CSV files")
    
    jsonl_files = loader.get_available_files(extensions=['.jsonl'], recursive=True)
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Test 5: Format input data
    print("\nTest 5: Format all input data")
    # Filter out None items before formatting
    valid_data = [item for item in all_data if item is not None]
    formatted_data = loader.format_all_data(valid_data)
    
    # Count items with complete fields (all four input fields present)
    complete_items = sum(1 for item in formatted_data if all(k in item and item[k] is not None 
                       for k in ['symbol', 'emotion', 'breath', 'paradox']))
    
    print(f"Total items after formatting: {len(formatted_data)}")
    print(f"Items with complete input fields: {complete_items}")
    
    return all_data, formatted_data

if __name__ == "__main__":
    all_data, formatted_data = test_recursive_data_loading()
    
    # Print a few sample items
    print("\n===== Sample Data Items =====")
    for i, item in enumerate(all_data[:3]):
        if item is None:
            print(f"\nItem {i+1}: None")
            continue
        print(f"\nItem {i+1}:")
        for key, value in item.items():
            # Don't print long values
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            print(f"  {key}: {value}") 