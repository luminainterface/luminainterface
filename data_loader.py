import os
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Iterator
import pandas as pd

logger = logging.getLogger("DataLoader")

class DataLoader:
    """
    Handles loading and parsing data from multiple file formats:
    - txt
    - json
    - jsonl (JSON Lines)
    - csv
    """
    
    def __init__(self, data_dir: Union[str, Path] = "training_data"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def get_available_files(self, extensions: List[str] = None, recursive: bool = True) -> List[Path]:
        """
        Get all data files with the specified extensions
        
        Args:
            extensions: List of file extensions to search for (e.g. ['.txt', '.json'])
            recursive: Whether to search recursively through subdirectories
        """
        if extensions is None:
            extensions = ['.txt', '.json', '.jsonl', '.csv']
            
        files = []
        
        # Use recursive globbing if requested
        if recursive:
            for ext in extensions:
                # The ** pattern matches directories recursively
                files.extend(list(self.data_dir.glob(f"**/*{ext}")))
        else:
            # Just search the top directory
            for ext in extensions:
                files.extend(list(self.data_dir.glob(f"*{ext}")))
            
        logger.info(f"Found {len(files)} data files")
        
        # Log the discovered files for debugging
        for file in files:
            logger.debug(f"Found data file: {file}")
            
        return files
    
    def load_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load data from a file based on its extension
        
        Args:
            file_path: Path to the file to load
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Ensure the data is a list
                    if isinstance(data, dict):
                        data = [data]
                    return data
                    
            elif extension == '.jsonl':
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            data.append(json.loads(line))
                return data
                
            elif extension == '.csv':
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    return list(reader)
                    
            elif extension == '.txt':
                return self._parse_txt_file(file_path)
                
            else:
                logger.warning(f"Unsupported file extension: {extension} for file {file_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return []
            
    def _parse_txt_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Parse our custom text file format with colon-separated metadata headers
        
        Format:
            symbol:value emotion:value breath:value paradox:value
            Description text on following lines
            
            symbol:value emotion:value breath:value paradox:value
            Another description text
        """
        data = []
        current_item = None
        current_description = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines unless we're in the middle of a description
                if not line:
                    if current_item and current_description:
                        current_item['story'] = ' '.join(current_description)
                        data.append(current_item)
                        current_item = None
                        current_description = []
                    continue
                
                # Check if this is a header line with metadata
                if all(part in line for part in ['symbol:', 'emotion:', 'breath:', 'paradox:']):
                    # If we were processing an item, save it before starting a new one
                    if current_item and current_description:
                        current_item['story'] = ' '.join(current_description)
                        data.append(current_item)
                        
                    # Start a new item
                    current_item = {}
                    current_description = []
                    
                    # Parse the metadata
                    parts = line.split()
                    for part in parts:
                        if ':' in part:
                            key, value = part.split(':', 1)
                            current_item[key] = value
                            
                # If not a header and we have a current item, add to description
                elif current_item is not None:
                    current_description.append(line)
        
        # Don't forget the last item
        if current_item and current_description:
            current_item['story'] = ' '.join(current_description)
            data.append(current_item)
            
        return data
    
    def load_all_data(self, extensions: List[str] = None, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Load data from all files with the specified extensions
        
        Args:
            extensions: List of file extensions to load
            recursive: Whether to search recursively through subdirectories
        """
        files = self.get_available_files(extensions, recursive)
        all_data = []
        
        for file_path in files:
            file_data = self.load_file(file_path)
            # Add source file info to each record
            for i in range(len(file_data)):
                # Convert non-dictionary items to dictionaries
                if not isinstance(file_data[i], dict):
                    file_data[i] = {"value": file_data[i]}
                
                # Now it's safe to add the source file information
                file_data[i]['_source_file'] = str(file_path)
                
            all_data.extend(file_data)
            
        logger.info(f"Loaded {len(all_data)} data items total from {len(files)} files")
        return all_data
    
    def batch_iterator(self, data: List[Dict[str, Any]], batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """Iterate through data in batches"""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
            
    def create_subdirectory(self, subdir_name: str) -> Path:
        """Create a subdirectory within the data directory"""
        subdir_path = self.data_dir / subdir_name
        subdir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created subdirectory: {subdir_path}")
        return subdir_path
    
    def save_data(self, data: List[Dict[str, Any]], file_path: Union[str, Path], 
                 format: str = "json") -> bool:
        """Save data to a file in the specified format"""
        file_path = Path(file_path)
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            elif format.lower() == "jsonl":
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')
            elif format.lower() == "csv":
                if not data:
                    logger.warning("No data to save to CSV")
                    return False
                    
                with open(file_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
            elif format.lower() == "txt":
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        if isinstance(item, dict):
                            f.write(json.dumps(item) + '\n')
                        else:
                            f.write(str(item) + '\n')
            else:
                logger.error(f"Unsupported format: {format}")
                return False
                
            logger.info(f"Saved {len(data)} items to {format} file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {str(e)}")
            return False

    def format_all_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Standardize and format all data items to ensure they have consistent fields
        
        Args:
            data: List of data items to format
            
        Returns:
            List of formatted data items
        """
        formatted_data = []
        
        # Define required input fields
        required_fields = ['symbol', 'emotion', 'breath', 'paradox']
        
        for item in data:
            # Skip None items
            if item is None:
                continue
                
            # Create a new dict to hold formatted data
            formatted_item = {}
            
            # Copy over all fields
            for key, value in item.items():
                # Skip None keys or internal metadata fields
                if key is None or (isinstance(key, str) and key.startswith('_')):
                    continue
                
                # Convert string values like "0.92" to floats if appropriate
                if key == 'signal' and isinstance(value, str):
                    try:
                        formatted_item[key] = float(value)
                    except (ValueError, TypeError):
                        formatted_item[key] = value
                else:
                    formatted_item[key] = value
            
            # Ensure all required fields exist (set to None if missing)
            for field in required_fields:
                if field not in formatted_item:
                    formatted_item[field] = None
                    
            formatted_data.append(formatted_item)
            
        logger.info(f"Formatted {len(formatted_data)} data items")
        return formatted_data

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    
    # Create example subdirectories for testing
    loader.create_subdirectory("category_a")
    loader.create_subdirectory("category_b")
    
    # Test writing a file to a subdirectory
    sample_data = [
        {"symbol": "star", "emotion": "awe", "breath": "deep", "paradox": "cosmos"},
        {"symbol": "moon", "emotion": "calm", "breath": "gentle", "paradox": "cycles"}
    ]
    loader.save_data(sample_data, "training_data/category_a/example.json")
    
    # Find all files recursively
    files = loader.get_available_files(recursive=True)
    print(f"Available files (recursive search): {files}")
    
    # Load all data from all subdirectories
    all_data = loader.load_all_data(recursive=True)
    print(f"Total data items: {len(all_data)}")
    
    if all_data:
        print(f"Sample item: {all_data[0]}") 