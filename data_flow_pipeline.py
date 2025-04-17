import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Callable
import uuid

logger = logging.getLogger(__name__)

class DataProcessor:
    """Base data processor component for pipeline stages"""
    
    def __init__(self, name: str = None):
        self.name = name or f"processor_{uuid.uuid4().hex[:6]}"
        self.metrics = {
            "processed_count": 0,
            "error_count": 0,
            "avg_process_time": 0.0,
            "total_process_time": 0.0
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through this processor
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processed data dictionary
        """
        start_time = time.time()
        self.metrics["processed_count"] += 1
        
        try:
            result = self._process_implementation(data)
            
            # Add processor metadata
            if "_metadata" not in result:
                result["_metadata"] = {}
            
            result["_metadata"][f"processed_by_{self.name}"] = True
            result["_metadata"][f"{self.name}_timestamp"] = time.time()
            
        except Exception as e:
            logger.error(f"Error in {self.name}.process(): {str(e)}")
            self.metrics["error_count"] += 1
            
            # Create error result while preserving original data
            result = data.copy() if isinstance(data, dict) else {"raw_input": data}
            
            if "_metadata" not in result:
                result["_metadata"] = {}
                
            result["_metadata"][f"{self.name}_error"] = str(e)
            result["_metadata"][f"{self.name}_status"] = "error"
        
        # Update processing time metrics
        process_time = time.time() - start_time
        self.metrics["total_process_time"] += process_time
        self.metrics["avg_process_time"] = (
            self.metrics["total_process_time"] / self.metrics["processed_count"]
        )
        
        return result
    
    def _process_implementation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of processing logic (to be overridden by subclasses)
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processed data dictionary
        """
        logger.warning(f"Using default _process_implementation in {self.name}")
        return data


class InputParser(DataProcessor):
    """Parse raw input into structured format"""
    
    def __init__(self):
        super().__init__(name="input_parser")
    
    def _process_implementation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data into structured format
        
        Args:
            data: Input data, either raw string or dictionary
            
        Returns:
            Structured data dictionary
        """
        result = {}
        
        # Handle different input types
        if isinstance(data, str):
            # Process raw text input
            result["text"] = data
            result["type"] = "text"
            
            # Extract special parameters with format :parameter:value
            special_params = {}
            text_parts = []
            
            for part in data.split():
                if part.startswith(":") and ":" in part[1:]:
                    param_name, param_value = part[1:].split(":", 1)
                    special_params[param_name] = param_value
                else:
                    text_parts.append(part)
            
            # Update text if parameters were extracted
            if special_params:
                result["text"] = " ".join(text_parts)
                result["parameters"] = special_params
                
                # Extract common parameters
                if "symbol" in special_params:
                    result["symbol"] = special_params["symbol"]
                if "emotion" in special_params:
                    result["emotion"] = special_params["emotion"]
                if "breath" in special_params:
                    result["breath"] = special_params["breath"]
                if "paradox" in special_params:
                    result["paradox"] = special_params["paradox"]
                    
        elif isinstance(data, dict):
            # Copy dictionary data
            result = data.copy()
            result["type"] = data.get("type", "dict")
            
        else:
            # Handle other data types
            result["raw"] = str(data)
            result["type"] = type(data).__name__
        
        # Add metadata
        result["_metadata"] = {
            "timestamp": time.time(),
            "id": str(uuid.uuid4())
        }
        
        return result


class OutputFormatter(DataProcessor):
    """Format processed data for consistent output"""
    
    def __init__(self):
        super().__init__(name="output_formatter")
    
    def _process_implementation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format data for consistent output
        
        Args:
            data: Input data dictionary
            
        Returns:
            Formatted output dictionary
        """
        result = {
            "status": "success",
            "response_id": str(uuid.uuid4()),
            "timestamp": time.time()
        }
        
        # Extract and standardize core output fields
        if "output" in data:
            result["output"] = data["output"]
        elif "response" in data:
            result["output"] = data["response"]
        elif "result" in data:
            result["output"] = data["result"]
        elif "text" in data:
            result["output"] = data["text"]
            
        # Add standard response fields if provided
        for field in ["action", "glyph", "story", "signal"]:
            if field in data:
                result[field] = data[field]
        
        # Add emotion field if available
        if "emotion" in data:
            result["emotion"] = data["emotion"]
            
        # Include parameters if available
        if "parameters" in data:
            result["parameters"] = data["parameters"]
            
        # Include metadata
        result["_metadata"] = data.get("_metadata", {})
        result["_metadata"]["output_formatted"] = True
        
        return result


class DataFlowPipeline:
    """
    Simplified data flow pipeline for Lumina v1
    
    This pipeline processes data through a series of stages:
    1. Input parsing (converts raw input to structured format)
    2. User-defined processing stages (main data transformation)
    3. Output formatting (standardizes output format)
    """
    
    def __init__(self):
        # Initialize standard pipeline components
        self.input_parser = InputParser()
        self.output_formatter = OutputFormatter()
        
        # Processing stages between input and output
        self.processing_stages: List[DataProcessor] = []
        
        # Pipeline metadata
        self.pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"
        self.metrics = {
            "processed_count": 0,
            "error_count": 0,
            "avg_process_time": 0.0,
            "total_process_time": 0.0
        }
        
        logger.info(f"Initialized DataFlowPipeline (id: {self.pipeline_id})")
    
    def add_processor(self, processor: DataProcessor) -> None:
        """
        Add a processor to the pipeline
        
        Args:
            processor: DataProcessor to add to the pipeline
        """
        self.processing_stages.append(processor)
        logger.info(f"Added processor '{processor.name}' to pipeline")
    
    def add_processing_function(self, func: Callable, name: str = None) -> None:
        """
        Add a processing function to the pipeline
        
        Args:
            func: Function that takes a dict and returns a dict
            name: Optional name for the function
        """
        # Create a wrapper processor for the function
        class FunctionProcessor(DataProcessor):
            def __init__(self, func, name):
                super().__init__(name=name or func.__name__)
                self.func = func
            
            def _process_implementation(self, data):
                return self.func(data)
        
        # Add the processor to the pipeline
        self.add_processor(FunctionProcessor(func, name))
    
    def process(self, data: Any) -> Dict[str, Any]:
        """
        Process data through the entire pipeline
        
        Args:
            data: Input data in any format
            
        Returns:
            Processed and formatted output dictionary
        """
        start_time = time.time()
        self.metrics["processed_count"] += 1
        
        try:
            # Step 1: Parse input
            current_data = self.input_parser.process(data)
            
            # Step 2: Process through all stages
            for processor in self.processing_stages:
                current_data = processor.process(current_data)
            
            # Step 3: Format output
            result = self.output_formatter.process(current_data)
            
            # Add pipeline metadata
            if "_metadata" not in result:
                result["_metadata"] = {}
                
            result["_metadata"]["pipeline_id"] = self.pipeline_id
            result["_metadata"]["pipeline_stages"] = [p.name for p in self.processing_stages]
            result["_metadata"]["pipeline_timestamp"] = time.time()
            
        except Exception as e:
            logger.error(f"Unhandled error in pipeline: {str(e)}")
            self.metrics["error_count"] += 1
            
            # Create error result
            result = {
                "status": "error",
                "error": str(e),
                "response_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "_metadata": {
                    "pipeline_id": self.pipeline_id,
                    "pipeline_error": True
                }
            }
        
        # Update processing time metrics
        process_time = time.time() - start_time
        self.metrics["total_process_time"] += process_time
        self.metrics["avg_process_time"] = (
            self.metrics["total_process_time"] / self.metrics["processed_count"]
        )
        
        # Log pipeline completion
        logger.debug(f"Pipeline completed in {process_time:.4f}s")
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get pipeline metrics
        
        Returns:
            Dictionary of pipeline metrics
        """
        metrics = self.metrics.copy()
        
        # Add component metrics
        metrics["components"] = {
            "input_parser": self.input_parser.metrics,
            "output_formatter": self.output_formatter.metrics
        }
        
        for i, processor in enumerate(self.processing_stages):
            metrics["components"][f"stage_{i}_{processor.name}"] = processor.metrics
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset all pipeline metrics"""
        self.metrics = {
            "processed_count": 0,
            "error_count": 0,
            "avg_process_time": 0.0,
            "total_process_time": 0.0
        }
        
        # Reset component metrics
        self.input_parser.metrics = {
            "processed_count": 0,
            "error_count": 0,
            "avg_process_time": 0.0,
            "total_process_time": 0.0
        }
        
        self.output_formatter.metrics = {
            "processed_count": 0,
            "error_count": 0,
            "avg_process_time": 0.0,
            "total_process_time": 0.0
        }
        
        for processor in self.processing_stages:
            processor.metrics = {
                "processed_count": 0,
                "error_count": 0,
                "avg_process_time": 0.0,
                "total_process_time": 0.0
            }
        
        logger.info("Reset all pipeline metrics") 