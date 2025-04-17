"""
Content Generator for AutoWiki System
Provides AI-powered content generation capabilities
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class ContentGenerator:
    def __init__(self):
        self.logger = logger
        self._initialized = False
        self.model = None
        self.tokenizer = None
        self.generation_history = {}
        self._initialize()
        
    def _initialize(self):
        """Initialize content generator"""
        try:
            # Initialize language model for content generation
            model_name = "gpt2"  # Using GPT-2 as base model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            self._initialized = True
            logger.info("Content generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize content generator: {str(e)}")
            raise
            
    def generate_content(self, prompt: str, max_length: int = 500, **kwargs) -> Dict[str, Any]:
        """Generate content based on prompt"""
        try:
            # Encode prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate content
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Store generation
            generation_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.generation_history[generation_id] = {
                'prompt': prompt,
                'generated_text': generated_text,
                'parameters': {
                    'max_length': max_length,
                    **kwargs
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'id': generation_id,
                'content': generated_text,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate content: {str(e)}")
            raise
            
    def expand_section(self, section_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Expand an existing section with additional content"""
        try:
            # Prepare prompt with context
            prompt = f"Original section:\n{section_text}\n\nExpand this section with additional relevant information."
            if context:
                prompt = f"Context: {context.get('topic', '')}\n{prompt}"
                
            # Generate expansion
            expansion = self.generate_content(
                prompt,
                max_length=len(section_text.split()) * 2,  # Double the original length
                temperature=0.8  # Slightly higher temperature for creativity
            )
            
            return {
                'original': section_text,
                'expansion': expansion['content'],
                'id': expansion['id']
            }
            
        except Exception as e:
            logger.error(f"Failed to expand section: {str(e)}")
            raise
            
    def generate_summary(self, content: str, max_length: int = 200) -> Dict[str, Any]:
        """Generate a summary of the content"""
        try:
            # Prepare prompt
            prompt = f"Summarize the following content:\n{content}\n\nSummary:"
            
            # Generate summary
            summary = self.generate_content(
                prompt,
                max_length=max_length,
                temperature=0.3  # Lower temperature for more focused summary
            )
            
            return {
                'original_length': len(content.split()),
                'summary_length': len(summary['content'].split()),
                'summary': summary['content'],
                'id': summary['id']
            }
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            raise
            
    def generate_outline(self, topic: str, sections: int = 5) -> Dict[str, Any]:
        """Generate an article outline"""
        try:
            # Prepare prompt
            prompt = f"Create a detailed outline for an article about {topic} with {sections} main sections."
            
            # Generate outline
            outline = self.generate_content(
                prompt,
                max_length=sections * 50,  # Approximate length per section
                temperature=0.6
            )
            
            return {
                'topic': topic,
                'outline': outline['content'],
                'id': outline['id']
            }
            
        except Exception as e:
            logger.error(f"Failed to generate outline: {str(e)}")
            raise
            
    def get_generation_history(self) -> List[Dict[str, Any]]:
        """Get history of content generations"""
        return [
            {
                'id': generation_id,
                **generation_data
            }
            for generation_id, generation_data in self.generation_history.items()
        ]
        
    def get_generation(self, generation_id: str) -> Optional[Dict[str, Any]]:
        """Get specific generation by ID"""
        return self.generation_history.get(generation_id)
        
    def clear_generation_history(self):
        """Clear generation history"""
        self.generation_history = {} 