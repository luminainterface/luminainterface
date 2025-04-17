"""
WikiProcessorNode for LUMINA v7.5
Handles Wikipedia content processing in the node system
"""

import asyncio
import logging
import wikipedia
from datetime import datetime
from typing import List, Optional, Dict, Any
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import nltk
from .base_node import Node, NodeMetadata, NodeType

logger = logging.getLogger(__name__)

class WikiProcessorNode(Node):
    """Node for processing and retrieving Wikipedia content"""
    
    def __init__(self):
        metadata = NodeMetadata(
            name="Wikipedia Processor",
            description="Processes text input to find and fetch relevant Wikipedia content",
            category="Content",
            type=NodeType.PROCESSOR,
            color="#4B9CD3",  # Blue color for information processing
            icon="🌐"  # Globe icon for web content
        )
        super().__init__(metadata)
        
        # Initialize NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Add input ports
        self.add_input_port("text", str, "Input text to process")
        self.add_input_port("auto_update", bool, "Enable/disable auto-updates")
        self.add_input_port("update_interval", int, "Update interval in seconds")
        
        # Add output ports
        self.add_output_port("wiki_content", dict, "Processed Wikipedia content")
        self.add_output_port("keywords", list, "Extracted keywords")
        self.add_output_port("status", str, "Current processing status")
        self.add_output_port("error", str, "Error messages if any")
        
        # Initialize state
        self._last_search = None
        self._processing = False
        self._update_task: Optional[asyncio.Task] = None
        
    async def process(self) -> None:
        """Process input text and find relevant Wikipedia content"""
        try:
            # Get input values
            text = self.get_input_value("text")
            auto_update = self.get_input_value("auto_update")
            update_interval = self.get_input_value("update_interval")
            
            if not text:
                self.set_output_value("status", "Waiting for input")
                return
                
            # Update status
            self.set_output_value("status", "Processing")
            self._processing = True
            
            # Extract keywords and search Wikipedia
            keywords = self._extract_keywords(text)
            
            for keyword in keywords:
                if keyword == self._last_search:
                    continue
                    
                try:
                    # Search Wikipedia
                    summary = wikipedia.summary(keyword, sentences=2)
                    self._last_search = keyword
                    
                    # Format output
                    wiki_content = (
                        f"📚 Related to '{keyword}':\n"
                        f"{summary}\n"
                        f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
                    )
                    
                    # Set outputs
                    self.set_output_value("wiki_content", wiki_content)
                    self.set_output_value("status", "Complete")
                    self.set_output_value("error", None)
                    break
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    continue
                except wikipedia.exceptions.PageError:
                    continue
                except Exception as e:
                    self.set_output_value("error", f"Error processing '{keyword}': {str(e)}")
            
            # Handle auto-updates
            if auto_update and update_interval:
                if self._update_task:
                    self._update_task.cancel()
                self._update_task = asyncio.create_task(self._auto_update(update_interval))
                
        except Exception as e:
            self.set_output_value("error", f"Processing error: {str(e)}")
            self.set_output_value("status", "Error")
            logger.error(f"Error in WikiProcessorNode: {e}")
        finally:
            self._processing = False
            
    async def _auto_update(self, interval: int):
        """Automatically update wiki content at specified intervals"""
        while True:
            try:
                await asyncio.sleep(interval)
                if not self._processing:
                    await self.process()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-update: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from input text"""
        words = text.split()
        # Filter out common words and short words
        keywords = [word for word in words if len(word) > 4 and word.lower() not in {
            'about', 'above', 'after', 'again', 'their', 'would', 'could',
            'should', 'which', 'there', 'where', 'when', 'what', 'have'
        }]
        return keywords[:3]  # Return top 3 potential keywords
        
    def cleanup(self):
        """Clean up resources when node is deleted"""
        if self._update_task:
            self._update_task.cancel()
            self._update_task = None 