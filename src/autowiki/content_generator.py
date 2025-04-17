import logging
from datetime import datetime
from queue import Queue
import random

class ContentGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.queue = Queue()
        self.templates = {
            'article': {
                'intro': [
                    "This article explores {topic} in detail.",
                    "An overview of {topic} and its applications.",
                    "Understanding {topic} and its importance."
                ],
                'body': [
                    "Key aspects of {topic} include {aspects}.",
                    "The main components of {topic} are {components}.",
                    "Important considerations for {topic} involve {considerations}."
                ],
                'conclusion': [
                    "In conclusion, {topic} plays a vital role in {field}.",
                    "The future of {topic} holds great promise for {field}.",
                    "Continued research in {topic} will advance {field}."
                ]
            }
        }
        
    def initialize(self):
        """Initialize content generator"""
        try:
            while not self.queue.empty():
                self.queue.get()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize content generator: {str(e)}")
            return False
            
    def generate(self, request):
        """Generate content based on request"""
        try:
            if not request.get('topic'):
                return None
                
            content = {
                'title': request['topic'],
                'sections': []
            }
            
            # Generate introduction
            intro = random.choice(self.templates['article']['intro'])
            content['sections'].append({
                'type': 'introduction',
                'content': intro.format(topic=request['topic'])
            })
            
            # Generate body sections
            if request.get('aspects'):
                body = random.choice(self.templates['article']['body'])
                content['sections'].append({
                    'type': 'body',
                    'content': body.format(
                        topic=request['topic'],
                        aspects=', '.join(request['aspects']),
                        components=', '.join(request.get('components', [])),
                        considerations=', '.join(request.get('considerations', []))
                    )
                })
                
            # Generate conclusion
            conclusion = random.choice(self.templates['article']['conclusion'])
            content['sections'].append({
                'type': 'conclusion',
                'content': conclusion.format(
                    topic=request['topic'],
                    field=request.get('field', 'the field')
                )
            })
            
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to generate content: {str(e)}")
            return None
            
    def add_to_queue(self, request):
        """Add content generation request to queue"""
        try:
            self.queue.put(request)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add request to queue: {str(e)}")
            return False
            
    def get_queue(self):
        """Get content generation queue"""
        return list(self.queue.queue)
        
    def process_queue(self):
        """Process all requests in the queue"""
        try:
            results = []
            while not self.queue.empty():
                request = self.queue.get()
                content = self.generate(request)
                if content:
                    results.append(content)
            return results
        except Exception as e:
            self.logger.error(f"Failed to process queue: {str(e)}")
            return [] 