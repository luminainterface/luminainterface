"""Crawler service for fetching and processing content from multiple sources."""

import os
import asyncio
from log import logger

class Crawler:
    def __init__(self, pdf_processor, redis_client, vector_store, embedding_model, pdf_training_path, cache_ttl, PDF_PRIORITY):
        self.pdf_processor = pdf_processor
        self.redis_client = redis_client
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.pdf_training_path = pdf_training_path
        self.cache_ttl = cache_ttl
        self.PDF_PRIORITY = PDF_PRIORITY

    async def _process_pdf_content(self):
        """Process PDF content with medium priority."""
        try:
            for file in os.listdir(self.pdf_training_path):
                if file.endswith('.pdf'):
                    file_path = os.path.join(self.pdf_training_path, file)
                    logger.info(f"Processing PDF file: {file}")
                    
                    try:
                        # Process the PDF using async method
                        pdf_data = await self.pdf_processor.process_pdf(file_path)
                        
                        # Store the extracted text in Redis for caching
                        cache_key = f"pdf:{os.path.basename(file_path)}"
                        await self.redis_client.set_cache(
                            cache_key,
                            pdf_data,
                            self.cache_ttl
                        )
                        
                        # Add concepts to crawl queue
                        for concept in pdf_data.get('concepts', []):
                            await self.redis_client.add_to_crawl_queue(
                                concept=concept,
                                weight=self.PDF_PRIORITY,
                                source="pdf",
                                metadata={
                                    "pdf_title": pdf_data['metadata'].get('title'),
                                    "pdf_author": pdf_data['metadata'].get('author'),
                                    "pdf_path": pdf_data['metadata'].get('file_path')
                                }
                            )
                        
                        # Store the PDF text in vector store
                        if pdf_data.get('text'):
                            # Run CPU-bound embedding generation in thread pool
                            loop = asyncio.get_event_loop()
                            text_embedding = await loop.run_in_executor(
                                None,
                                self.embedding_model.encode,
                                pdf_data['text']
                            )
                            
                            await self.vector_store.upsert_vectors(
                                vectors=[text_embedding],
                                metadata=[{
                                    "title": pdf_data['metadata'].get('title'),
                                    "type": "pdf",
                                    "author": pdf_data['metadata'].get('author'),
                                    "file_path": pdf_data['metadata'].get('file_path'),
                                    "processed_at": pdf_data.get('processed_at')
                                }],
                                ids=[f"pdf_{os.path.basename(file_path)}"]
                            )
                        
                        logger.info(f"Successfully processed PDF: {file}")
                        
                    except Exception as e:
                        logger.error(f"Error processing PDF file {file}: {str(e)}")
                        continue
                    
        except Exception as e:
 