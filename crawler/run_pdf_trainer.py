import os
import asyncio
import yaml
from pathlib import Path
from crawler.pdf_trainer import PDFTrainer
import logging
from shared.log_config import setup_logging

logger = setup_logging('pdf-trainer')

async def main():
    revectorize = os.getenv("REVECTORIZE", "0") == "1"
    try:
        # Load config
        config_path = Path("/app/config/pdf_trainer.yml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        logger.info("Loaded configuration successfully")
        
        # Initialize and run trainer
        trainer = PDFTrainer(config, revectorize=revectorize)
        logger.info("Initialized PDF trainer")
        await trainer._init_qdrant_collection()
        # Run the trainer
        await trainer.process_all_pdfs()
        
    except Exception as e:
        logger.error(f"Error in PDF trainer: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 