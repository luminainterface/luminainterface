import logging

logger = logging.getLogger(__name__)

class LuminaTrainer:
    def __init__(self, model_path: str = "models/wiki_trained"):
        self.model_path = model_path
        logger.info(f"Mock LuminaTrainer initialized with model path: {model_path}")
        
    def train(self, training_file: str):
        """Mock training method"""
        logger.info(f"Mock training on file: {training_file}")
        return True 