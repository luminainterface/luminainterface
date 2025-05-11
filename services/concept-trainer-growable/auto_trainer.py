import time
import os
import sys
import logging
sys.path.append(os.path.dirname(__file__))
from train_demo import retrain_model, save_extended_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler("auto_trainer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("auto_trainer")

POLL_INTERVAL = 30  # seconds
EPOCHS_PER_RUN = 100

if __name__ == "__main__":
    logger.info(f"Running a new {EPOCHS_PER_RUN}-epoch training session every {POLL_INTERVAL} seconds...")
    while True:
        logger.info("Starting scheduled training session...")
        retrain_model(max_epochs=EPOCHS_PER_RUN)
        save_extended_dataset()
        logger.info("Training session complete. Sleeping for %d seconds.", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL) 