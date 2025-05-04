import asyncio
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from lumina_core.memory.pruning import run_pruning_job

scheduler = AsyncIOScheduler()

# Default to 3 AM UTC if not specified
DEFAULT_CRON = "0 3 * * *"
PRUNE_CRON = os.getenv("PRUNE_CRON", DEFAULT_CRON)

async def run_scheduled_pruning():
    """Run the pruning job and log results."""
    try:
        logger.info("Starting scheduled pruning job")
        results = await run_pruning_job()
        logger.info(
            "Scheduled pruning complete",
            extra={
                "pruned": results["pruned"],
                "remaining": results["remaining"],
                "total_before": results["total_before"]
            }
        )
        # Update last run timestamp
        scheduler.last_run = asyncio.get_event_loop().time()
    except Exception as e:
        logger.error(f"Scheduled pruning failed: {e}")

def setup_scheduler():
    """Setup the scheduler with all jobs."""
    try:
        # Parse cron expression
        trigger = CronTrigger.from_crontab(PRUNE_CRON)
        logger.info(f"Setting up pruning job with schedule: {PRUNE_CRON}")
        
        # Add pruning job
        scheduler.add_job(
            run_scheduled_pruning,
            trigger=trigger,
            id="pruning_job",
            name="Daily vector pruning"
        )
        
        # Start the scheduler
        scheduler.start()
        logger.info("Scheduler started with pruning job")
        
    except ValueError as e:
        logger.error(f"Invalid cron expression '{PRUNE_CRON}': {e}")
        # Fall back to default
        trigger = CronTrigger.from_crontab(DEFAULT_CRON)
        scheduler.add_job(
            run_scheduled_pruning,
            trigger=trigger,
            id="pruning_job",
            name="Daily vector pruning (fallback)"
        )
        scheduler.start()
        logger.info(f"Scheduler started with fallback schedule: {DEFAULT_CRON}")

def shutdown_scheduler():
    """Shutdown the scheduler gracefully."""
    scheduler.shutdown()
    logger.info("Scheduler shut down") 