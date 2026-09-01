"""Process-wide logger writing to ``logs/`` and to stderr."""

import logging
import os
from datetime import datetime


def setup_logger():
    """Return a timestamped logger writing to file and console."""
    if len(logging.getLogger().handlers) > 0:
        return logging.getLogger(__name__)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs("./logs", exist_ok=True)
    log_file = os.path.join("./logs", f"run_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),  # Log to a new file for each run
            logging.StreamHandler(),  # Also log to console
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")

    return logger
