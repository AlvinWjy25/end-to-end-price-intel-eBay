import logging
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
LOG_DIR = ROOT_DIR / 'logs' / 'deployment'
LOG_DIR.mkdir(exist_ok=True, parents=True)

def setup_logger(run_type: str) -> logging.Logger:
    """
    Configures a logger to output to both console and a timestamped file.
    Resolves directories dynamically to prevent path issues.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = LOG_DIR / f"{timestamp}_{run_type}.log"

    logger = logging.getLogger(run_type)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_filename)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console Handler (Cleaner terminal output)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s]: %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger