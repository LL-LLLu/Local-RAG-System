# logger_config.py
import logging
from datetime import datetime
from pathlib import Path  # This is what Path refers to
import sys

def setup_logger(name='rag_system', log_level=logging.INFO):
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
    
    Returns:
        Logger instance
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)  # Creates 'logs' directory if it doesn't exist
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # File handler - logs everything
    log_file = log_dir / f'rag_{datetime.now().strftime("%Y%m%d")}.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler - only warnings and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Example usage
if __name__ == "__main__":
    # Get logger
    logger = setup_logger()
    
    # Use it
    logger.debug("This is a debug message - only in file")
    logger.info("This is an info message - only in file")
    logger.warning("This is a warning - in console and file")
    logger.error("This is an error - in console and file")
    
    try:
        1/0
    except Exception as e:
        logger.exception("An error occurred:")  # Logs with traceback