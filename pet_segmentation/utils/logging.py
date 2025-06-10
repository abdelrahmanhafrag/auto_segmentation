"""
Enhanced logging configuration using structlog for structured logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from structlog.stdlib import LoggerFactory, add_log_level, add_logger_name

from ..config.settings import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: bool = False
) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        json_format: Whether to use JSON formatting
    """
    settings = get_settings()
    
    # Determine log level
    if log_level is None:
        log_level = settings.log_level.value
    
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure structlog
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_format or settings.debug:
        # JSON format for production or debugging
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Human-readable format for development
        shared_processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )
    
    structlog.configure(
        processors=shared_processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        
        if json_format:
            file_formatter = logging.Formatter('%(message)s')
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pydicom").setLevel(logging.INFO)


def get_logger(name: str = "pet_segmentation") -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls with parameters and execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        logger.debug(
            "Function called",
            function=func.__name__,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys())
        )
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(
                "Function completed",
                function=func.__name__,
                execution_time=f"{execution_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                "Function failed",
                function=func.__name__,
                error=str(e),
                execution_time=f"{execution_time:.3f}s"
            )
            raise
    
    return wrapper


def log_processing_progress(total_items: int, description: str = "Processing"):
    """
    Context manager for logging processing progress.
    
    Args:
        total_items: Total number of items to process
        description: Description of the processing task
    """
    from contextlib import contextmanager
    import time
    
    @contextmanager
    def progress_logger():
        logger = get_logger("progress")
        start_time = time.time()
        
        logger.info(
            "Processing started",
            description=description,
            total_items=total_items
        )
        
        try:
            yield logger
            
            execution_time = time.time() - start_time
            logger.info(
                "Processing completed",
                description=description,
                total_items=total_items,
                execution_time=f"{execution_time:.2f}s"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Processing failed",
                description=description,
                error=str(e),
                execution_time=f"{execution_time:.2f}s"
            )
            raise
    
    return progress_logger()


# Initialize logging on module import
try:
    setup_logging()
except Exception:
    # Fallback to basic logging if setup fails
    logging.basicConfig(level=logging.INFO) 