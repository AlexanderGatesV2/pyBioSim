import logging
import logging.handlers
import os
import sys
from typing import Dict, Optional, Union
from utils.custom_handlers import CustomRotatingFileHandler

# Dictionary to track configured loggers
_configured_loggers: Dict[str, logging.Logger] = {}

# Default formatter
_default_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_logger(module_name: str, log_file: Optional[str] = None, 
               console_level: int = logging.INFO, 
               file_level: int = logging.DEBUG) -> logging.Logger:
    """
    Get or create a logger for a specific module with proper configuration.
    
    Args:
        module_name: Name of the module requesting the logger
        log_file: Optional path to log file. If None, only console logging is used
        console_level: Logging level for console output
        file_level: Logging level for file output
        
    Returns:
        Configured logger instance
    """
    # Check if logger already configured
    if module_name in _configured_loggers:
        return _configured_loggers[module_name]
    
    # Create new logger
    logger = logging.getLogger(module_name)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    try:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(_default_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if log file specified)
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Use our CustomRotatingFileHandler for log rotation
            file_handler = CustomRotatingFileHandler(
                log_file,
                mode='a',
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(_default_formatter)
            logger.addHandler(file_handler)
        
        # Set logger level to minimum of handlers
        logger.setLevel(min(console_level, file_level if log_file else logging.CRITICAL))
        
        # Store configured logger
        _configured_loggers[module_name] = logger
        
        return logger
        
    except Exception as e:
        # Fallback to basic console logger in case of error
        print(f"Failed to configure logger for {module_name}: {e}")
        basic_logger = logging.getLogger(module_name)
        basic_logger.setLevel(logging.INFO)
        
        if not basic_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(_default_formatter)
            basic_logger.addHandler(handler)
            
        return basic_logger

def configure_logging(log_file: Optional[str] = None, 
                     console_level: int = logging.CRITICAL, 
                     file_level: int = logging.CRITICAL) -> None:
    """
    Configure the root logger (backward compatibility function).
    
    Args:
        log_file: Optional path to log file
        console_level: Logging level for console output
        file_level: Logging level for file output
    """
    try:
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(_default_formatter)
        logging.root.addHandler(console_handler)

        # File handler (if log file specified)
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Use CustomRotatingFileHandler for consistency with module loggers
            file_handler = CustomRotatingFileHandler(
                log_file,
                mode='a',
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(_default_formatter)
            logging.root.addHandler(file_handler)

        # Set root logger level to the minimum of console and file levels
        logging.root.setLevel(min(console_level, file_level))

    except Exception as e:
        print(f"Failed to configure logging: {e}")

def set_logging_level(level: Union[str, int], module_name: Optional[str] = None) -> None:
    """
    Set the logging level for a specific module or the root logger.
    
    Args:
        level: Logging level (can be string like 'DEBUG' or int constant)
        module_name: Optional module name. If None, sets level for root logger
    """
    try:
        # Convert string level to int if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper())
            
        # Set level for specific module or root
        if module_name:
            logging.getLogger(module_name).setLevel(level)
        else:
            logging.root.setLevel(level)
            
    except (AttributeError, TypeError) as e:
        print(f"Failed to set logging level: {e}")

def setup_module_logger(module_name: str, log_file: Optional[str] = None,
                        console_level: int = logging.INFO,
                        file_level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up a logger for a specific module with default settings.
    
    Args:
        module_name: Name of the module
        log_file: Optional specific log file for this module
        console_level: Logging level for console output
        file_level: Logging level for file output
        
    Returns:
        Configured logger
    """
    return get_logger(module_name, log_file, console_level, file_level)
