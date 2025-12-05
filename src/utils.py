#!/usr/bin/env python3
"""
Utilities Module
Shared utilities for logging, configuration, and file operations
"""

import yaml
import pickle
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import colorlog


def setup_logging(
    log_dir: str = "logs", log_level: str = "INFO", module_name: str = None
):
    """
    Setup logging with color output and file handlers

    Parameters:
    -----------
    log_dir : str
        Directory for log files
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    module_name : str
        Name for the log file (defaults to date)
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Log file name
    if module_name:
        log_file = (
            Path(log_dir) / f"{module_name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
    else:
        log_file = (
            Path(log_dir) / f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

    # Create formatter
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Color formatter for console
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt=date_format,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    # File formatter
    file_formatter = logging.Formatter(log_format, datefmt=date_format)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler (with color)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(color_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging initialized: {log_file}")

    return root_logger


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load YAML configuration file

    Parameters:
    -----------
    config_path : str
        Path to config file

    Returns:
    --------
    dict : Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.info(f"Configuration loaded from {config_path}")

    return config


def save_config(config: Dict, config_path: str = "config.yaml"):
    """
    Save configuration to YAML file

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_path : str
        Path to save config
    """
    config_path = Path(config_path)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logging.info(f"Configuration saved to {config_path}")


def save_checkpoint(obj: Any, path: str):
    """
    Save object to pickle file

    Parameters:
    -----------
    obj : any
        Object to save
    path : str
        Path to save file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    logging.info(f"Checkpoint saved: {path}")


def load_checkpoint(path: str) -> Any:
    """
    Load object from pickle file

    Parameters:
    -----------
    path : str
        Path to pickle file

    Returns:
    --------
    any : Loaded object
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    logging.info(f"Checkpoint loaded: {path}")

    return obj


def format_number(num: float, decimals: int = 2, percentage: bool = False) -> str:
    """
    Format number for display

    Parameters:
    -----------
    num : float
        Number to format
    decimals : int
        Decimal places
    percentage : bool
        Format as percentage

    Returns:
    --------
    str : Formatted string
    """
    if percentage:
        return f"{num*100:.{decimals}f}%"
    else:
        return f"{num:,.{decimals}f}"


def get_quarter_end(date_str: str) -> str:
    """
    Get quarter end date from any date

    Parameters:
    -----------
    date_str : str
        Date in YYYY-MM-DD format

    Returns:
    --------
    str : Quarter end date
    """
    import pandas as pd

    date = pd.to_datetime(date_str)
    qtr_end = date.to_period("Q").end_time
    return qtr_end.strftime("%Y-%m-%d")


def create_directory_structure(base_dir: str = "."):
    """
    Create standard directory structure

    Parameters:
    -----------
    base_dir : str
        Base directory
    """
    base = Path(base_dir)

    directories = [
        "data/13f_parquet",
        "data/cache",
        "data/models",
        "data/signals",
        "logs",
        "src",
        "reports",
    ]

    for dir_path in directories:
        (base / dir_path).mkdir(parents=True, exist_ok=True)

    logging.info(f"Directory structure created in {base}")


def validate_config(config: Dict) -> bool:
    """
    Validate configuration

    Parameters:
    -----------
    config : dict
        Configuration dictionary

    Returns:
    --------
    bool : True if valid
    """
    required_keys = [
        "data",
        "features",
        "models",
        "regimes",
        "portfolio",
        "trading",
        "broker",
        "risk",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")

    # Validate WRDS username
    if not config["data"]["wrds_username"]:
        raise ValueError("WRDS username not set in config")

    # Validate paths
    parquet_path = Path(config["data"]["paths"]["13f_parquet"])
    if not parquet_path.exists():
        raise FileNotFoundError(f"13F parquet directory not found: {parquet_path}")

    logging.info("Configuration validated successfully")

    return True


if __name__ == "__main__":
    # Test utilities
    print("Utilities Module - Loaded successfully")

    # Test logging
    setup_logging()
    logging.info("Test log message")
    logging.warning("Test warning message")
    logging.error("Test error message")
