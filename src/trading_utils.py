# trading_utils.py
import logging
import numpy as np

def setup_logger(name="trading_bot", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def calculate_reward(portfolio_value, initial_value):
    """Calculate reward as the percentage change from the initial value."""
    return (portfolio_value - initial_value) / initial_value * 100

def normalize_data(data):
    """Normalize a numpy array or list."""
    data = np.array(data)
    return (data - data.min()) / (data.max() - data.min() + 1e-8)
