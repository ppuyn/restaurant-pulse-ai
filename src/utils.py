"""Utility functions for Restaurant Pulse AI"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json

from config import LOG_LEVEL, LOG_FORMAT

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def setup_logger(name: str) -> logging.Logger:
    """Setup a logger with the configured format."""
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(LOG_LEVEL)
    return logger_instance


def generate_date_range(
    start_date: datetime, days: int
) -> List[datetime]:
    """Generate a date range."""
    return [start_date + timedelta(days=i) for i in range(days)]


def add_time_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Add temporal features to dataframe."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    df["day_of_week"] = df[date_column].dt.dayofweek  # 0=Monday, 6=Sunday
    df["day_of_month"] = df[date_column].dt.day
    df["month"] = df[date_column].dt.month
    df["quarter"] = df[date_column].dt.quarter
    df["week_of_year"] = df[date_column].dt.isocalendar().week
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Cyclical encoding for day of week and month
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


def add_lag_features(
    df: pd.DataFrame, columns: List[str], lags: List[int]
) -> pd.DataFrame:
    """Add lag features to dataframe."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    # Note: Don't drop NaN immediately - let caller decide
    return df


def add_rolling_features(
    df: pd.DataFrame, columns: List[str], windows: List[int]
) -> pd.DataFrame:
    """Add rolling window statistics."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            for window in windows:
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window).mean()
                df[f"{col}_rolling_std_{window}"] = df[col].rolling(window).std()
    # Note: Don't drop NaN immediately - let caller decide
    return df


def normalize_features(
    X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features using z-score normalization."""
    # Ensure we're working with standard numpy arrays
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=np.float64)
    else:
        X = np.array(X, dtype=np.float64)
    
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    X_normalized = (X - mean) / std
    
    return X_normalized, mean, std


def inverse_normalize(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Inverse normalization to get original scale."""
    return X * std + mean


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency."""
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value:.{decimals}f}%"


def dict_to_json(data: Dict, indent: int = 2) -> str:
    """Convert dictionary to pretty JSON."""
    return json.dumps(data, indent=indent, default=str)


def save_dict_to_json(data: Dict, filepath: str) -> None:
    """Save dictionary to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved data to {filepath}")


def load_dict_from_json(filepath: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded data from {filepath}")
    return data


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""
    
    def __init__(self):
        self.results = {}
    
    def calculate_all(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """Calculate all metrics."""
        prefix_str = f"{prefix}_" if prefix else ""
        self.results[f"{prefix_str}mae"] = calculate_mae(y_true, y_pred)
        self.results[f"{prefix_str}rmse"] = calculate_rmse(y_true, y_pred)
        self.results[f"{prefix_str}mape"] = calculate_mape(y_true, y_pred)
        return self.results
    
    def get_summary(self) -> Dict[str, float]:
        """Get metrics summary."""
        return self.results
    
    def print_summary(self) -> None:
        """Print metrics summary."""
        logger.info("=== Metrics Summary ===")
        for metric, value in self.results.items():
            logger.info(f"{metric}: {value:.4f}")
