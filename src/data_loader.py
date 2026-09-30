"""Data loading and generation module for Restaurant Pulse AI"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import logging

from config import (
    HISTORICAL_DATA_DAYS,
    FORECAST_HORIZON_DAYS,
    SAMPLE_RESTAURANTS,
    DATA_DIR,
)
from utils import setup_logger, generate_date_range, add_time_features

logger = setup_logger(__name__)


def generate_synthetic_sales_data(
    restaurant_id: str,
    days: int = HISTORICAL_DATA_DAYS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic sales data for a restaurant.
    
    Features:
    - Base demand with trend
    - Weekly seasonality (weekend boost)
    - Monthly seasonality
    - Random noise
    - Special events (spikes)
    """
    np.random.seed(seed)
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    dates = generate_date_range(start_date, days)
    
    # Base demand (customers per day)
    base_demand = 45
    
    # Trend component (gradual increase/decrease)
    trend = np.linspace(0, 15, days)
    
    # Weekly seasonality (stronger on weekends)
    day_of_week = np.array([d.weekday() for d in dates])
    weekly_seasonality = np.where(day_of_week >= 4, 1.3, 0.9)  # Weekend boost
    
    # Monthly seasonality
    month = np.array([d.month for d in dates])
    # Summer boost (June-August), holiday season (Dec-Jan)
    monthly_seasonality = np.where(np.isin(month, [6, 7, 8, 12, 1]), 1.2, 1.0)
    
    # Random noise
    noise = np.random.normal(1, 0.1, days)
    
    # Special events (random spikes - promotions, events)
    special_events = np.zeros(days)
    num_events = int(days * 0.05)  # 5% of days have events
    event_indices = np.random.choice(days, num_events, replace=False)
    special_events[event_indices] = np.random.uniform(1.3, 1.6, num_events)
    special_events = np.where(special_events == 0, 1.0, special_events)
    
    # Combine all components
    customers = base_demand * (1 + trend / 100) * weekly_seasonality * monthly_seasonality * noise * special_events
    customers = np.maximum(customers, 10).astype(int)
    
    # Revenue per customer (average check size)
    avg_check_sizes = {
        "rest_001": 35.50,
        "rest_002": 42.00,
        "rest_003": 28.75,
        "rest_004": 55.00,
        "rest_005": 32.00,
    }
    
    avg_check = avg_check_sizes.get(restaurant_id, 35.00)
    check_size_variation = np.random.normal(avg_check, avg_check * 0.1, days)
    check_size_variation = np.maximum(check_size_variation, 5)  # Minimum check size
    
    revenue = customers * check_size_variation
    
    df = pd.DataFrame({
        "restaurant_id": restaurant_id,
        "date": dates,
        "customers": customers,
        "avg_check_size": check_size_variation,
        "revenue": revenue,
        "is_weekend": (day_of_week >= 4).astype(int),
    })
    
    return df


def generate_synthetic_weather_data(
    days: int = HISTORICAL_DATA_DAYS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic weather data.
    Weather impacts restaurant demand (temperature, precipitation, etc.)
    """
    np.random.seed(seed)
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    dates = generate_date_range(start_date, days)
    
    # Temperature with seasonal variation
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    base_temp = 60
    seasonal_temp = 20 * np.sin(2 * np.pi * day_of_year / 365)
    temperature = base_temp + seasonal_temp + np.random.normal(0, 5, days)
    
    # Humidity
    humidity = 50 + 20 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 10, days)
    humidity = np.clip(humidity, 20, 95)
    
    # Precipitation probability (affects outdoor dining, mood)
    precipitation = np.random.choice([0, 1], days, p=[0.75, 0.25])
    
    # Wind speed
    wind_speed = np.abs(np.random.normal(8, 3, days))
    
    df = pd.DataFrame({
        "date": dates,
        "temperature": temperature,
        "humidity": humidity,
        "precipitation": precipitation,
        "wind_speed": wind_speed,
    })
    
    return df


def combine_datasets(
    sales_df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    """Combine sales and weather data."""
    # Convert both date columns to date only (removing time component)
    sales_df['date'] = pd.to_datetime(sales_df['date']).dt.normalize()
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.normalize()
    
    df = sales_df.merge(weather_df, on="date", how="left")
    
    # Forward fill and backward fill any NaN weather values (handle missing dates)
    weather_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed']
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    df = add_time_features(df, date_column="date")
    return df


def load_or_generate_data(
    restaurant_id: str = "rest_001",
    days: int = HISTORICAL_DATA_DAYS,
    regenerate: bool = False,
) -> pd.DataFrame:
    """Load existing data or generate new data."""
    data_file = DATA_DIR / f"{restaurant_id}_historical_data.csv"
    
    if data_file.exists() and not regenerate:
        logger.info(f"Loading existing data from {data_file}")
        df = pd.read_csv(data_file)
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    logger.info(f"Generating synthetic data for {restaurant_id}")
    sales_df = generate_synthetic_sales_data(restaurant_id, days=days)
    weather_df = generate_synthetic_weather_data(days=days)
    df = combine_datasets(sales_df, weather_df)
    
    # Save to file
    df.to_csv(data_file, index=False)
    logger.info(f"Saved data to {data_file}")
    
    return df


def prepare_features_labels(
    df: pd.DataFrame,
    target_column: str = "revenue",
    lag_days: List[int] = None,
    rolling_windows: List[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and labels for modeling.
    
    Args:
        df: Input dataframe
        target_column: Column to predict
        lag_days: Number of days to use for lag features
        rolling_windows: Window sizes for rolling statistics
    
    Returns:
        Features DataFrame and target Series
    """
    df = df.copy()
    
    # Drop NaN rows
    df = df.dropna()
    
    # Select features for modeling
    feature_cols = [
        col for col in df.columns
        if col not in ["restaurant_id", "date", target_column, "avg_check_size"]
    ]
    
    X = df[feature_cols].copy()
    y = df[target_column].copy()
    
    logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
    if feature_cols:
        logger.info(f"Features: {feature_cols[:5]}... (showing first 5)")
    
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_split: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets (time-based split)."""
    split_idx = int(len(X) * (1 - test_split))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def get_restaurant_info(restaurant_id: str) -> Dict:
    """Get restaurant information."""
    for rest in SAMPLE_RESTAURANTS:
        if rest["id"] == restaurant_id:
            return rest
    return {}


def get_all_restaurant_ids() -> List[str]:
    """Get all available restaurant IDs."""
    return [rest["id"] for rest in SAMPLE_RESTAURANTS]


class DataManager:
    """Manager for data operations."""
    
    def __init__(self, restaurant_id: str = "rest_001"):
        self.restaurant_id = restaurant_id
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self, regenerate: bool = False) -> pd.DataFrame:
        """Load data."""
        self.data = load_or_generate_data(self.restaurant_id, regenerate=regenerate)
        return self.data
    
    def prepare_data(self):
        """Prepare features and labels."""
        if self.data is None:
            self.load_data()
        
        X, y = prepare_features_labels(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = split_train_test(X, y)
    
    def get_latest_data(self, days: int = 30) -> pd.DataFrame:
        """Get latest data points."""
        if self.data is None:
            self.load_data()
        return self.data.tail(days)
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics."""
        if self.data is None:
            self.load_data()
        
        return {
            "total_records": len(self.data),
            "date_range": f"{self.data['date'].min().date()} to {self.data['date'].max().date()}",
            "avg_customers": self.data["customers"].mean(),
            "avg_revenue": self.data["revenue"].mean(),
            "revenue_std": self.data["revenue"].std(),
        }
