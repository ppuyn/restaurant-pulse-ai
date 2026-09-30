"""Configuration module for Restaurant Pulse AI"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo")  # Can also use gpt-3.5-turbo

# Data Configuration
HISTORICAL_DATA_DAYS = 365
FORECAST_HORIZON_DAYS = 30  # Forecast next 30 days
TEST_SPLIT_RATIO = 0.2

# Model Configuration
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# Anomaly Detection Configuration
ANOMALY_THRESHOLD_STD = 2.5  # Number of standard deviations for anomaly
ANOMALY_SENSITIVITY = "medium"  # low, medium, high

# Dashboard Configuration
STREAMLIT_PAGE_CONFIG = {
    "page_title": "Restaurant Pulse AI Dashboard",
    "page_icon": "🍽️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Sample Restaurant Data
SAMPLE_RESTAURANTS = [
    {
        "id": "rest_001",
        "name": "The Golden Fork",
        "cuisine": "Italian",
        "seating_capacity": 85,
        "avg_check_size": 35.50,
    },
    {
        "id": "rest_002",
        "name": "Tokyo Sunrise",
        "cuisine": "Japanese",
        "seating_capacity": 60,
        "avg_check_size": 42.00,
    },
    {
        "id": "rest_003",
        "name": "El Sabor Latino",
        "cuisine": "Mexican",
        "seating_capacity": 120,
        "avg_check_size": 28.75,
    },
    {
        "id": "rest_004",
        "name": "Le Petit Bistro",
        "cuisine": "French",
        "seating_capacity": 50,
        "avg_check_size": 55.00,
    },
    {
        "id": "rest_005",
        "name": "Dragon Palace",
        "cuisine": "Chinese",
        "seating_capacity": 100,
        "avg_check_size": 32.00,
    },
]
