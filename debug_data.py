"""Debug script to check data loading and preparation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DataManager
import pandas as pd

# Load data
data_manager = DataManager(restaurant_id="rest_001")
data_manager.load_data(regenerate=True)

print("Raw data shape:", data_manager.data.shape)
print("\nColumn names:")
print(data_manager.data.columns.tolist())

print("\nFirst 5 rows:")
print(data_manager.data.head())

print("\nData types:")
print(data_manager.data.dtypes)

print("\nNaN counts:")
print(data_manager.data.isnull().sum())

print("\nData info:")
print(data_manager.data.info())

# Check what happens when we prepare
print("\n" + "="*80)
print("Preparing data...")
data_manager.prepare_data()

print(f"X_train shape: {data_manager.X_train.shape}")
print(f"X_test shape: {data_manager.X_test.shape}")
print(f"y_train shape: {data_manager.y_train.shape}")
print(f"y_test shape: {data_manager.y_test.shape}")

if len(data_manager.X_train) > 0:
    print("\nX_train columns:", data_manager.X_train.columns.tolist())
    print("X_train first 5 rows:")
    print(data_manager.X_train.head())
