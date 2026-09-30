"""Demand forecast model using Gradient Boosting for Restaurant Pulse AI"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import logging

from config import XGBOOST_PARAMS, MODELS_DIR, FORECAST_HORIZON_DAYS
from utils import setup_logger, normalize_features, inverse_normalize, MetricsCalculator

logger = setup_logger(__name__)


class DemandForecastModel:
    """Gradient Boosting-based demand forecasting model."""
    
    def __init__(self, model_name: str = "demand_forecast"):
        """
        Initialize the demand forecast model.
        
        Args:
            model_name: Name of the model for saving/loading
        """
        self.model_name = model_name
        self.model_path = MODELS_DIR / f"{model_name}_model.pkl"
        self.scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
        
        self.model = None
        self.feature_names = None
        self.feature_mean = None
        self.feature_std = None
        self.metrics = MetricsCalculator()
        self.is_trained = False
    
    def build_model(self) -> GradientBoostingRegressor:
        """Build Gradient Boosting model."""
        # Convert xgboost params to sklearn compatible ones
        self.model = GradientBoostingRegressor(
            n_estimators=XGBOOST_PARAMS.get("n_estimators", 100),
            max_depth=XGBOOST_PARAMS.get("max_depth", 6),
            learning_rate=XGBOOST_PARAMS.get("learning_rate", 0.1),
            subsample=XGBOOST_PARAMS.get("subsample", 0.8),
            random_state=XGBOOST_PARAMS.get("random_state", 42),
        )
        logger.info("Gradient Boosting model built")
        return self.model
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            self.build_model()
        
        self.feature_names = X_train.columns.tolist()
        
        # Normalize features
        X_train_normalized, self.feature_mean, self.feature_std = normalize_features(
            X_train.values
        )
        
        # Train model (scikit-learn doesn't support eval_set for GradientBoosting)
        self.model.fit(
            X_train_normalized,
            y_train,
        )
        
        self.is_trained = True
        logger.info("Model training completed")
        
        # Calculate training metrics
        y_pred_train = self.predict(X_train)
        metrics = self.metrics.calculate_all(y_train.values, y_pred_train, prefix="train")
        
        if X_val is not None and y_val is not None:
            y_pred_val = self.predict(X_val)
            self.metrics.calculate_all(y_val.values, y_pred_val, prefix="val")
            metrics.update(self.metrics.results)
        
        self.metrics.print_summary()
        return metrics
    
    def predict(
        self,
        X: pd.DataFrame,
        return_unnormalized: bool = True,
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            return_unnormalized: Whether to return predictions in original scale
        
        Returns:
            Predictions array
        """
        if self.model is None or not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Normalize features
        X_normalized, _, _ = normalize_features(
            X.values, self.feature_mean, self.feature_std
        )
        
        # Make predictions
        predictions = self.model.predict(X_normalized)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def forecast_future(
        self,
        last_data: pd.DataFrame,
        horizon_days: int = FORECAST_HORIZON_DAYS,
    ) -> pd.DataFrame:
        """
        Forecast future demand for next N days.
        
        Args:
            last_data: Recent historical data with features
            horizon_days: Number of days to forecast
        
        Returns:
            DataFrame with forecasts
        """
        if self.model is None or not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        forecasts = []
        current_data = last_data.copy()
        
        for i in range(horizon_days):
            # Prepare features for next day
            # In a real scenario, you'd need to handle feature engineering differently
            # For now, we'll use the most recent features as a base
            next_features = current_data.iloc[-1:].reset_index(drop=True)
            
            # Make prediction
            pred = self.predict(next_features)[0]
            
            # Create forecast record
            next_date = pd.to_datetime(next_features["date"].iloc[0]) + pd.Timedelta(days=1)
            forecast_record = {
                "date": next_date,
                "forecast_revenue": pred,
                "forecast_horizon_days": i + 1,
            }
            forecasts.append(forecast_record)
        
        forecast_df = pd.DataFrame(forecasts)
        return forecast_df
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not built")
        
        # Get feature importances from sklearn model
        importances = self.model.feature_importances_
        importance_dict = {name: imp for name, imp in zip(self.feature_names, importances)}
        
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=["feature", "importance"]
        ).sort_values("importance", ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Save model
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        scaler_dict = {
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "feature_names": self.feature_names,
        }
        with open(self.scaler_path, "wb") as f:
            pickle.dump(scaler_dict, f)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self) -> None:
        """Load model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load model
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(self.scaler_path, "rb") as f:
            scaler_dict = pickle.load(f)
            self.feature_mean = scaler_dict["feature_mean"]
            self.feature_std = scaler_dict["feature_std"]
            self.feature_names = scaler_dict["feature_names"]
        
        self.is_trained = True
        logger.info(f"Model loaded from {self.model_path}")
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        y_pred = self.predict(X_test)
        metrics = self.metrics.calculate_all(y_test.values, y_pred, prefix="test")
        self.metrics.print_summary()
        return metrics


class EnsembleForecaster:
    """Ensemble of multiple models for robust forecasting."""
    
    def __init__(self, model_names: Optional[List[str]] = None):
        """Initialize ensemble."""
        if model_names is None:
            model_names = ["demand_forecast_v1"]
        
        self.models = {name: DemandForecastModel(name) for name in model_names}
        self.weights = {name: 1.0 / len(model_names) for name in model_names}
    
    def add_model(self, model_name: str, weight: float = None) -> None:
        """Add a model to ensemble."""
        self.models[model_name] = DemandForecastModel(model_name)
        
        # Rebalance weights if not specified
        if weight is None:
            weight = 1.0 / len(self.models)
            for name in self.models:
                self.weights[name] = weight
        else:
            self.weights[model_name] = weight
    
    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Dict]:
        """Train all models."""
        results = {}
        for name, model in self.models.items():
            logger.info(f"Training model: {name}")
            results[name] = model.train(X_train, y_train, X_val, y_val)
        return results
    
    def predict_ensemble(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.
        
        Returns:
            Tuple of (ensemble_predictions, prediction_std)
        """
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            weighted_pred = pred * self.weights[name]
            predictions.append(weighted_pred)
        
        ensemble_pred = np.sum(predictions, axis=0)
        
        # Calculate uncertainty (std of individual predictions)
        all_preds = np.array([model.predict(X) for model in self.models.values()])
        pred_std = np.std(all_preds, axis=0)
        
        return ensemble_pred, pred_std
    
    def save_all_models(self) -> None:
        """Save all models."""
        for model in self.models.values():
            model.save_model()
        logger.info("All models saved")
    
    def load_all_models(self) -> None:
        """Load all models."""
        for model in self.models.values():
            model.load_model()
        logger.info("All models loaded")
