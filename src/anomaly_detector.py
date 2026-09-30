"""Anomaly detection engine for Restaurant Pulse AI"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from config import ANOMALY_THRESHOLD_STD, ANOMALY_SENSITIVITY
from utils import setup_logger

logger = setup_logger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected."""
    REVENUE_DROP = "revenue_drop"
    REVENUE_SPIKE = "revenue_spike"
    DEMAND_DROP = "demand_drop"
    DEMAND_SPIKE = "demand_spike"
    TREND_CHANGE = "trend_change"
    UNUSUAL_PATTERN = "unusual_pattern"


@dataclass
class Anomaly:
    """Represents an anomaly event."""
    type: AnomalyType
    date: pd.Timestamp
    severity: float  # 0-1 score
    description: str
    metrics: Dict
    recommended_actions: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "date": str(self.date.date()),
            "severity": round(self.severity, 3),
            "description": self.description,
            "metrics": self.metrics,
            "recommended_actions": self.recommended_actions,
        }


class AnomalyDetector:
    """Detect anomalies in restaurant data."""
    
    def __init__(
        self,
        threshold_std: float = ANOMALY_THRESHOLD_STD,
        sensitivity: str = ANOMALY_SENSITIVITY,
    ):
        """
        Initialize anomaly detector.
        
        Args:
            threshold_std: Number of standard deviations for anomaly
            sensitivity: Detection sensitivity (low, medium, high)
        """
        self.threshold_std = threshold_std
        self.sensitivity = sensitivity
        
        # Adjust threshold based on sensitivity
        sensitivity_multipliers = {"low": 1.5, "medium": 1.0, "high": 0.75}
        self.threshold_std *= sensitivity_multipliers.get(sensitivity, 1.0)
        
        self.anomalies = []
        self.baseline_stats = {}
    
    def compute_baseline_stats(
        self,
        data: pd.DataFrame,
        columns: List[str],
        window_days: int = 30,
    ) -> Dict:
        """
        Compute baseline statistics from historical data.
        
        Args:
            data: Historical data
            columns: Columns to compute statistics for
            window_days: Number of days for rolling statistics
        
        Returns:
            Dictionary with baseline statistics
        """
        stats = {}
        for col in columns:
            if col not in data.columns:
                continue
            
            # Overall mean and std
            stats[col] = {
                "mean": data[col].mean(),
                "std": data[col].std(),
                "median": data[col].median(),
                "q25": data[col].quantile(0.25),
                "q75": data[col].quantile(0.75),
                "min": data[col].min(),
                "max": data[col].max(),
            }
            
            # Rolling statistics
            stats[f"{col}_rolling"] = {
                "mean": data[col].rolling(window_days).mean().iloc[-1],
                "std": data[col].rolling(window_days).std().iloc[-1],
            }
        
        self.baseline_stats = stats
        logger.info(f"Baseline statistics computed for {len(columns)} columns")
        return stats
    
    def detect_level_anomalies(
        self,
        data: pd.DataFrame,
        column: str,
    ) -> List[Anomaly]:
        """
        Detect anomalies based on deviation from baseline (Z-score).
        
        Args:
            data: Data to analyze
            column: Column to analyze
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if column not in self.baseline_stats:
            logger.warning(f"No baseline stats for {column}")
            return anomalies
        
        stats = self.baseline_stats[column]
        mean = stats["mean"]
        std = stats["std"]
        
        if std == 0:
            return anomalies
        
        # Calculate z-scores
        z_scores = np.abs((data[column].values - mean) / std)
        threshold = self.threshold_std
        
        anomaly_mask = z_scores > threshold
        
        for idx, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly:
                value = data[column].iloc[idx]
                z_score = z_scores[idx]
                severity = min(z_score / (threshold * 2), 1.0)
                
                # Determine anomaly type
                if value < mean:
                    anom_type = AnomalyType.REVENUE_DROP if "revenue" in column.lower() else AnomalyType.DEMAND_DROP
                else:
                    anom_type = AnomalyType.REVENUE_SPIKE if "revenue" in column.lower() else AnomalyType.DEMAND_SPIKE
                
                # Generate description
                pct_change = ((value - mean) / mean) * 100
                description = f"{column} is {abs(pct_change):.1f}% {'below' if pct_change < 0 else 'above'} baseline"
                
                # Recommend actions
                actions = self._get_recommended_actions(anom_type, pct_change)
                
                anomaly = Anomaly(
                    type=anom_type,
                    date=data.index[idx] if isinstance(data.index, pd.DatetimeIndex) else data["date"].iloc[idx],
                    severity=severity,
                    description=description,
                    metrics={
                        "value": float(value),
                        "baseline_mean": float(mean),
                        "z_score": float(z_score),
                        "pct_change": float(pct_change),
                    },
                    recommended_actions=actions,
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_trend_anomalies(
        self,
        data: pd.DataFrame,
        column: str,
        window_days: int = 7,
    ) -> List[Anomaly]:
        """
        Detect changes in trend direction.
        
        Args:
            data: Data to analyze
            column: Column to analyze
            window_days: Window for trend comparison
        
        Returns:
            List of detected trend anomalies
        """
        anomalies = []
        
        if len(data) < window_days * 2:
            return anomalies
        
        # Calculate rolling change
        pct_change = data[column].pct_change(window_days) * 100
        
        # Detect trend reversals
        # Look at recent trend vs historical trend
        recent_trend = pct_change.iloc[-window_days:].mean()
        historical_trend = pct_change.iloc[:-window_days].mean()
        
        if np.isnan(recent_trend) or np.isnan(historical_trend):
            return anomalies
        
        trend_change = recent_trend - historical_trend
        
        # Check if trend change is significant
        trend_std = pct_change.std()
        if trend_std > 0 and abs(trend_change) > 2 * trend_std:
            severity = min(abs(trend_change) / (4 * trend_std), 1.0)
            
            description = f"Trend reversal detected: recent trend is {'positive' if recent_trend > 0 else 'negative'}"
            
            actions = ["Monitor closely", "Review external factors (seasonality, promotions, competitions)"]
            if recent_trend < historical_trend:
                actions.append("Consider promotional campaigns to boost demand")
            else:
                actions.append("Prepare for increased demand")
            
            anomaly = Anomaly(
                type=AnomalyType.TREND_CHANGE,
                date=data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else data["date"].iloc[-1],
                severity=severity,
                description=description,
                metrics={
                    "recent_trend": float(recent_trend),
                    "historical_trend": float(historical_trend),
                    "trend_change": float(trend_change),
                },
                recommended_actions=actions,
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def detect_forecast_anomalies(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        threshold_pct: float = 20.0,
    ) -> List[Anomaly]:
        """
        Detect anomalies by comparing actual vs forecast.
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            threshold_pct: Percentage threshold for anomaly
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(actual) != len(forecast):
            logger.warning("Actual and forecast lengths do not match")
            return anomalies
        
        # Calculate percentage errors
        errors = np.abs((actual.values - forecast.values) / np.maximum(forecast.values, 1)) * 100
        
        error_mean = errors.mean()
        error_std = errors.std()
        
        for idx, error in enumerate(errors):
            if error > error_mean + 2 * error_std:
                severity = min((error - error_mean) / (2 * error_std), 1.0)
                
                description = f"Forecast error of {error:.1f}% for actual value {actual.iloc[idx]:.0f} vs forecast {forecast.iloc[idx]:.0f}"
                
                actions = ["Investigate cause of forecast error", "Check for data quality issues", "Retrain model with new data"]
                
                anomaly = Anomaly(
                    type=AnomalyType.UNUSUAL_PATTERN,
                    date=actual.index[idx] if isinstance(actual.index, pd.DatetimeIndex) else idx,
                    severity=severity,
                    description=description,
                    metrics={
                        "actual": float(actual.iloc[idx]),
                        "forecast": float(forecast.iloc[idx]),
                        "error_pct": float(error),
                    },
                    recommended_actions=actions,
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_all_anomalies(
        self,
        data: pd.DataFrame,
        columns: List[str] = None,
        include_forecasts: bool = False,
        actual_col: str = "revenue",
        forecast_col: str = "forecast",
    ) -> List[Anomaly]:
        """
        Detect all types of anomalies.
        
        Args:
            data: Data to analyze
            columns: Columns to check for anomalies
            include_forecasts: Whether to include forecast comparison
            actual_col: Column name for actual values
            forecast_col: Column name for forecast values
        
        Returns:
            Sorted list of anomalies (by severity)
        """
        if columns is None:
            columns = ["revenue", "customers"]
        
        all_anomalies = []
        
        # Level anomalies
        for col in columns:
            if col in data.columns:
                level_anom = self.detect_level_anomalies(data, col)
                all_anomalies.extend(level_anom)
        
        # Trend anomalies
        for col in columns:
            if col in data.columns:
                trend_anom = self.detect_trend_anomalies(data, col)
                all_anomalies.extend(trend_anom)
        
        # Forecast anomalies
        if include_forecasts and actual_col in data.columns and forecast_col in data.columns:
            forecast_anom = self.detect_forecast_anomalies(
                data[actual_col],
                data[forecast_col]
            )
            all_anomalies.extend(forecast_anom)
        
        # Sort by severity
        all_anomalies.sort(key=lambda x: x.severity, reverse=True)
        
        self.anomalies = all_anomalies
        logger.info(f"Detected {len(all_anomalies)} anomalies")
        
        return all_anomalies
    
    @staticmethod
    def _get_recommended_actions(
        anomaly_type: AnomalyType,
        pct_change: float,
    ) -> List[str]:
        """Get recommended actions based on anomaly type."""
        actions = []
        
        if anomaly_type == AnomalyType.REVENUE_DROP:
            actions = [
                "Launch promotional campaign to boost sales",
                "Review menu pricing strategy",
                "Analyze customer feedback for satisfaction issues",
                "Check competitor activity",
                "Consider inventory optimization",
            ]
        elif anomaly_type == AnomalyType.REVENUE_SPIKE:
            actions = [
                "Analyze what drove the spike (external event, promotion, etc.)",
                "Ensure sufficient inventory and staffing",
                "Document successful factors for future reference",
                "Consider capacity planning",
            ]
        elif anomaly_type == AnomalyType.DEMAND_DROP:
            actions = [
                "Increase marketing efforts",
                "Review service quality",
                "Check for operational issues",
                "Consider limited-time offers",
            ]
        elif anomaly_type == AnomalyType.DEMAND_SPIKE:
            actions = [
                "Ensure adequate staffing levels",
                "Verify inventory levels",
                "Prepare for potential service challenges",
                "Consider upselling opportunities",
            ]
        
        return actions
    
    def get_critical_anomalies(self, threshold: float = 0.7) -> List[Anomaly]:
        """Get critical anomalies (high severity)."""
        return [a for a in self.anomalies if a.severity >= threshold]
    
    def get_anomalies_summary(self) -> Dict:
        """Get summary of detected anomalies."""
        if not self.anomalies:
            return {"total": 0, "critical": 0, "by_type": {}}
        
        by_type = {}
        for anomaly in self.anomalies:
            anom_type = anomaly.type.value
            by_type[anom_type] = by_type.get(anom_type, 0) + 1
        
        critical = len(self.get_critical_anomalies())
        
        return {
            "total": len(self.anomalies),
            "critical": critical,
            "by_type": by_type,
        }
