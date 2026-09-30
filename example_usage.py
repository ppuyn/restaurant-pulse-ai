"""
Example usage script for Restaurant Pulse AI

Demonstrates:
1. Data loading and exploration
2. Training demand forecast model
3. Detecting anomalies
4. Generating strategy recommendations
5. Creating comprehensive reports
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Local imports
from config import (
    SAMPLE_RESTAURANTS,
    HISTORICAL_DATA_DAYS,
    FORECAST_HORIZON_DAYS,
)
from data_loader import (
    DataManager,
    generate_synthetic_sales_data,
    generate_synthetic_weather_data,
    combine_datasets,
)
from forecast_model import DemandForecastModel, EnsembleForecaster
from anomaly_detector import AnomalyDetector
from strategy_agent import StrategyAgent, RecommendationEngine
from utils import setup_logger, format_currency, format_percentage

logger = setup_logger(__name__)


def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("🍽️  RESTAURANT PULSE AI - DEMONSTRATION")
    print("="*80 + "\n")
    
    # Step 1: Load and explore data
    print("STEP 1: Data Loading & Exploration")
    print("-" * 80)
    
    restaurant_id = "rest_001"
    rest_info = next(r for r in SAMPLE_RESTAURANTS if r["id"] == restaurant_id)
    
    print(f"Restaurant: {rest_info['name']}")
    print(f"Cuisine: {rest_info['cuisine']}")
    print(f"Capacity: {rest_info['seating_capacity']} seats")
    print(f"Avg Check: {format_currency(rest_info['avg_check_size'])}\n")
    
    # Load data
    data_manager = DataManager(restaurant_id)
    data_manager.load_data(regenerate=True)  # Generate fresh synthetic data
    
    data_summary = data_manager.get_data_summary()
    print("Historical Data Summary:")
    for key, value in data_summary.items():
        if isinstance(value, str):
            print(f"  {key}: {value}")
        elif isinstance(value, float):
            if "revenue" in key.lower():
                print(f"  {key}: {format_currency(value)}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Prepare data for modeling
    data_manager.prepare_data()
    print(f"\nTrain set size: {len(data_manager.X_train)}")
    print(f"Test set size: {len(data_manager.X_test)}")
    
    # Step 2: Train demand forecast model
    print("\n" + "="*80)
    print("STEP 2: Training Demand Forecast Model")
    print("-" * 80)
    
    model = DemandForecastModel(model_name="demand_forecast_v1")
    model.build_model()
    
    print("Training XGBoost model...")
    metrics = model.train(
        data_manager.X_train,
        data_manager.y_train,
        data_manager.X_test,
        data_manager.y_test,
    )
    
    print("\nTraining completed!")
    print("Metrics Summary:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    importance = model.get_feature_importance(top_n=10)
    for idx, row in importance.iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance']}")
    
    # Step 3: Anomaly Detection
    print("\n" + "="*80)
    print("STEP 3: Anomaly Detection")
    print("-" * 80)
    
    detector = AnomalyDetector(sensitivity="medium")
    analysis_data = data_manager.get_latest_data(60)
    
    print("Computing baseline statistics...")
    detector.compute_baseline_stats(analysis_data, ["revenue", "customers"])
    
    print("Detecting anomalies...")
    anomalies = detector.detect_all_anomalies(
        analysis_data,
        columns=["revenue", "customers"],
    )
    
    summary = detector.get_anomalies_summary()
    print(f"\nAnomaly Detection Summary:")
    print(f"  Total anomalies: {summary['total']}")
    print(f"  Critical anomalies: {summary['critical']}")
    print(f"  By type: {summary['by_type']}")
    
    # Show critical anomalies
    critical = detector.get_critical_anomalies(threshold=0.7)
    if critical:
        print(f"\nCritical Anomalies (Top 3):")
        for anom in critical[:3]:
            print(f"\n  Type: {anom.type.value}")
            print(f"  Date: {anom.date.date()}")
            print(f"  Severity: {format_percentage(anom.severity * 100)}")
            print(f"  Description: {anom.description}")
            print(f"  Actions: {', '.join(anom.recommended_actions[:2])}")
    
    # Step 4: Strategy Generation
    print("\n" + "="*80)
    print("STEP 4: AI-Powered Strategy Generation")
    print("-" * 80)
    
    agent = StrategyAgent()
    
    # Forecast analysis
    forecast_data = {
        "horizon_days": FORECAST_HORIZON_DAYS,
        "avg_forecast": data_summary["avg_revenue"],
        "trend": "increasing" if data_summary["avg_revenue"] > 0 else "stable",
        "confidence": "high",
    }
    
    historical_data = {
        "avg_revenue": data_summary["avg_revenue"],
        "avg_customers": data_summary["avg_customers"],
        "trend": "stable",
        "volatility": "medium",
    }
    
    print("Analyzing demand forecast...")
    analysis = agent.analyze_demand_forecast(forecast_data, historical_data)
    
    print("\nForecast Analysis:")
    print(analysis["analysis"][:500] + "..." if len(analysis["analysis"]) > 500 else analysis["analysis"])
    
    # Generate recommendations
    print("\nGenerating business recommendations...")
    engine = RecommendationEngine()
    
    recommendations = engine.generate_recommendations(
        analysis,
        [a.to_dict() for a in anomalies[:5]],  # Top 5 anomalies
        constraints={"budget": 5000, "max_staff": 15},
    )
    
    print(f"\nGenerated {len(recommendations)} recommendations:\n")
    
    # Print recommendations grouped by priority
    priority_order = ["critical", "high", "medium", "low"]
    for priority in priority_order:
        priority_recs = [r for r in recommendations if r.get("priority") == priority]
        if priority_recs:
            print(f"\n{priority.upper()} Priority ({len(priority_recs)} items):")
            for rec in priority_recs[:2]:  # Show top 2 per priority
                print(f"  ✓ {rec['title']}")
                print(f"    {rec['description']}")
    
    # Step 5: Forecasting Future Demand
    print("\n" + "="*80)
    print("STEP 5: Future Demand Forecasting")
    print("-" * 80)
    
    print(f"Generating {FORECAST_HORIZON_DAYS}-day forecast...")
    recent_data = data_manager.get_latest_data(30)
    
    try:
        forecast_df = model.forecast_future(recent_data, horizon_days=FORECAST_HORIZON_DAYS)
        
        print(f"\nForecast Summary (Next {FORECAST_HORIZON_DAYS} days):")
        print(f"  Average forecast revenue: {format_currency(forecast_df['forecast_revenue'].mean())}")
        print(f"  Min: {format_currency(forecast_df['forecast_revenue'].min())}")
        print(f"  Max: {format_currency(forecast_df['forecast_revenue'].max())}")
        print(f"  Std Dev: {format_currency(forecast_df['forecast_revenue'].std())}")
        
        print(f"\nForecast for next 7 days:")
        for idx, row in forecast_df.head(7).iterrows():
            print(f"  {row['date'].date()}: {format_currency(row['forecast_revenue'])}")
    
    except Exception as e:
        print(f"  Note: Detailed forecasting requires proper feature engineering. {str(e)}")
    
    # Step 6: Summary & Action Items
    print("\n" + "="*80)
    print("STEP 6: Executive Summary & Action Items")
    print("="*80 + "\n")
    
    print(f"Restaurant: {rest_info['name']}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis Period: Last 60 days")
    
    print(f"\nKey Findings:")
    print(f"  • Average daily revenue: {format_currency(data_summary['avg_revenue'])}")
    print(f"  • Average daily customers: {data_summary['avg_customers']:.0f}")
    print(f"  • Revenue volatility: ${data_summary['revenue_std']:.2f}")
    print(f"  • Anomalies detected: {summary['total']} (Critical: {summary['critical']})")
    
    print(f"\nImmediate Actions:")
    critical_recs = [r for r in recommendations if r.get("priority") == "critical"]
    if critical_recs:
        for i, rec in enumerate(critical_recs[:3], 1):
            print(f"  {i}. {rec['title']}")
    else:
        print(f"  1. Continue monitoring demand patterns")
        print(f"  2. Review menu pricing strategy")
        print(f"  3. Analyze competitor activities")
    
    print("\nNext Steps:")
    print(f"  • Implement dashboard for continuous monitoring")
    print(f"  • Schedule weekly strategy reviews")
    print(f"  • Set up automated alerts for critical anomalies")
    print(f"  • Monitor forecast accuracy and retrain models weekly")
    
    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
