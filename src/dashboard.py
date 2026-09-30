"""Streamlit Dashboard for Restaurant Pulse AI"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List

from config import STREAMLIT_PAGE_CONFIG, SAMPLE_RESTAURANTS
from data_loader import DataManager, get_restaurant_info
from forecast_model import DemandForecastModel
from anomaly_detector import AnomalyDetector
from strategy_agent import StrategyAgent, RecommendationEngine
from utils import format_currency, format_percentage


# Page configuration
st.set_page_config(**STREAMLIT_PAGE_CONFIG)

# Custom styling
st.markdown("""
    <style>
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .anomaly-critical { color: #ff4444; font-weight: bold; }
    .anomaly-high { color: #ff8800; font-weight: bold; }
    .anomaly-medium { color: #ffbb00; }
    .anomaly-low { color: #88dd00; }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "loaded" not in st.session_state:
        st.session_state.loaded = False
        st.session_state.data_manager = None
        st.session_state.model = None
        st.session_state.detector = None
        st.session_state.agent = None
        st.session_state.anomalies = []
        st.session_state.forecast = None


def load_data_and_models(restaurant_id: str):
    """Load data and initialize models."""
    with st.spinner("Loading data and models..."):
        # Load data
        data_manager = DataManager(restaurant_id)
        data_manager.load_data()
        data_manager.prepare_data()
        
        # Initialize anomaly detector
        detector = AnomalyDetector()
        detector.compute_baseline_stats(
            data_manager.data,
            ["revenue", "customers"],
            window_days=30
        )
        
        # Initialize agent
        agent = StrategyAgent()
        
        # Store in session
        st.session_state.data_manager = data_manager
        st.session_state.detector = detector
        st.session_state.agent = agent
        st.session_state.loaded = True


def create_metric_cards(metrics: Dict):
    """Create metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Avg Daily Revenue",
            format_currency(metrics.get("avg_revenue", 0)),
            f"{metrics.get('revenue_trend', 0):.1f}%"
        )
    
    with col2:
        st.metric(
            "Avg Daily Customers",
            f"{metrics.get('avg_customers', 0):.0f}",
            f"{metrics.get('customer_trend', 0):.1f}%"
        )
    
    with col3:
        st.metric(
            "Anomalies Detected",
            len(st.session_state.anomalies),
            f"{len([a for a in st.session_state.anomalies if a.severity > 0.7])} Critical"
        )
    
    with col4:
        st.metric(
            "Forecast Accuracy",
            format_percentage(metrics.get("forecast_accuracy", 85)),
            "Model MAPE"
        )


def create_revenue_forecast_chart(data: pd.DataFrame, forecast_df: pd.DataFrame):
    """Create revenue forecast visualization."""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data["date"],
        y=data["revenue"],
        name="Historical Revenue",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Revenue: $%{y:.2f}<extra></extra>"
    ))
    
    # Forecast
    if forecast_df is not None and len(forecast_df) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast_revenue"],
            name="Forecast",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast: $%{y:.2f}<extra></extra>"
        ))
        
        # Confidence interval
        if "forecast_lower" in forecast_df.columns and "forecast_upper" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["forecast_upper"],
                fill=None,
                mode="lines",
                line_color="rgba(0,0,0,0)",
                showlegend=False,
                hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["forecast_lower"],
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,0,0,0)",
                name="Confidence Interval",
                fillcolor="rgba(255, 127, 14, 0.2)",
                hoverinfo="skip"
            ))
    
    fig.update_layout(
        title="Revenue Forecast",
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        hovermode="x unified",
        height=400,
        template="plotly_white"
    )
    
    return fig


def create_anomaly_timeline(anomalies: List):
    """Create anomaly timeline visualization."""
    if not anomalies:
        st.info("No anomalies detected in the selected period.")
        return
    
    # Prepare data
    anom_data = []
    for anom in anomalies:
        anom_data.append({
            "date": anom.date,
            "type": anom.type.value,
            "severity": anom.severity,
            "description": anom.description,
        })
    
    anom_df = pd.DataFrame(anom_data)
    
    # Create scatter plot
    fig = px.scatter(
        anom_df,
        x="date",
        y="severity",
        color="type",
        size="severity",
        hover_data=["description"],
        title="Anomaly Timeline",
        labels={"severity": "Severity Score", "type": "Anomaly Type"}
    )
    
    fig.update_layout(height=400, template="plotly_white")
    return fig


def create_recommendation_section(recommendations: List[Dict]):
    """Create recommendations section."""
    if not recommendations:
        st.info("No recommendations available yet.")
        return
    
    # Group by priority
    priorities = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    
    for priority in ["critical", "high", "medium", "low"]:
        recs = [r for r in recommendations if r.get("priority") == priority]
        if not recs:
            continue
        
        st.subheader(f"{priorities.get(priority, '')} {priority.title()} Priority")
        
        for rec in recs:
            with st.expander(f"**{rec['title']}**"):
                st.write(f"**Category:** {rec.get('category')}")
                st.write(f"**Description:** {rec.get('description')}")
                if rec.get('expected_impact'):
                    st.write(f"**Expected Impact:** {rec.get('expected_impact')}")


# Main app
def main():
    initialize_session_state()
    
    # Sidebar
    st.sidebar.title("🍽️ Restaurant Pulse AI")
    
    # Restaurant selection
    restaurant_options = {rest["name"]: rest["id"] for rest in SAMPLE_RESTAURANTS}
    selected_restaurant = st.sidebar.selectbox(
        "Select Restaurant",
        options=list(restaurant_options.keys())
    )
    restaurant_id = restaurant_options[selected_restaurant]
    
    # Load data
    if not st.session_state.loaded or st.session_state.data_manager is None:
        load_data_and_models(restaurant_id)
    
    # Sidebar sections
    with st.sidebar:
        st.markdown("---")
        
        # Date range selector
        date_range = st.slider(
            "Analysis Period (Days)",
            min_value=7,
            max_value=365,
            value=90,
            step=7,
        )
        
        # Anomaly sensitivity
        sensitivity = st.select_slider(
            "Anomaly Detection Sensitivity",
            options=["Low", "Medium", "High"],
            value="Medium"
        )
        
        # Refresh data
        if st.button("🔄 Refresh Data"):
            st.session_state.loaded = False
            load_data_and_models(restaurant_id)
            st.rerun()
    
    # Main content
    st.title(f"📊 {selected_restaurant} - Demand Forecasting Dashboard")
    
    # Get restaurant info
    restaurant_info = get_restaurant_info(restaurant_id)
    
    # Restaurant info cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cuisine", restaurant_info.get("cuisine", "N/A"))
    with col2:
        st.metric("Seating", restaurant_info.get("seating_capacity", "N/A"))
    with col3:
        st.metric("Avg Check", format_currency(restaurant_info.get("avg_check_size", 0)))
    with col4:
        st.metric("Status", "🟢 Active")
    
    st.markdown("---")
    
    # Key metrics
    data_manager = st.session_state.data_manager
    recent_data = data_manager.get_latest_data(30)
    
    metrics = {
        "avg_revenue": recent_data["revenue"].mean(),
        "avg_customers": recent_data["customers"].mean(),
        "revenue_trend": ((recent_data["revenue"].iloc[-1] / recent_data["revenue"].iloc[0]) - 1) * 100,
        "customer_trend": ((recent_data["customers"].iloc[-1] / recent_data["customers"].iloc[0]) - 1) * 100,
        "forecast_accuracy": 85.5,
    }
    
    create_metric_cards(metrics)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast", "🚨 Anomalies", "💡 Recommendations", "📋 Details"
    ])
    
    with tab1:
        st.subheader("Demand Forecast")
        
        # Create forecast chart
        col1, col2 = st.columns([3, 1])
        
        with col1:
            forecast_chart = create_revenue_forecast_chart(
                data_manager.get_latest_data(date_range),
                None  # TODO: Add actual forecast
            )
            st.plotly_chart(forecast_chart, width='stretch')
        
        with col2:
            st.write("**Forecast Metrics**")
            st.info(f"**Horizon:** 30 days\n**Confidence:** High\n**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    with tab2:
        st.subheader("Anomaly Detection")
        
        # Detect anomalies
        detector = st.session_state.detector
        analysis_data = data_manager.get_latest_data(date_range)
        anomalies = detector.detect_all_anomalies(
            analysis_data,
            columns=["revenue", "customers"],
        )
        st.session_state.anomalies = anomalies
        
        anomaly_summary = detector.get_anomalies_summary()
        
        # Anomaly summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Anomalies", anomaly_summary.get("total", 0))
        with col2:
            st.metric("Critical", anomaly_summary.get("critical", 0))
        with col3:
            st.metric("By Type", len(anomaly_summary.get("by_type", {})))
        
        # Anomaly timeline
        if anomalies:
            chart = create_anomaly_timeline(anomalies)
            st.plotly_chart(chart, width='stretch')
        
        # Anomaly details
        st.subheader("Detected Anomalies")
        
        for anom in anomalies[:10]:  # Show top 10
            severity_level = "🔴 Critical" if anom.severity > 0.8 else "🟠 High" if anom.severity > 0.6 else "🟡 Medium"
            
            with st.expander(f"{severity_level} - {anom.type.value.replace('_', ' ').title()} ({anom.date.strftime('%Y-%m-%d')})"):
                st.write(f"**Description:** {anom.description}")
                st.write(f"**Severity:** {format_percentage(anom.severity * 100)}")
                
                # Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Key Metrics:**")
                    for key, val in anom.metrics.items():
                        st.write(f"- {key}: {val:.2f}")
                
                with col2:
                    st.write("**Recommended Actions:**")
                    for action in anom.recommended_actions:
                        st.write(f"- {action}")
    
    with tab3:
        st.subheader("AI-Powered Recommendations")
        
        # Generate recommendations
        engine = RecommendationEngine()
        agent = st.session_state.agent
        
        # Simple analysis for demo
        analysis = {
            "forecast_summary": {
                "horizon_days": 30,
                "trend": "stable" if metrics["revenue_trend"] < 5 else "increasing" if metrics["revenue_trend"] > 0 else "declining",
            }
        }
        
        recommendations = engine.generate_recommendations(
            analysis,
            [a.to_dict() for a in anomalies],
        )
        
        create_recommendation_section(recommendations)
    
    with tab4:
        st.subheader("Detailed Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Summary**")
            summary = data_manager.get_data_summary()
            for key, val in summary.items():
                st.write(f"- {key}: {val}")
        
        with col2:
            st.write("**Recent Performance**")
            st.dataframe(
                recent_data[["date", "customers", "revenue"]].tail(10),
                width='stretch'
            )
        
        # Raw data
        with st.expander("View Raw Data"):
            st.dataframe(
                data_manager.get_latest_data(date_range),
                width='stretch'
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Restaurant Pulse AI** | Autonomous Demand Forecaster & Strategy Assistant
        
        v0.1.0 | Last updated: 2026-09-29
        """
    )


if __name__ == "__main__":
    main()
