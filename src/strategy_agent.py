"""LangChain-based Strategy Agent for Restaurant Pulse AI"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import OPENAI_API_KEY, LLM_MODEL
from utils import setup_logger

logger = setup_logger(__name__)


class StrategyAgent:
    """LangChain-based agent for generating AI-driven business strategies."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = LLM_MODEL,
    ):
        """
        Initialize the strategy agent.
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Agent will operate in demo mode.")
        
        self.llm = None
        self._initialize_llm()
        
        self.conversation_history = []
        self.recommendations = []
        self.strategies = {}
    
    def _initialize_llm(self) -> None:
        """Initialize LLM."""
        if self.api_key:
            try:
                self.llm = ChatOpenAI(
                    api_key=self.api_key,
                    model=self.model,
                    temperature=0.7,
                )
                logger.info(f"LLM initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                self.llm = None
        else:
            logger.info("LLM not initialized (no API key)")
    
    def analyze_demand_forecast(
        self,
        forecast_data: Dict[str, Any],
        historical_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze demand forecast and generate strategic insights.
        
        Args:
            forecast_data: Forecast information
            historical_data: Historical data summary
        
        Returns:
            Analysis results and recommendations
        """
        # Prepare analysis prompt
        analysis_prompt = self._create_forecast_analysis_prompt(
            forecast_data, historical_data
        )
        
        if self.llm:
            try:
                response = self.llm.invoke([
                    SystemMessage(content=self._get_system_prompt()),
                    HumanMessage(content=analysis_prompt),
                ])
                analysis_result = response.content
            except Exception as e:
                logger.error(f"LLM API error: {e}")
                analysis_result = self._fallback_forecast_analysis(forecast_data)
        else:
            analysis_result = self._fallback_forecast_analysis(forecast_data)
        
        result = {
            "type": "forecast_analysis",
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis_result,
            "forecast_summary": forecast_data,
        }
        
        self.conversation_history.append(result)
        return result
    
    def analyze_anomalies(
        self,
        anomalies: List[Dict[str, Any]],
        restaurant_context: Dict[str, Any],
    ) -> Dict[str, List[Dict]]:
        """
        Analyze anomalies and generate targeted strategies.
        
        Args:
            anomalies: List of detected anomalies
            restaurant_context: Restaurant information
        
        Returns:
            Strategies for each anomaly
        """
        strategies = {}
        
        for anomaly in anomalies:
            strategy_prompt = self._create_anomaly_strategy_prompt(
                anomaly, restaurant_context
            )
            
            if self.llm:
                try:
                    response = self.llm.invoke([
                        SystemMessage(content=self._get_system_prompt()),
                        HumanMessage(content=strategy_prompt),
                    ])
                    strategy_text = response.content
                except Exception as e:
                    logger.error(f"LLM API error: {e}")
                    strategy_text = self._fallback_anomaly_strategy(anomaly)
            else:
                strategy_text = self._fallback_anomaly_strategy(anomaly)
            
            strategies[anomaly.get("type", "unknown")] = {
                "anomaly": anomaly,
                "strategy": strategy_text,
                "timestamp": datetime.now().isoformat(),
            }
        
        self.strategies.update(strategies)
        return strategies
    
    def generate_action_plan(
        self,
        forecast: Dict,
        anomalies: List[Dict],
        constraints: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive action plan.
        
        Args:
            forecast: Forecast data
            anomalies: Detected anomalies
            constraints: Business constraints (budget, capacity, etc.)
        
        Returns:
            Complete action plan
        """
        prompt = self._create_action_plan_prompt(
            forecast, anomalies, constraints
        )
        
        if self.llm:
            try:
                response = self.llm.invoke([
                    SystemMessage(content=self._get_system_prompt()),
                    HumanMessage(content=prompt),
                ])
                action_plan = response.content
            except Exception as e:
                logger.error(f"LLM API error: {e}")
                action_plan = self._fallback_action_plan(forecast, anomalies)
        else:
            action_plan = self._fallback_action_plan(forecast, anomalies)
        
        result = {
            "action_plan": action_plan,
            "generated_at": datetime.now().isoformat(),
            "forecast_horizon": forecast.get("horizon_days", 30),
            "anomalies_addressed": len(anomalies),
        }
        
        self.conversation_history.append(result)
        return result
    
    def get_recommendations(
        self,
        category: Optional[str] = None,
    ) -> List[Dict]:
        """Get stored recommendations, optionally filtered by category."""
        recommendations = self.recommendations
        
        if category:
            recommendations = [r for r in recommendations if r.get("category") == category]
        
        return recommendations
    
    def add_recommendation(
        self,
        title: str,
        description: str,
        category: str,
        priority: str = "medium",
        expected_impact: Optional[str] = None,
    ) -> None:
        """Add a new recommendation."""
        rec = {
            "title": title,
            "description": description,
            "category": category,
            "priority": priority,
            "expected_impact": expected_impact,
            "created_at": datetime.now().isoformat(),
        }
        self.recommendations.append(rec)
    
    # Prompt templates
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return """You are an expert restaurant business consultant AI named RestaurantPulse AI. 
Your role is to analyze restaurant data, demand forecasts, and anomalies to provide 
actionable, data-driven business recommendations.

Your recommendations should be:
- Specific and actionable
- Based on the provided data
- Prioritized by impact and urgency
- Considerate of typical restaurant operations and constraints
- Forward-looking and strategic

Focus on areas like: inventory management, staffing, pricing, promotions, menu optimization, 
and capacity planning."""
    
    def _create_forecast_analysis_prompt(
        self,
        forecast_data: Dict,
        historical_data: Dict,
    ) -> str:
        """Create a prompt for forecast analysis."""
        return f"""Analyze the following demand forecast for a restaurant:

Historical Data:
- Average daily revenue: ${historical_data.get('avg_revenue', 0):.2f}
- Average daily customers: {historical_data.get('avg_customers', 0):.0f}
- Revenue trend: {historical_data.get('trend', 'stable')}
- Volatility: {historical_data.get('volatility', 'medium')}

Forecast for next {forecast_data.get('horizon_days', 30)} days:
- Expected average revenue: ${forecast_data.get('avg_forecast', 0):.2f}
- Trend: {forecast_data.get('trend', 'unknown')}
- Confidence: {forecast_data.get('confidence', 'medium')}

Please provide:
1. Key insights from the forecast
2. Potential opportunities and risks
3. Strategic recommendations
4. Suggested metrics to monitor"""
    
    def _create_anomaly_strategy_prompt(
        self,
        anomaly: Dict,
        restaurant_context: Dict,
    ) -> str:
        """Create a prompt for anomaly strategy."""
        return f"""An anomaly has been detected in a restaurant's data:

Anomaly Details:
- Type: {anomaly.get('type', 'unknown')}
- Date: {anomaly.get('date', 'unknown')}
- Severity: {anomaly.get('severity', 0):.1%}
- Description: {anomaly.get('description', '')}
- Metrics: {json.dumps(anomaly.get('metrics', {}), indent=2)}

Restaurant Context:
- Name: {restaurant_context.get('name', 'Unknown')}
- Cuisine: {restaurant_context.get('cuisine', 'Unknown')}
- Capacity: {restaurant_context.get('seating_capacity', 'Unknown')} seats

Please provide:
1. Root cause analysis possibilities
2. Immediate actions to take
3. Short-term strategies (1-2 weeks)
4. Long-term strategies (1-3 months)
5. Metrics to verify effectiveness"""
    
    def _create_action_plan_prompt(
        self,
        forecast: Dict,
        anomalies: List,
        constraints: Optional[Dict],
    ) -> str:
        """Create a prompt for action planning."""
        forecast_str = json.dumps(forecast, indent=2, default=str)
        anomalies_str = json.dumps(anomalies[:5], indent=2, default=str)  # Limit anomalies
        constraints_str = json.dumps(constraints or {}, indent=2)
        
        return f"""Create a comprehensive action plan based on the following data:

Forecast Data:
{forecast_str}

Top Anomalies:
{anomalies_str}

Business Constraints:
{constraints_str}

Please generate an action plan that includes:
1. Prioritized list of actions (next 30 days)
2. Resource requirements (staff, inventory, budget)
3. Implementation timeline
4. Success metrics and KPIs
5. Risk mitigation strategies"""
    
    # Fallback methods for demo mode
    
    @staticmethod
    def _fallback_forecast_analysis(forecast_data: Dict) -> str:
        """Fallback forecast analysis (no LLM)."""
        avg_forecast = forecast_data.get('avg_forecast', 0)
        trend = forecast_data.get('trend', 'stable')
        
        analysis = f"""
# Forecast Analysis Report

## Summary
Expected average daily revenue: ${avg_forecast:.2f}
Forecasted trend: {trend}

## Key Insights
1. Market trend appears to be {trend} over the forecast period
2. Capacity planning should account for expected demand levels
3. Staffing levels should be adjusted based on expected customer volume

## Strategic Recommendations
1. **Inventory Management**: Adjust inventory levels based on forecasted demand
2. **Staffing**: Plan staffing schedules according to predicted customer traffic
3. **Promotions**: Consider targeted promotions if forecast shows decline

## Monitoring Metrics
- Daily revenue vs forecast
- Customer count vs baseline
- Service quality metrics
- Inventory turnover rates
"""
        return analysis.strip()
    
    @staticmethod
    def _fallback_anomaly_strategy(anomaly: Dict) -> str:
        """Fallback anomaly strategy (no LLM)."""
        anom_type = anomaly.get('type', 'unknown')
        severity = anomaly.get('severity', 0)
        
        if 'drop' in anom_type.lower():
            action = "Launch promotional campaign, review menu pricing, analyze customer feedback"
        elif 'spike' in anom_type.lower():
            action = "Ensure adequate staffing and inventory, prepare for service demand"
        else:
            action = "Investigate root cause, monitor closely, review operational metrics"
        
        strategy = f"""
# Strategy for {anom_type.title()}

## Severity: {severity:.1%}

## Recommended Actions
1. {action}
2. Verify data accuracy and check for data quality issues
3. Review recent business activities and external events
4. Monitor similar metrics for corroboration

## Short-term (1-2 weeks)
- Implement quick-win strategies
- Increase monitoring frequency
- Communicate with team

## Long-term (1-3 months)
- Analyze root causes in depth
- Implement systematic improvements
- Update forecasting models with learnings
"""
        return strategy.strip()
    
    @staticmethod
    def _fallback_action_plan(forecast: Dict, anomalies: List) -> str:
        """Fallback action plan (no LLM)."""
        num_anomalies = len(anomalies)
        horizon = forecast.get('horizon_days', 30)
        
        plan = f"""
# 30-Day Action Plan

## Executive Summary
- Forecast horizon: {horizon} days
- Critical anomalies to address: {num_anomalies}

## Priority Actions (Week 1)
1. Establish daily monitoring dashboard
2. Communicate forecast to management team
3. Prepare contingency plans for identified risks
4. Review inventory and staffing levels

## Operational Adjustments (Week 2-4)
1. Implement menu adjustments if needed
2. Launch targeted marketing campaigns
3. Optimize staffing based on demand forecast
4. Enhance customer experience initiatives

## Resource Requirements
- Staff time for monitoring: 1-2 hours daily
- Budget for potential promotions: Variable
- IT resources for dashboard updates: Minimal

## Success Metrics
- Forecast accuracy (MAPE < 15%)
- Revenue vs forecast variance
- Anomaly detection effectiveness
- Action plan completion rate

## Risk Mitigation
- Have contingency budget available
- Maintain communication with teams
- Regular model retraining
- Weekly performance reviews
"""
        return plan.strip()


class RecommendationEngine:
    """Engine for generating structured recommendations."""
    
    def __init__(self):
        self.agent = StrategyAgent()
        self.recommendations = []
    
    def generate_recommendations(
        self,
        analysis_result: Dict,
        anomalies: List[Dict],
        constraints: Optional[Dict] = None,
    ) -> List[Dict]:
        """Generate structured recommendations."""
        recommendations = []
        
        # Forecast-based recommendations
        forecast_recs = self._generate_forecast_recommendations(analysis_result)
        recommendations.extend(forecast_recs)
        
        # Anomaly-based recommendations
        for anomaly in anomalies:
            anomaly_recs = self._generate_anomaly_recommendations(anomaly)
            recommendations.extend(anomaly_recs)
        
        # Sort by priority and save
        recommendations.sort(
            key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.get("priority", "low"), 4)
        )
        
        self.recommendations = recommendations
        return recommendations
    
    @staticmethod
    def _generate_forecast_recommendations(analysis: Dict) -> List[Dict]:
        """Generate forecast-based recommendations."""
        recommendations = []
        forecast = analysis.get('forecast_summary', {})
        
        trend = forecast.get('trend', 'stable')
        if trend == 'declining':
            recommendations.append({
                "title": "Launch Promotional Campaign",
                "description": "Declining demand trend detected. Launch targeted promotions to maintain revenue.",
                "category": "Marketing",
                "priority": "high",
                "expected_impact": "5-10% revenue boost",
            })
        elif trend == 'increasing':
            recommendations.append({
                "title": "Expand Capacity Planning",
                "description": "Increasing demand expected. Prepare staffing and inventory accordingly.",
                "category": "Operations",
                "priority": "high",
                "expected_impact": "Better service quality and customer satisfaction",
            })
        
        return recommendations
    
    @staticmethod
    def _generate_anomaly_recommendations(anomaly: Dict) -> List[Dict]:
        """Generate anomaly-based recommendations."""
        recommendations = []
        anom_type = anomaly.get('type', '')
        severity = anomaly.get('severity', 0)
        
        if 'drop' in anom_type.lower() and severity > 0.7:
            recommendations.append({
                "title": f"Urgent: Address {anom_type.replace('_', ' ').title()}",
                "description": f"A critical demand drop is occurring. Immediate action required.",
                "category": "Urgent",
                "priority": "critical",
                "expected_impact": "Revenue recovery",
            })
        elif 'spike' in anom_type.lower():
            recommendations.append({
                "title": f"Capitalize on {anom_type.replace('_', ' ').title()}",
                "description": f"Demand spike detected. Optimize operations to maximize revenue.",
                "category": "Operations",
                "priority": "high",
                "expected_impact": "Revenue maximization",
            })
        
        return recommendations
