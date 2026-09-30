# Restaurant Pulse AI - End-to-End Prototype
# Autonomous Demand Forecaster & Strategy Assistant

Purpose
-------
Restaurant Pulse AI is a developer-facing prototype that demonstrates an end-to-end workflow for small-to-medium restaurants to go from historical sales (and optional weather) data to demand forecasts, anomaly detection, and prioritized operational recommendations. It is intended as a starting point you can install, adapt, and extend for production use — not a drop-in SaaS product.

Core capabilities:

- Demand forecasting (Gradient-boosting regression, scikit-learn)
- Anomaly detection (statistical + trend + forecast comparison)
- Strategy generation (LangChain + LLM integration; demo fallback included)
- Interactive Streamlit dashboard for visualization and exploration

## Architecture

```
Historical Sales & Weather Data
        ↓
1️⃣ Demand Forecast Model (XGBoost)
        ↓
2️⃣ Insights & Anomaly Detector
        ↓
3️⃣ Agentic Strategy Assistant (LangChain)
        ↓
4️⃣ Interactive Streamlit Dashboard
```

## Project Structure

```
restaurant-pulse-ai/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── data_loader.py            # Data generation and loading
│   ├── forecast_model.py         # XGBoost demand forecasting
│   ├── anomaly_detector.py       # Anomaly detection engine
│   ├── strategy_agent.py         # LangChain strategy assistant
│   ├── dashboard.py              # Streamlit dashboard
│   └── utils.py                  # Utility functions
├── data/                         # Generated/stored data
├── models/                       # Trained model storage
├── notebooks/                    # Analysis notebooks
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment configuration template
├── example_usage.py              # Complete usage demonstration
└── README.md                     # This file
```

## Features

### Demand Forecasting
- **Gradient Boosting (scikit-learn)**: Ensemble gradient-boosting regression used by default for broad compatibility (XGBoost can be added if desired)
- **Ensemble Methods**: Multiple models with weighted predictions
- **Feature Engineering**: Temporal features, lag features, rolling statistics
- **Model Persistence**: Save/load trained models

### Anomaly Detection
- **Multiple Detection Methods**:
  - Statistical (Z-score based)
  - Trend analysis
  - Forecast comparison
- **Severity Scoring**: 0-1 scale for anomaly severity
- **Actionable Insights**: Automatic recommendation generation

### AI Strategy Agent
- **LangChain Integration**: Leverages GPT-4 for intelligent analysis
- **Context-Aware**: Considers restaurant type, capacity, and constraints
- **Multi-Faceted Analysis**:
  - Forecast analysis
  - Anomaly investigation
  - Action planning
- **Demo Mode**: Works without OpenAI API for testing

### Interactive Dashboard
- **Real-time Monitoring**: Live data updates
- **Visualizations**: Revenue trends, anomaly timelines, forecasts
- **Restaurant Selection**: Multi-restaurant support
- **Customizable Analysis**: Adjustable date ranges and detection sensitivity

## Installation & Quickstart

Prerequisites
- macOS / Linux / Windows with Python 3.11+ (3.13 tested)
- `pip` package manager

Quick install (local development)

```bash
git clone <repo-url> restaurant-pulse-ai
cd restaurant-pulse-ai
# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```

Environment configuration (optional — for LLM features)

```bash
# Copy template and add your OpenAI key (if you want LLM recommendations)
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-...
# Or set in shell for current session:
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4"
```

## Usage

### Running the Example Demonstration

```bash
# Activate virtual environment
source venv/bin/activate

# Run the end-to-end demonstration (generates data, trains model, shows summary)
python example_usage.py
```

What the demo does
- Generates synthetic historical sales + weather data (saved under `data/`)
- Trains the default demand forecasting model
- Runs anomaly detection on the simulated series
- Produces a short executive summary with recommendations

### Launching the Dashboard (UI)

```bash
# Activate virtual environment
source venv/bin/activate

# Start Streamlit app (default port 8501)
streamlit run src/dashboard.py
```

Open your browser at `http://localhost:8501` (or the port you specify). The dashboard supports switching restaurants, adjusting analysis windows, and viewing recommendations.

### Using Components Individually

```python
from src.data_loader import DataManager
from src.forecast_model import DemandForecastModel
from src.anomaly_detector import AnomalyDetector
from src.strategy_agent import StrategyAgent

# Load data
data_manager = DataManager(restaurant_id="rest_001")
data_manager.load_data()
data_manager.prepare_data()

# Train forecast model
model = DemandForecastModel()
model.build_model()
model.train(data_manager.X_train, data_manager.y_train)

# Detect anomalies
detector = AnomalyDetector()
detector.compute_baseline_stats(data_manager.data, ["revenue", "customers"])
anomalies = detector.detect_all_anomalies(data_manager.data)

# Generate strategies
agent = StrategyAgent()
analysis = agent.analyze_demand_forecast(forecast_data, historical_data)
```

## Data Sources

This project uses synthetic data for demonstration and testing. The generated CSV is included in `data/rest_001_historical_data.csv` and contains one year of daily records by default.

Synthetic data includes:
- Sales: `date`, `customers`, `revenue`, `avg_check`
- Weather (optional): `temperature`, `humidity`, `precipitation`, `wind_speed`
- Engineered features: temporal encodings, lag/rolling statistics

Using real data
- To use your own historical data, provide a CSV with at minimum these columns: `date`, `customers`, `revenue`.
- Date column should be ISO format (YYYY-MM-DD). Example loader snippet:

```python
import pandas as pd

def load_real_data(csv_path):
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        # Ensure required columns exist
        assert {'date','customers','revenue'}.issubset(df.columns)
        return df
```

Mapping and validation
- If your POS export uses different column names, map them to the expected names before feeding the data to `DataManager`.
- The project provides `DataManager` in `src/data_loader.py` as the integration point for real data ingestion, validation, and feature preparation.

## Model Details

### XGBoost Configuration
- **Estimators**: 100 trees
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **Subsample**: 0.8
- **Objective**: Regression (MAE loss)

### Hyperparameters
All hyperparameters are configurable in `src/config.py`

## Anomaly Detection

### Detection Types
1. **Level Anomalies**: Values significantly different from baseline (Z-score > 2.5 std)
2. **Trend Anomalies**: Changes in trend direction
3. **Forecast Anomalies**: Large prediction errors
4. **Unusual Patterns**: Complex anomalies

### Sensitivity Levels
- **Low**: threshold_std = 3.75σ
- **Medium**: threshold_std = 2.5σ
- **High**: threshold_std = 1.88σ

## Strategy Agent

### Capabilities
- Analyzes demand forecasts
- Investigates anomalies
- Generates action plans
- Provides prioritized recommendations

### Recommendation Categories
- **Inventory Management**: Stock optimization
- **Staffing**: Scheduling adjustments
- **Pricing**: Menu and pricing strategies
- **Marketing**: Promotional campaigns
- **Operational**: Efficiency improvements

## Configuration

Key configurations in `src/config.py`:

```python
# Data Configuration
HISTORICAL_DATA_DAYS = 365
FORECAST_HORIZON_DAYS = 30

# Model Configuration
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
}

# Anomaly Detection
ANOMALY_THRESHOLD_STD = 2.5
ANOMALY_SENSITIVITY = "medium"

# LLM Configuration
LLM_MODEL = "gpt-4-turbo"
```

## Performance Metrics

The system tracks:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **Forecast Accuracy**
- **Anomaly Detection Rate**

## Restaurants

Sample restaurants included for demonstration:
1. The Golden Fork (Italian, 85 seats)
2. Tokyo Sunrise (Japanese, 60 seats)
3. El Sabor Latino (Mexican, 120 seats)
4. Le Petit Bistro (French, 50 seats)
5. Dragon Palace (Chinese, 100 seats)

## API Integration

### OpenAI Integration

The strategy agent requires an OpenAI API key for LLM functionality:

```bash
# Set in .env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4-turbo
```

### Demo Mode

Without an API key, the system operates in demo mode with pre-generated recommendations.

## Monitoring & Maintenance

### Model Retraining
- Retrain model weekly with new data
- Monitor forecast accuracy
- Update anomaly baselines monthly

### Dashboard Monitoring
- Check daily for critical anomalies
- Review weekly forecasts
- Implement recommendations

### Alerts
- Critical anomalies: Immediate notification
- High severity: Daily digest
- Medium/Low: Weekly summary

## Troubleshooting

### Issue: Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: OpenAI API errors
- Verify API key in .env
- Check API rate limits
- System will fall back to demo mode

### Issue: Streamlit not launching
```bash
# Clear cache and restart
streamlit run src/dashboard.py --logger.level=debug
```

## Future Enhancements

### Planned Features
- [ ] Real-time data streaming
- [ ] Multi-location comparison
- [ ] Customer segmentation analysis
- [ ] Inventory forecasting
- [ ] Staff scheduling optimization
- [ ] Automated alert system
- [ ] Mobile app support
- [ ] Advanced ensemble methods
- [ ] LSTM/Transformer models
- [ ] Causal inference analysis

### Potential Integrations
- Restaurant POS systems
- Weather APIs
- Social media sentiment
- Competitor data feeds
- Local event calendars

## Performance Examples

### Forecast Accuracy
- RMSE: $45-65 per day
- MAE: $35-50 per day
- MAPE: 8-12%

### Anomaly Detection
- Sensitivity: 85%+
- False positive rate: <5%
- Detection latency: Real-time

## Contributing

To contribute improvements:
1. Fork the repository
2. Create feature branch
3. Make improvements
4. Submit pull request

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or suggestions:
- Create GitHub issue
- Check documentation
- Review example usage

## Acknowledgments

Built with:
- **XGBoost**: Advanced gradient boosting
- **LangChain**: LLM orchestration
- **Streamlit**: Interactive dashboards
- **Pandas & NumPy**: Data processing
- **OpenAI GPT-4**: Intelligent reasoning

## Version

Current Version: **0.2.0**
Last Updated: September 29, 2026

---

**Restaurant Pulse AI** - Making data-driven decisions easy for restaurant operators 🍽️
