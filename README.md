Enhanced Leveraged ETF Delta Strategy
Overview
This project implements a trading strategy for leveraged ETFs using delta analysis, machine learning (LSTM + RandomForest), and live trading capabilities. It includes a Streamlit dashboard for backtesting and live trading, data fetching with caching, feature engineering, and Prometheus monitoring for production readiness.
Features

Data Fetching: Fetches historical and options data using yfinance and a real-time API (e.g., CBOE).
Feature Engineering: Computes technical indicators (RSI, MACD, VAMI, etc.) and options-based features.
Machine Learning: Hybrid LSTM + RandomForest model for predicting price movements.
Trading: Supports backtesting, live trading, and dry runs with Interactive Brokers (IBKR) integration.
Monitoring: Prometheus metrics for trade latency, portfolio value, and uptime.
Reporting: HTML report generation with a styled Jinja2 template.

Prerequisites

Python: 3.9 or higher
Docker: For containerized deployment
API Keys:
Alpha Vantage API key for WebSocket data streaming.
Real-time options API key (e.g., CBOE) for options data.


Interactive Brokers TWS/Gateway: For live trading (optional for dry runs).

Installation
1. Clone the Repository
Clone this repository to your local machine:
git clone <repository-url>
cd enhanced-leveraged-etf-delta-strategy

2. Create a Virtual Environment
Create and activate a Python virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
Install the required Python packages listed in requirements.txt:
pip install -r requirements.txt

Here’s the requirements.txt content for reference:
streamlit==1.36.0
yfinance==0.2.40
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
tensorflow==2.16.2
joblib==1.4.2
requests==2.32.3
websocket-client==1.8.0
ib-insync==0.9.70
jinja2==3.1.4
prometheus-client==0.20.0

4. Prepare Configuration

Ensure the cache directory exists for data caching:mkdir -p cache


(Optional) Configure Slack for alerts by adding a SLACK_WEBHOOK_URL environment variable in your deployment.

Running the Application
1. Local Execution
Run the Streamlit app locally:
streamlit run app.py

Access the dashboard at http://localhost:8501.
2. Docker Deployment
The application is containerized using Docker and Docker Compose.
Build and Run with Docker Compose
docker-compose up --build

Access the dashboard at http://localhost:8501.
Notes

The Docker setup persists the cache directory and strategy.log file using volumes.
Prometheus metrics are exposed on http://localhost:8000/metrics.

Usage
Dashboard Features

Parameters: Adjust sliders for confidence threshold, VIX slope, OI ratio, and OI change thresholds.
Mode Selection: Choose between:
Backtest: Run a historical backtest and download an HTML report.
Live Trading: Execute trades with IBKR (requires API keys and TWS/Gateway).
Dry Run: Simulate live trading without executing orders.


Data Summary: View available tickers and data range.

Monitoring

Prometheus metrics are available at http://localhost:8000/metrics.
Key metrics include:
trades_total: Total trades executed.
portfolio_value: Current portfolio value.
trade_latency_seconds: Trade execution latency.
data_fetch_duration_seconds: Data fetch duration.
system_uptime_seconds: System uptime.


Alerts are logged to strategy.log. Optionally configure Slack alerts in metrics.py.

Testing
Run the unit tests to verify the system:
pip install pytest
pytest test_strategy.py -v

Project Structure

app.py: Main application orchestrating the strategy and Streamlit UI.
data_loader.py: Fetches and caches historical and options data.
feature_engineer.py: Prepares features and technical indicators.
model.py: Implements the ML pipeline, backtesting, live trading, and reporting.
trader.py: Handles WebSocket streaming and IBKR trading.
metrics.py: Provides Prometheus metrics and alerting.
test_strategy.py: Unit tests for the system.
report_template.html: Jinja2 template for HTML reports.
Dockerfile & docker-compose.yml: Containerization setup.

Future Improvements

PDF Export: Add WeasyPrint to model.py for PDF report generation.
Advanced Monitoring: Set up Grafana dashboards for Prometheus metrics.
Slack Alerts: Integrate real-time Slack notifications with a webhook.
Performance: Optimize feature engineering for larger datasets.

License
This project is licensed under the MIT License.

Generated on May 06, 2025

# AgenticTrade
