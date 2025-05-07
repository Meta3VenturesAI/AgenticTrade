from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time
import logging
import requests
from threading import Lock
import os

# Setup logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('strategy.log', mode='a', encoding='utf-8')])
logger = logging.getLogger()

# Get Slack webhook URL from environment variable or use placeholder
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX')
if SLACK_WEBHOOK_URL == 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX':
    logger.warning("Using placeholder Slack webhook URL. Replace with your real webhook URL via SLACK_WEBHOOK_URL environment variable.")

# Initialize Prometheus metrics
metric_lock = Lock()

# Counter for total trades executed
trades_total = Counter('trades_total', 'Total number of trades executed')

# Gauge for current portfolio value
portfolio_value = Gauge('portfolio_value', 'Current portfolio value')

# Histogram for trade execution latency (in seconds)
trade_latency = Histogram('trade_latency_seconds', 'Latency of trade execution')

# Histogram for data fetch duration (in seconds)
data_fetch_duration = Histogram('data_fetch_duration_seconds', 'Duration of data fetch operations')

# Gauge for system uptime (in seconds)
uptime = Gauge('system_uptime_seconds', 'System uptime since start')

# Start HTTP server for Prometheus to scrape metrics
def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus metrics HTTP server.
    
    Args:
        port (int): Port to expose metrics on.
    """
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

# Record trade execution
def record_trade(latency: float) -> None:
    """Record a trade execution with latency.
    
    Args:
        latency (float): Latency of the trade in seconds.
    """
    with metric_lock:
        trades_total.inc()
        trade_latency.observe(latency)
        logger.info(f"Recorded trade with latency {latency}s")

# Update portfolio value
def update_portfolio_value(value: float) -> None:
    """Update the current portfolio value.
    
    Args:
        value (float): Current portfolio value.
    """
    with metric_lock:
        portfolio_value.set(value)
        logger.info(f"Updated portfolio value to {value}")

# Record data fetch duration
def record_data_fetch(duration: float) -> None:
    """Record the duration of a data fetch operation.
    
    Args:
        duration (float): Duration of the fetch in seconds.
    """
    with metric_lock:
        data_fetch_duration.observe(duration)
        logger.info(f"Recorded data fetch duration {duration}s")

# Update system uptime
start_time = time.time()
def update_uptime() -> None:
    """Update the system uptime metric."""
    with metric_lock:
        uptime.set(time.time() - start_time)
        logger.debug("Updated system uptime")

# Log and alert on critical events
def log_alert(message: str, level: str = "WARNING") -> None:
    """Log an alert message and send to Slack if configured.
    
    Args:
        message (str): Alert message.
        level (str): Logging level (e.g., "WARNING", "ERROR").
    """
    if level == "ERROR":
        logger.error(message)
    else:
        logger.warning(message)
    
    # Send to Slack
    try:
        payload = {"text": f"Alert: {message} at {time.strftime('%Y-%m-%d %H:%M:%S')}"}
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code != 200:
            logger.error(f"Failed to send Slack alert: {response.text}")
        else:
            logger.info("Slack alert sent successfully")
    except requests.RequestException as e:
        logger.error(f"Slack alert request failed: {e}")

# Example usage (to be integrated into other modules)
if __name__ == "__main__":
    start_metrics_server(8000)
    while True:
        update_uptime()
        time.sleep(1)
