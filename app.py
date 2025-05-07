import streamlit as st
import pandas as pd
import numpy as np
import logging
import queue
import threading
from datetime import datetime, timedelta
import os
from typing import List, Dict, Tuple, Any
import multiprocessing as mp
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('strategy.log', mode='a', encoding='utf-8')])
logger = logging.getLogger()

# Import data loader and other modules
from data_loader import fetch_data, fetch_options_data
from feature_engineer import prepare_features
from model import backtest_strategy, live_trading, SELECTED_FEATURES

# Configuration and Context
@dataclass
class Context:
    tickers: List[Dict[str, Any]]
    vix_ticker: str
    start_date: str = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')  # Last 2 years
    end_date: str = datetime.now().strftime('%Y-%m-%d')  # Current date
    min_data_length: int = 100
    transaction_cost: float = 0.001
    default_confidence_threshold: float = 0.6
    default_vix_slope_threshold: float = 0.5
    default_oi_ratio_threshold: float = 0.2
    default_oi_change_threshold: float = 0.1
    intraday_threshold: float = 0.02
    spike_threshold: float = 0.02
    trailing_stop_percent: float = 0.05
    max_drawdown_limit: float = 0.1
    alpha_vantage_ws_url: str = "wss://ws.eodhistoricaldata.com/ws/us?api_token={}"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    ws_batch_size: int = 50
    order_batch_size: int = 10
    retrain_interval: int = 100
    retrain_precision_threshold: float = 0.5
    max_data_points: int = 50
    max_total_data_points: int = 1000
    options_cache_duration: int = 300
    options_cache_clear_interval: int = 3600

    def __post_init__(self):
        if not self.tickers:
            raise ValueError("Tickers list cannot be empty")
        if self.min_data_length < 100:
            raise ValueError("min_data_length must be at least 100")

# Initialize context with runtime input
def get_context() -> Context:
    st.sidebar.header("Date Configuration")
    start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'))
    end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", datetime.now().strftime('%Y-%m-%d'))
    return Context(
        tickers=[
            {'leveraged_etf': 'TQQQ', 'base_index': 'QQQ', 'leverage_ratio': 3},
            {'leveraged_etf': 'UPRO', 'base_index': 'SPY', 'leverage_ratio': 3},
            {'leveraged_etf': 'SQQQ', 'base_index': 'QQQ', 'leverage_ratio': -3}
        ],
        vix_ticker='^VIX',
        start_date=start_date,
        end_date=end_date
    )

def calculate_atr(high_data, low_data, close_data, period=14):
    """Calculate Average True Range for dynamic position sizing.
    
    Args:
        high_data (pd.Series): High price data.
        low_data (pd.Series): Low price data.
        close_data (pd.Series): Close price data.
        period (int): Lookback period for ATR calculation.
    
    Returns:
        pd.Series: Average True Range values.
    """
    high_low = high_data - low_data
    high_close = (high_data - close_data.shift()).abs()
    low_close = (low_data - close_data.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

@st.cache_data
def run_strategy(context: Context) -> Tuple[pd.DataFrame, pd.DataFrame, List[List[str]], Any, Any, Any]:
    """Run the strategy pipeline with caching.
    
    Args:
        context (Context): Configuration and state object.
    
    Returns:
        Tuple containing data, trade log, metrics, and models.
    """
    try:
        close_data, open_data, high_data, low_data, returns, successful_tickers = fetch_data(
            [pair['leveraged_etf'] for pair in context.tickers] +
            [pair['base_index'] for pair in context.tickers] + [context.vix_ticker],
            start_date=context.start_date,
            end_date=context.end_date,
            interval='1d'  # Use daily interval for historical data
        )
        vix_data = close_data[[context.vix_ticker]].copy()
        volatility = close_data.pct_change().rolling(window=10).std()
        options_date = close_data.index[-1].strftime('%Y-%m-%d')
        real_time_options_api_key = os.getenv("REAL_TIME_OPTIONS_API_KEY", "")
        options_data = fetch_options_data(context.tickers[0]['base_index'], options_date, real_time_options_api_key)

        # Prepare features
        delta = pd.Series(0, index=close_data.index)  # Simplified for demo
        features = prepare_features(close_data, returns, delta, context.tickers[0]['base_index'], vix_data, volatility, options_data)

        # Backtest parameters
        best_params = {'seq_length': 5}
        portfolio_returns, trade_log, metrics, lstm_model, rf_pipeline, scaler = backtest_strategy(
            close_data, open_data, high_data, low_data, features, context.tickers, SELECTED_FEATURES,
            best_params, context.default_confidence_threshold, context.default_vix_slope_threshold,
            context.default_oi_ratio_threshold, context.default_oi_change_threshold, None
        )

        return close_data, trade_log, metrics, lstm_model, rf_pipeline, scaler
    except Exception as e:
        logger.error(f"Strategy execution failed: {e}")
        st.error(f"Error running strategy: {e}")
        # Return dummy data to allow UI to render
        return pd.DataFrame(index=pd.date_range(context.start_date, context.end_date)), pd.DataFrame(), [], None, None, None

def live_trading_process(lstm_model, rf_model, scaler, tickers, max_data_points, confidence_threshold,
                         vix_slope_threshold, oi_ratio_threshold, oi_change_threshold, data_queue, trade_queue,
                         alpha_vantage_api_key, real_time_options_api_key, stop_event, calculate_atr, max_drawdown_limit):
    """Run the live trading logic in a separate process."""
    try:
        # Import trader modules within the process to avoid event loop issues
        from trader import AlphaVantageWS, IBKRTrader
        import asyncio

        logger.info(f"Starting live trading process with Alpha Vantage API key: {alpha_vantage_api_key[:4]}... (masked)")
        logger.info(f"Using Polygon.io API key for options data: {real_time_options_api_key[:4]}... (masked)")

        # Create an event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Initialize WebSocket client
        logger.info("Initializing Alpha Vantage WebSocket client")
        ws_client = AlphaVantageWS(alpha_vantage_api_key, tickers, data_queue)
        ws_client.start()
        logger.info("Alpha Vantage WebSocket client started")

        # Initialize IBKR trader
        logger.info("Initializing IBKR trader")
        ibkr_trader = IBKRTrader(data_queue, trade_queue, dry_run=False)
        loop.run_until_complete(ibkr_trader.connect())
        logger.info("IBKR trader connected")

        # Run live trading
        logger.info("Starting live trading")
        loop.run_until_complete(live_trading(
            lstm_model, rf_model, scaler, tickers, max_data_points,
            confidence_threshold, vix_slope_threshold, oi_ratio_threshold, oi_change_threshold,
            data_queue, trade_queue, ws_client, real_time_options_api_key, stop_event,
            calculate_atr, max_drawdown_limit
        ))

        # Cleanup
        logger.info("Cleaning up live trading process")
        ws_client.stop()
        loop.run_until_complete(ibkr_trader.disconnect())
        loop.close()
        logger.info("Live trading process cleanup complete")
    except Exception as e:
        logger.error(f"Live trading process failed: {e}")
        stop_event.set()

def run_streamlit() -> None:
    """Run the Streamlit dashboard for the trading strategy.
    
    Manages UI, live trading, and resource cleanup.
    """
    context = get_context()
    st.title("Enhanced Leveraged ETF Delta Strategy Dashboard")

    # Run strategy
    close_data, test_log_df, test_metrics, lstm_model, rf_model, scaler = run_strategy(context)

    st.header("Parameters")
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, context.default_confidence_threshold, 0.05)
    vix_slope_threshold = st.slider("VIX Slope Threshold", 0.1, 1.0, context.default_vix_slope_threshold, 0.1)
    oi_ratio_threshold = st.slider("OI Ratio Threshold", 0.0, 0.5, context.default_oi_ratio_threshold, 0.05)
    oi_change_threshold = st.slider("OI Change Threshold", 0.0, 0.5, context.default_oi_change_threshold, 0.05)
    mode = st.selectbox("Mode", ["Backtest", "Live Trading", "Dry Run"])
    alpha_vantage_api_key = st.text_input("Alpha Vantage API Key (for Live Trading)", value=os.getenv("ALPHA_VANTAGE_API_KEY", ""), type="password")
    real_time_options_api_key = st.text_input("Real-Time Options API Key (e.g., Polygon.io)", value=os.getenv("REAL_TIME_OPTIONS_API_KEY", ""), type="password")

    st.header("Data Summary")
    if not close_data.empty:
        st.write(f"**Tickers Available**: {', '.join(close_data.columns)}")
        st.write(f"**Date Range**: {context.start_date} to {context.end_date}")
        st.write(f"**Data Points**: {len(close_data)} periods")
    else:
        st.warning("No data available. Adjust the date range or check logs for errors.")

    if mode == "Backtest":
        if not close_data.empty:
            st.header("Backtest Results")
            st.dataframe(test_log_df)
            st.header("Performance Metrics")
            st.table(test_metrics)
            html_report = generate_report(close_data, test_log_df, test_metrics, context.start_date, context.end_date)
            st.download_button("Download HTML Report", data=html_report, file_name="strategy_report.html", mime="text/html")
        else:
            st.error("Cannot run backtest: No data available.")

    else:  # Live Trading or Dry Run
        if not alpha_vantage_api_key or not real_time_options_api_key:
            st.error("Please provide both API keys for live trading or dry run.")
            return

        data_queue = mp.Queue()
        trade_queue = mp.Queue()
        stop_event = mp.Event()
        tickers = [pair['leveraged_etf'] for pair in context.tickers] + [pair['base_index'] for pair in context.tickers] + [context.vix_ticker]

        # Start live trading in a separate process
        live_process = mp.Process(
            target=live_trading_process,
            args=(lstm_model, rf_model, scaler, tickers, context.max_data_points,
                  confidence_threshold, vix_slope_threshold, oi_ratio_threshold, oi_change_threshold,
                  data_queue, trade_queue, alpha_vantage_api_key, real_time_options_api_key,
                  stop_event, calculate_atr, context.max_drawdown_limit),
            daemon=True
        )
        live_process.start()

        st.header("Live Trading")
        st.write("Starting live trading...")

        stop_button = st.button("Stop Live Trading")
        while live_process.is_alive() and not stop_event.is_set():
            if stop_button:
                stop_event.set()
                break
            try:
                # Display incoming data (simplified for demo)
                data = data_queue.get_nowait()
                st.write(f"Received data: {data}")
            except queue.Empty:
                pass
            time.sleep(0.1)

        # Cleanup
        stop_event.set()
        live_process.join(timeout=5)
        if live_process.is_alive():
            live_process.terminate()
            logger.warning("Live trading process did not terminate gracefully; terminated")
        st.success("Live trading session ended. Resources cleaned up.")

if __name__ == "__main__":
    run_streamlit()
