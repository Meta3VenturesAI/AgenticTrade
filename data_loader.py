import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from typing import Tuple, List, Dict, Any
from joblib import Memory
import requests
import time
from threading import Lock

# Setup logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('strategy.log', mode='a', encoding='utf-8')])
logger = logging.getLogger()

# Initialize disk-backed cache
cache_dir = './cache'
memory = Memory(cache_dir, verbose=0)
cache_lock = Lock()

# Configuration
CONFIG = {
    "min_data_length": 100,
    "options_cache_duration": 300,
    "options_cache_clear_interval": 3600,
    "real_time_options_api_url": "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}?apiKey={api_key}"
}

options_cache = {}
options_cache_timestamps = {}
last_cache_clear_time = time.time()

def fetch_single_ticker(ticker: str, start_date: str, end_date: str, interval: str = "1d", min_data_length: int = None) -> pd.DataFrame:
    """Fetch data for a single ticker with retry logic.
    
    Args:
        ticker (str): Ticker symbol to fetch.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        interval (str): Data interval (e.g., "1d").
        min_data_length (int, optional): Minimum data length required. Defaults to CONFIG["min_data_length"].
    
    Returns:
        pd.DataFrame: Fetched data for the ticker.
    """
    min_length = min_data_length if min_data_length is not None else CONFIG["min_data_length"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download_ticker():
        try:
            logger.info(f"Fetching data for {ticker} from {start_date} to {end_date} with interval {interval}")
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, threads=False)
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                raise ValueError(f"No data returned for {ticker}")
            if len(data) < min_length:
                logger.warning(f"Insufficient data for {ticker}: {len(data)} periods (required: {min_length})")
                raise ValueError(f"Insufficient data for {ticker}: {len(data)} periods")
            logger.info(f"Successfully fetched {len(data)} periods for {ticker}")
            return data
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            raise
    
    try:
        current_start_date = pd.to_datetime(start_date)
        current_end_date = pd.to_datetime(end_date)
        while (current_end_date - current_start_date).days > 30:
            try:
                data = download_ticker()
                return data
            except Exception as e:
                logger.warning(f"Adjusting date range for {ticker}: {e}")
                current_start_date += pd.Timedelta(days=30)
        return download_ticker()
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()

def fetch_data(tickers: List[str], start_date: str, end_date: str, interval: str = "1d", min_data_length: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Fetch historical data for multiple tickers in parallel.
    
    Args:
        tickers (List[str]): List of ticker symbols.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        interval (str): Data interval (e.g., "1d").
        min_data_length (int, optional): Minimum data length required. Defaults to CONFIG["min_data_length"].
    
    Returns:
        Tuple containing close, open, high, low, returns data, and successful tickers.
    """
    logger.info(f"Fetching data for {tickers}")
    successful_tickers = []
    all_data = {}

    # Parallel fetch
    with ThreadPoolExecutor() as executor:
        future_to_ticker = {executor.submit(fetch_single_ticker, ticker, start_date, end_date, interval, min_data_length): ticker for ticker in tickers}
        for future in future_to_ticker:
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if not data.empty:
                    all_data[ticker] = data
                    successful_tickers.append(ticker)
                    logger.info(f"Successfully fetched data for {ticker}")
                else:
                    logger.warning(f"No data fetched for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")

    if not successful_tickers:
        # Fallback: Try fetching data for a minimal subset of tickers
        fallback_tickers = ['SPY', '^VIX']  # More likely to have data
        logger.info(f"Fallback: Trying to fetch data for {fallback_tickers}")
        for ticker in fallback_tickers:
            try:
                data = fetch_single_ticker(ticker, start_date, end_date, interval, min_data_length)
                if not data.empty:
                    all_data[ticker] = data
                    successful_tickers.append(ticker)
                    logger.info(f"Fallback: Successfully fetched data for {ticker}")
            except Exception as e:
                logger.error(f"Fallback: Error fetching {ticker}: {e}")

        if not successful_tickers:
            logger.error("No tickers returned sufficient data, even after fallback")
            raise ValueError("No tickers returned sufficient data")

    # Combine and align data
    combined_data = pd.concat([all_data[ticker] for ticker in successful_tickers], axis=1, keys=successful_tickers)
    original_len = len(combined_data)
    
    close_data = combined_data.xs('Adj Close', axis=1, level=1)
    open_data = combined_data.xs('Open', axis=1, level=1)
    high_data = combined_data.xs('High', axis=1, level=1)
    low_data = combined_data.xs('Low', axis=1, level=1)

    # Log index alignment losses
    aligned_len = len(close_data)
    if aligned_len < original_len:
        logger.warning(f"Index alignment dropped {original_len - aligned_len} rows")

    # Handle missing values
    missing_values = close_data.isna().sum().sum()
    close_data = close_data.interpolate(method='linear').dropna()
    open_data = open_data.interpolate(method='linear').dropna()
    high_data = high_data.interpolate(method='linear').dropna()
    low_data = low_data.interpolate(method='linear').dropna()
    returns = close_data.pct_change().dropna()

    logger.info(f"Data fetched for {successful_tickers}: {len(close_data)} periods, {missing_values} missing values interpolated")
    return close_data, open_data, high_data, low_data, returns, successful_tickers

@memory.cache
def fetch_options_data_cached(ticker: str, date: str, real_time_api_key: str) -> pd.DataFrame:
    """Fetch options data with disk-backed caching.
    
    Args:
        ticker (str): Ticker symbol.
        date (str): Date in YYYY-MM-DD format.
        real_time_api_key (str): API key for real-time options data.
    
    Returns:
        pd.DataFrame: Options data.
    """
    global last_cache_clear_time
    current_time = time.time()

    # Clear in-memory cache periodically
    with cache_lock:
        if (current_time - last_cache_clear_time) > CONFIG["options_cache_clear_interval"]:
            options_cache.clear()
            options_cache_timestamps.clear()
            last_cache_clear_time = current_time
            logger.info("Cleared options data cache")

    if not real_time_api_key:
        logger.warning(f"No real-time API key provided for {ticker}. Skipping options data fetch.")
        return pd.DataFrame()

    try:
        logger.info(f"Fetching options data for {ticker} on {date}")
        url = CONFIG["real_time_options_api_url"].format(ticker=ticker, date=date, api_key=real_time_api_key)
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            options_data = pd.DataFrame(response.json().get('results', []))
            if not options_data.empty:
                # Map Polygon.io response to expected format
                options_data = options_data.rename(columns={
                    'open': 'openInterest',  # Adjust based on actual Polygon.io response fields
                    'volume': 'volume'
                })
                return options_data
        # Fallback to yfinance if Polygon.io fails
        stock = yf.Ticker(ticker)
        options = stock.option_chain(date=date)
        if options is None or options.calls.empty or options.puts.empty:
            logger.warning(f"No options data available for {ticker} on {date}")
            return pd.DataFrame()
        return pd.concat([options.calls, options.puts]).sort_values('strike')
    except requests.RequestException as e:
        logger.error(f"Options API request failed for {ticker}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Options data fetch error for {ticker}: {e}")
        return pd.DataFrame()

def fetch_options_data(ticker: str, date: str, real_time_api_key: str) -> pd.DataFrame:
    """Fetch options data with in-memory and disk-backed caching.
    
    Args:
        ticker (str): Ticker symbol.
        date (str): Date in YYYY-MM-DD format.
        real_time_api_key (str): API key for real-time options data.
    
    Returns:
        pd.DataFrame: Options data.
    """
    cache_key = f"{ticker}_{date}"
    current_time = time.time()

    with cache_lock:
        if cache_key in options_cache and (current_time - options_cache_timestamps[cache_key]) < CONFIG["options_cache_duration"]:
            logger.info(f"Using in-memory cached options data for {ticker} on {date}")
            return options_cache[cache_key]

    # Use disk-backed cache
    result = fetch_options_data_cached(ticker, date, real_time_api_key)
    with cache_lock:
        options_cache[cache_key] = result
        options_cache_timestamps[cache_key] = current_time
    return result
