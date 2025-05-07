import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Tuple, List, Union
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('strategy.log', mode='a', encoding='utf-8')])
logger = logging.getLogger()

def calculate_delta(leveraged_etf: pd.DataFrame, underlying_index: pd.DataFrame, leverage_ratio: List[int]) -> pd.Series:
    """Calculate the delta between leveraged ETF and underlying index returns.
    
    Args:
        leveraged_etf (pd.DataFrame): Leveraged ETF price data.
        underlying_index (pd.DataFrame): Underlying index price data.
        leverage_ratio (List[int]): Leverage ratios.
    
    Returns:
        pd.Series: Delta values.
    """
    try:
        logger.info("Calculating delta")
        # Compute returns for each ticker
        leveraged_returns = leveraged_etf.iloc[:, 0].pct_change()  # First column
        underlying_returns = underlying_index.iloc[:, 0].pct_change()  # First column
        expected_returns = underlying_returns * leverage_ratio[0]
        delta = (leveraged_returns - expected_returns).fillna(0)
        return delta.reindex(leveraged_etf.index).fillna(0)
    except Exception as e:
        logger.error(f"Delta calculation error: {e}")
        raise

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) for a given price series.
    
    Args:
        prices (pd.Series): Price data.
        period (int): RSI period.
    
    Returns:
        pd.Series: RSI values.
    """
    try:
        logger.info("Computing RSI")
        delta = prices.diff()
        gains = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        losses = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = np.where(losses == 0, np.inf, gains / losses)
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi, index=prices.index).bfill().fillna(0)
    except Exception as e:
        logger.error(f"RSI computation error: {e}")
        raise

def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Compute the Moving Average Convergence Divergence (MACD) and signal line.
    
    Args:
        prices (pd.Series): Price data.
        fast (int): Fast EMA period.
        slow (int): Slow EMA period.
        signal (int): Signal line period.
    
    Returns:
        Tuple[pd.Series, pd.Series]: MACD and signal line.
    """
    try:
        logger.info("Computing MACD")
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    except Exception as e:
        logger.error(f"MACD computation error: {e}")
        raise

def compute_bollinger_width(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Compute the Bollinger Bands Width for a given price series.
    
    Args:
        prices (pd.Series): Price data.
        period (int): Period for the moving average and standard deviation.
        num_std (float): Number of standard deviations for the bands.
    
    Returns:
        pd.Series: Bollinger Bands Width values (normalized by the moving average).
    """
    try:
        logger.info("Computing Bollinger Bands Width")
        rolling_mean = prices.rolling(window=period, min_periods=1).mean()
        rolling_std = prices.rolling(window=period, min_periods=1).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        bb_width = (upper_band - lower_band) / rolling_mean
        return bb_width.fillna(0)
    except Exception as e:
        logger.error(f"Bollinger Bands Width computation error: {e}")
        raise

def detect_spikes(high: pd.DataFrame, low: pd.DataFrame, ticker: str, spike_threshold: float = 0.05) -> pd.Series:
    """Detect price spikes using high-low range.
    
    Args:
        high (pd.DataFrame): High price data.
        low (pd.DataFrame): Low price data.
        ticker (str): Ticker symbol.
        spike_threshold (float): Threshold for detecting spikes.
    
    Returns:
        pd.Series: Spike signals (1 for upward, -1 for downward, 0 otherwise).
    """
    try:
        logger.info(f"Detecting spikes for {ticker}")
        range_data = (high[ticker] - low[ticker]) / low[ticker]
        spikes = np.where(range_data > spike_threshold, 1,
                         np.where(range_data < -spike_threshold, -1, 0))
        return pd.Series(spikes, index=high.index).fillna(0)
    except Exception as e:
        logger.error(f"Spike detection error: {e}")
        raise

def intraday_momentum_signal(open_prices: pd.DataFrame, close_prices: pd.DataFrame, ticker: str) -> pd.Series:
    """Generate intraday momentum signals.
    
    Args:
        open_prices (pd.DataFrame): Open price data.
        close_prices (pd.DataFrame): Close price data.
        ticker (str): Ticker symbol.
    
    Returns:
        pd.Series: Momentum signals (1 for bullish, -1 for bearish, 0 otherwise).
    """
    try:
        logger.info(f"Generating intraday momentum signals for {ticker}")
        returns = (close_prices[ticker] - open_prices[ticker]) / open_prices[ticker]
        signals = np.where(returns > 0.01, 1,
                          np.where(returns < -0.01, -1, 0))
        return pd.Series(signals, index=close_prices.index).fillna(0)
    except Exception as e:
        logger.error(f"Intraday momentum signal error: {e}")
        raise

@lru_cache(maxsize=128)
def calculate_vami(returns: tuple, volatility: tuple) -> pd.Series:
    """Calculate Volatility Adjusted Momentum Index (VAMI).
    
    Args:
        returns (tuple): Returns data as a tuple for hashable type.
        volatility (tuple): Volatility data as a tuple for hashable type.
    
    Returns:
        pd.Series: VAMI values.
    """
    try:
        logger.info("Calculating VAMI")
        returns_series = pd.Series(returns, index=pd.date_range('2023-01-01', periods=len(returns)))
        volatility_series = pd.Series(volatility, index=pd.date_range('2023-01-01', periods=len(volatility)))
        vami = (returns_series / (volatility_series + 1e-6)).cumsum()
        return vami
    except Exception as e:
        logger.error(f"VAMI calculation error: {e}")
        raise

def prepare_features(close_data: pd.DataFrame, returns: pd.DataFrame, delta: pd.Series, ticker: str,
                    vix_data: pd.DataFrame, volatility: pd.DataFrame,
                    options_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for the model, including Bollinger Bands Width.
    
    Args:
        close_data (pd.DataFrame): Close price data.
        returns (pd.DataFrame): Returns data.
        delta (pd.Series): Delta values.
        ticker (str): Ticker symbol.
        vix_data (pd.DataFrame): VIX data.
        volatility (pd.DataFrame): Volatility data.
        options_data (pd.DataFrame): Options data.
    
    Returns:
        pd.DataFrame: Feature DataFrame.
    """
    try:
        logger.info(f"Preparing features for {ticker}")
        delta_series = delta.reindex(close_data.index).fillna(0)
        rsi = compute_rsi(close_data[ticker])
        macd, macd_signal = compute_macd(close_data[ticker])
        bb_width = compute_bollinger_width(close_data[ticker])
        vami = calculate_vami(tuple(returns[ticker].values), tuple(volatility[ticker].values))
        momentum = returns[ticker].rolling(window=5).mean().fillna(0)

        vix_slope = vix_data["^VIX"].diff().rolling(window=5).mean().reindex(close_data.index, method='ffill').fillna(0)

        if not options_data.empty:
            oi_ratio = options_data['openInterest'].rolling(window=5).mean().reindex(close_data.index, method='ffill').fillna(0)
            oi_change = options_data['openInterest'].diff().rolling(window=5).mean().reindex(close_data.index, method='ffill').fillna(0)
            implied_vol = options_data['implied_vol'].rolling(window=5).mean().reindex(close_data.index, method='ffill').fillna(0)
        else:
            oi_ratio = pd.Series(0, index=close_data.index)
            oi_change = pd.Series(0, index=close_data.index)
            implied_vol = pd.Series(0, index=close_data.index)

        target = (returns[ticker].shift(-1) > 0).astype(int).reindex(close_data.index).ffill().fillna(0)

        features = pd.DataFrame({
            'delta': delta_series,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'bb_width': bb_width,
            'vami': vami,
            'momentum': momentum,
            'vix_slope': vix_slope,
            'options_oi_ratio': oi_ratio,
            'options_oi_change': oi_change,
            'implied_vol': implied_vol,
            'target': target
        }).dropna()

        return features.iloc[:-1]  # Drop last row due to target shift
    except Exception as e:
        logger.error(f"Feature preparation error: {e}")
        raise
