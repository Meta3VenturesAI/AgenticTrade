import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import os
from data_loader import fetch_data
from feature_engineer import calculate_delta, compute_rsi, compute_macd, detect_spikes, intraday_momentum_signal, calculate_vami, prepare_features, compute_bollinger_width
from model import backtest_strategy, SELECTED_FEATURES

# Mock data for testing
@pytest.fixture
def mock_data():
    dates = pd.date_range('2023-01-01', '2023-01-10')
    close_data = pd.DataFrame({
        'SPXL': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'SPY': [50, 50.5, 51, 51.5, 52, 52.5, 53, 53.5, 54, 54.5]
    }, index=dates)
    open_data = close_data * 0.99
    high_data = close_data * 1.01
    low_data = close_data * 0.98
    returns = close_data.pct_change().fillna(0)
    delta = calculate_delta(close_data[['SPXL']], close_data[['SPY']], [3])
    vix_data = pd.DataFrame({'^VIX': [20, 21, 19, 22, 18, 23, 17, 24, 16, 25]}, index=dates)
    volatility = pd.DataFrame({
        'SPXL': [0.01, 0.02, 0.015, 0.025, 0.01, 0.03, 0.005, 0.04, 0.02, 0.01],
        'SPY': [0.005, 0.01, 0.007, 0.012, 0.005, 0.015, 0.003, 0.02, 0.01, 0.005]
    }, index=dates)
    options_data = pd.DataFrame({
        'openInterest': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'implied_vol': [0.2, 0.21, 0.19, 0.22, 0.18, 0.23, 0.17, 0.24, 0.16, 0.25]
    }, index=dates)
    return close_data, open_data, high_data, low_data, returns, delta, vix_data, volatility, options_data

def test_fetch_data():
    with patch('data_loader.yf.download') as mock_download:
        dates = pd.date_range('2023-01-01', periods=60)
        # Create a DataFrame with a single-level column index as yf.download would return
        mock_data = pd.DataFrame({
            'Close': [100 + i for i in range(60)],
            'Open': [99 + i for i in range(60)],
            'High': [101 + i for i in range(60)],
            'Low': [98 + i for i in range(60)],
            'Adj Close': [100 + i for i in range(60)]
        }, index=dates)
        mock_download.return_value = mock_data
        close_data, open_data, high_data, low_data, returns, tickers = fetch_data(['SPXL'], '2023-01-01', '2023-03-01', '1d', min_data_length=60)
        assert 'SPXL' in close_data.columns
        assert len(close_data) == 60
        assert np.isclose(returns['SPXL'].iloc[0], (101 - 100) / 100, rtol=1e-5)  # Check first non-NaN return

def test_calculate_delta(mock_data):
    close_data, _, _, _, _, _, _, _, _ = mock_data
    delta = calculate_delta(close_data[['SPXL']], close_data[['SPY']], [3])
    assert isinstance(delta, pd.Series)
    assert len(delta) == len(close_data)
    expected_first_delta = (101/100 - 1) - 3 * (50.5/50 - 1)  # -0.02
    assert np.isclose(delta.iloc[1], expected_first_delta, rtol=1e-5)

def test_compute_rsi(mock_data):
    close_data, _, _, _, _, _, _, _, _ = mock_data
    rsi = compute_rsi(close_data['SPXL'])
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(close_data)
    assert (rsi >= 0).all() and (rsi <= 100).all()

def test_compute_macd(mock_data):
    close_data, _, _, _, _, _, _, _, _ = mock_data
    macd, signal = compute_macd(close_data['SPXL'])
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)
    assert len(macd) == len(close_data)
    assert len(signal) == len(close_data)

def test_compute_bollinger_width(mock_data):
    close_data, _, _, _, _, _, _, _, _ = mock_data
    bb_width = compute_bollinger_width(close_data['SPXL'], period=5, num_std=2.0)
    assert isinstance(bb_width, pd.Series)
    assert len(bb_width) == len(close_data)
    assert (bb_width >= 0).all()
    prices = close_data['SPXL'].iloc[:5]
    rolling_mean = prices.mean()
    rolling_std = prices.std()
    expected_width = (2 * 2.0 * rolling_std) / rolling_mean
    assert np.isclose(bb_width.iloc[4], expected_width, rtol=1e-3)

def test_detect_spikes(mock_data):
    _, _, high_data, low_data, _, _, _, _, _ = mock_data
    spikes = detect_spikes(high_data, low_data, 'SPXL')
    assert isinstance(spikes, pd.Series)
    assert len(spikes) == len(high_data)
    assert spikes.isin([-1, 0, 1]).all()

def test_intraday_momentum_signal(mock_data):
    _, open_data, _, close_data, _, _, _, _, _ = mock_data
    signals = intraday_momentum_signal(open_data, close_data, 'SPXL')
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(close_data)
    assert signals.isin([-1, 0, 1]).all()

def test_calculate_vami(mock_data):
    _, _, _, _, returns, _, _, volatility, _ = mock_data
    vami = calculate_vami(tuple(returns['SPXL'].values), tuple(volatility['SPXL'].values))
    assert isinstance(vami, pd.Series)
    assert len(vami) == len(returns)
    assert (vami >= 0).all()

def test_prepare_features(mock_data):
    close_data, _, _, _, returns, delta, vix_data, volatility, options_data = mock_data
    features = prepare_features(close_data, returns, delta, 'SPXL', vix_data, volatility, options_data)  # Fixed typo and added comma
    assert isinstance(features, pd.DataFrame)
    assert 'delta' in features.columns
    assert 'rsi' in features.columns
    assert 'macd' in features.columns
    assert 'bb_width' in features.columns
    assert 'target' in features.columns
    assert len(features) == len(close_data) - 1

def test_backtest_strategy(mock_data):
    close_data, open_data, high_data, low_data, returns, delta, vix_data, volatility, options_data = mock_data
    features = prepare_features(close_data, returns, delta, 'SPXL', vix_data, volatility, options_data)
    features['target'] = [1] * 4 + [0] * 5  # Force both classes
    tickers = [{'leveraged_etf': 'SPXL', 'underlying': 'SPY'}]
    best_params = {'seq_length': 5}
    selected_features = SELECTED_FEATURES

    assert 'bb_width' in selected_features, "Bollinger Bands Width feature not included in selected_features"

    # Ensure the report file does not exist before running
    report_path = 'strategy_report.html'
    if os.path.exists(report_path):
        os.remove(report_path)

    portfolio_returns, trade_log, metrics, _, _, _ = backtest_strategy(
        close_data, open_data, high_data, low_data, features, tickers,
        selected_features, best_params, confidence_threshold=0.6,
        vix_slope_threshold=0.1, oi_ratio_threshold=0.5, oi_change_threshold=0.2,
        scaler=None
    )
    assert isinstance(portfolio_returns, pd.Series)
    assert isinstance(trade_log, pd.DataFrame)
    assert len(metrics) > 0
    assert trade_log.columns.tolist() == ['Date', 'Ticker', 'Action', 'Return']
    # Verify that the report was generated
    assert os.path.exists(report_path), "strategy_report.html was not generated"
