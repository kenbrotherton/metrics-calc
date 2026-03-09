"""
Modular Metrics Calculator for Rolling Time-Series Metrics

This module provides a flexible framework for calculating rolling metrics
that are stored per-date per-symbol, allowing historical queries like:
"What was AAPL's 20-day VWAP on 2025-01-01?"

Architecture:
- MetricCalculator: Base class for individual metric calculations
- MetricsEngine: Orchestrates multiple metric calculations
- Output: Date-indexed dataframes with all metrics per symbol
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class MetricCalculator(ABC):
    """Base class for metric calculators."""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def calculate(self, df):
        """
        Calculate metric for the given price dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with columns: open, high, low, close, volume
            Index: date
        
        Returns:
        --------
        pd.Series or pd.DataFrame
            Calculated metric(s) indexed by date
        """
        pass


class VWAPMetric(MetricCalculator):
    """Volume-Weighted Average Price over rolling window."""
    
    def __init__(self, window):
        super().__init__(f'vwap_{window}d')
        self.window = window
    
    def calculate(self, df):
        """Calculate rolling VWAP."""
        if 'close' not in df.columns or 'volume' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window=self.window).sum() / \
               df['volume'].rolling(window=self.window).sum()
        
        return vwap


class SMAMetric(MetricCalculator):
    """Simple Moving Average."""
    
    def __init__(self, window, column='close'):
        super().__init__(f'sma_{window}d')
        self.window = window
        self.column = column
    
    def calculate(self, df):
        """Calculate simple moving average."""
        if self.column not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        return df[self.column].rolling(window=self.window).mean()


class EMAMetric(MetricCalculator):
    """Exponential Moving Average."""
    
    def __init__(self, span, column='close'):
        super().__init__(f'ema_{span}d')
        self.span = span
        self.column = column
    
    def calculate(self, df):
        """Calculate exponential moving average."""
        if self.column not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        return df[self.column].ewm(span=self.span, adjust=False).mean()


class VolatilityMetric(MetricCalculator):
    """Rolling volatility (standard deviation of returns)."""
    
    def __init__(self, window, column='close'):
        super().__init__(f'volatility_{window}d')
        self.window = window
        self.column = column
    
    def calculate(self, df):
        """Calculate rolling volatility."""
        if self.column not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        returns = df[self.column].pct_change()
        return returns.rolling(window=self.window).std()


class MomentumMetric(MetricCalculator):
    """Price momentum (percentage change over window)."""
    
    def __init__(self, window, column='close'):
        super().__init__(f'momentum_{window}d')
        self.window = window
        self.column = column
    
    def calculate(self, df):
        """Calculate momentum."""
        if self.column not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        return df[self.column].pct_change(periods=self.window)


class RSIMetric(MetricCalculator):
    """Relative Strength Index."""
    
    def __init__(self, period=14):
        super().__init__(f'rsi_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate RSI."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class VolumeMetric(MetricCalculator):
    """Rolling volume statistics."""
    
    def __init__(self, window):
        super().__init__(f'avg_volume_{window}d')
        self.window = window
    
    def calculate(self, df):
        """Calculate rolling average volume."""
        if 'volume' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        return df['volume'].rolling(window=self.window).mean()


class BollingerBandsMetric(MetricCalculator):
    """Bollinger Bands (upper, middle, lower)."""
    
    def __init__(self, window=20, num_std=2):
        super().__init__(f'bollinger_{window}d')
        self.window = window
        self.num_std = num_std
    
    def calculate(self, df):
        """Calculate Bollinger Bands."""
        if 'close' not in df.columns:
            return pd.DataFrame(index=df.index)
        
        sma = df['close'].rolling(window=self.window).mean()
        std = df['close'].rolling(window=self.window).std()
        
        result = pd.DataFrame(index=df.index)
        result[f'{self.name}_middle'] = sma
        result[f'{self.name}_upper'] = sma + (std * self.num_std)
        result[f'{self.name}_lower'] = sma - (std * self.num_std)
        
        return result


class ATRMetric(MetricCalculator):
    """Average True Range."""
    
    def __init__(self, period=14):
        super().__init__(f'atr_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate ATR."""
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.period).mean()
        
        return atr


class MetricsEngine:
    """
    Orchestrates calculation of multiple metrics for a symbol's price data.
    
    This engine applies all registered metric calculators to price data
    and returns a date-indexed DataFrame with all metrics.
    """
    
    def __init__(self):
        self.calculators = []
    
    def register_metric(self, calculator):
        """Register a metric calculator."""
        if not isinstance(calculator, MetricCalculator):
            raise TypeError("Calculator must be instance of MetricCalculator")
        self.calculators.append(calculator)
        return self
    
    def register_default_metrics(self):
        """Register a standard set of metrics."""
        # VWAP at multiple timeframes
        self.register_metric(VWAPMetric(20))
        self.register_metric(VWAPMetric(50))
        self.register_metric(VWAPMetric(200))
        
        # Moving averages
        self.register_metric(SMAMetric(20))
        self.register_metric(SMAMetric(50))
        self.register_metric(SMAMetric(200))
        self.register_metric(EMAMetric(12))
        self.register_metric(EMAMetric(26))
        
        # Volatility
        self.register_metric(VolatilityMetric(20))
        self.register_metric(VolatilityMetric(50))
        
        # Momentum
        self.register_metric(MomentumMetric(21))
        self.register_metric(MomentumMetric(63))
        
        # Technical indicators
        self.register_metric(RSIMetric(14))
        self.register_metric(VolumeMetric(20))
        self.register_metric(BollingerBandsMetric(20))
        self.register_metric(ATRMetric(14))
        
        return self
    
    def calculate_all(self, df):
        """
        Calculate all registered metrics for the given price dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with columns: open, high, low, close, volume
            Index: date
        
        Returns:
        --------
        pd.DataFrame
            Date-indexed DataFrame with price data and all calculated metrics
        """
        if df is None or len(df) == 0:
            return None
        
        # Start with original price data
        result = df.copy()
        
        # Calculate each metric
        for calc in self.calculators:
            try:
                metric_result = calc.calculate(df)
                
                # Handle both Series and DataFrame results
                if isinstance(metric_result, pd.Series):
                    result[calc.name] = metric_result
                elif isinstance(metric_result, pd.DataFrame):
                    for col in metric_result.columns:
                        result[col] = metric_result[col]
                        
            except Exception as e:
                print(f"[!] Error calculating {calc.name}: {e}")
                continue
        
        return result
    
    def list_metrics(self):
        """List all registered metric names."""
        metrics = []
        for calc in self.calculators:
            if hasattr(calc, 'name'):
                metrics.append(calc.name)
        return metrics


def query_metric(metrics_df, date, metric_name):
    """
    Query a specific metric value for a specific date.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        Date-indexed dataframe with metrics
    date : str or datetime
        Date to query
    metric_name : str
        Name of the metric column
    
    Returns:
    --------
    float or None
        Metric value at that date, or None if not found
    """
    try:
        date = pd.to_datetime(date)
        if date in metrics_df.index and metric_name in metrics_df.columns:
            return metrics_df.loc[date, metric_name]
        return None
    except Exception:
        return None
