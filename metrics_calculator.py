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


class MACDMetric(MetricCalculator):
    """Moving Average Convergence Divergence."""
    
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__(f'macd_{fast}_{slow}_{signal}')
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate(self, df):
        """Calculate MACD, signal line, and histogram."""
        if 'close' not in df.columns:
            return pd.DataFrame(index=df.index)
        
        ema_fast = df['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        result = pd.DataFrame(index=df.index)
        result[f'{self.name}_line'] = macd_line
        result[f'{self.name}_signal'] = signal_line
        result[f'{self.name}_histogram'] = histogram
        
        return result


class StochasticMetric(MetricCalculator):
    """Stochastic Oscillator."""
    
    def __init__(self, period=14, smooth_k=3, smooth_d=3):
        super().__init__(f'stochastic_{period}d')
        self.period = period
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d
    
    def calculate(self, df):
        """Calculate Stochastic %K and %D."""
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(index=df.index)
        
        # Calculate rolling high and low
        low_min = df['low'].rolling(window=self.period).min()
        high_max = df['high'].rolling(window=self.period).max()
        
        # Calculate %K (fast stochastic)
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
        
        # Smooth %K
        stoch_k_smooth = stoch_k.rolling(window=self.smooth_k).mean()
        
        # Calculate %D (signal line)
        stoch_d = stoch_k_smooth.rolling(window=self.smooth_d).mean()
        
        result = pd.DataFrame(index=df.index)
        result[f'{self.name}_k'] = stoch_k_smooth
        result[f'{self.name}_d'] = stoch_d
        
        return result


class ROCMetric(MetricCalculator):
    """Rate of Change."""
    
    def __init__(self, period=12, column='close'):
        super().__init__(f'roc_{period}d')
        self.period = period
        self.column = column
    
    def calculate(self, df):
        """Calculate Rate of Change."""
        if self.column not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        roc = ((df[self.column] - df[self.column].shift(self.period)) / 
               df[self.column].shift(self.period)) * 100
        
        return roc


class VolumeRatioMetric(MetricCalculator):
    """Volume Ratio (current volume vs average volume)."""
    
    def __init__(self, window=20):
        super().__init__(f'volume_ratio_{window}d')
        self.window = window
    
    def calculate(self, df):
        """Calculate volume ratio."""
        if 'volume' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        avg_volume = df['volume'].rolling(window=self.window).mean()
        volume_ratio = df['volume'] / avg_volume
        
        return volume_ratio


class RVOLMetric(MetricCalculator):
    """Relative Volume - current volume compared to average volume."""
    
    def __init__(self, window=20):
        super().__init__(f'rvol_{window}d')
        self.window = window
    
    def calculate(self, df):
        """
        Calculate Relative Volume (RVOL).
        
        RVOL > 1.0 indicates volume is above average
        RVOL < 1.0 indicates volume is below average
        RVOL = 1.0 indicates volume equals the average
        """
        if 'volume' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        # Calculate average volume over the window
        avg_volume = df['volume'].rolling(window=self.window).mean()
        
        # RVOL is the ratio of current volume to average
        rvol = df['volume'] / avg_volume
        
        return rvol


class SharpeRatioMetric(MetricCalculator):
    """Rolling Sharpe Ratio (annualized)."""
    
    def __init__(self, window=252, risk_free_rate=0.02):
        super().__init__(f'sharpe_{window}d')
        self.window = window
        self.risk_free_rate = risk_free_rate
    
    def calculate(self, df):
        """Calculate rolling Sharpe ratio."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        returns = df['close'].pct_change()
        
        # Calculate rolling mean and std of returns
        mean_return = returns.rolling(window=self.window).mean()
        std_return = returns.rolling(window=self.window).std()
        
        # Annualize (assuming daily data)
        annual_return = mean_return * 252
        annual_std = std_return * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (annual_return - self.risk_free_rate) / annual_std
        
        return sharpe


class MaxDrawdownMetric(MetricCalculator):
    """Maximum Drawdown over rolling window."""
    
    def __init__(self, window=252):
        super().__init__(f'max_drawdown_{window}d')
        self.window = window
    
    def calculate(self, df):
        """Calculate rolling maximum drawdown."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        def rolling_max_dd(prices):
            if len(prices) == 0:
                return np.nan
            peak = prices.expanding(min_periods=1).max()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        
        max_dd = df['close'].rolling(window=self.window).apply(rolling_max_dd, raw=False)
        
        return max_dd


class VWAPDistanceMetric(MetricCalculator):
    """Distance from VWAP (as percentage)."""
    
    def __init__(self, window=20):
        super().__init__(f'vwap_distance_{window}d')
        self.window = window
    
    def calculate(self, df):
        """Calculate distance from VWAP."""
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)
        
        # Calculate VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window=self.window).sum() / \
               df['volume'].rolling(window=self.window).sum()
        
        # Distance as percentage
        distance = ((df['close'] - vwap) / vwap) * 100
        
        return distance


class OBVMetric(MetricCalculator):
    """On-Balance Volume."""
    
    def __init__(self, normalize_window=None):
        name = 'obv' if normalize_window is None else f'obv_norm_{normalize_window}d'
        super().__init__(name)
        self.normalize_window = normalize_window
    
    def calculate(self, df):
        """Calculate On-Balance Volume."""
        required_cols = ['close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)
        
        # Calculate price direction
        price_change = df['close'].diff()
        direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        
        # Calculate OBV
        obv = (df['volume'] * direction).cumsum()
        
        # Optional: normalize by rolling mean
        if self.normalize_window:
            obv_mean = obv.rolling(window=self.normalize_window).mean()
            obv_std = obv.rolling(window=self.normalize_window).std()
            obv = (obv - obv_mean) / obv_std
        
        return obv


class ADLineMetric(MetricCalculator):
    """Accumulation/Distribution Line (A/D)."""
    
    def __init__(self, normalize_window=None):
        name = 'ad_line' if normalize_window is None else f'ad_line_norm_{normalize_window}d'
        super().__init__(name)
        self.normalize_window = normalize_window
    
    def calculate(self, df):
        """
        Calculate Accumulation/Distribution Line.
        
        A/D Line measures the cumulative flow of money into and out of a security
        by comparing close price to the high-low range and weighting by volume.

        ADLineMetric Features:
            The A/D Line calculates:

            Money Flow Multiplier (MFM): ((Close - Low) - (High - Close)) / (High - Low)
            Ranges from -1 (close at low) to +1 (close at high)
            Money Flow Volume: MFM × Volume
            A/D Line: Cumulative sum of Money Flow Volume
            The metric includes optional normalization (ad_line_norm_50d) for better comparability across stocks.

            Interpretation:
            Rising A/D Line → Accumulation (buying pressure)
            Falling A/D Line → Distribution (selling pressure)
            Divergence from price → Potential reversal signal
            Price up + A/D down = Bearish divergence
            Price down + A/D up = Bullish divergence

        """
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)
        
        # Calculate Money Flow Multiplier
        # MFM = ((Close - Low) - (High - Close)) / (High - Low)
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        
        # Handle cases where high == low (no range)
        clv = clv.fillna(0)
        
        # Calculate Money Flow Volume
        money_flow_volume = clv * df['volume']
        
        # A/D Line is cumulative sum of Money Flow Volume
        ad_line = money_flow_volume.cumsum()
        
        # Optional: normalize by rolling mean for stationarity
        if self.normalize_window:
            ad_mean = ad_line.rolling(window=self.normalize_window).mean()
            ad_std = ad_line.rolling(window=self.normalize_window).std()
            ad_line = (ad_line - ad_mean) / ad_std
        
        return ad_line


class CCIMetric(MetricCalculator):
    """Commodity Channel Index."""
    
    def __init__(self, period=20):
        super().__init__(f'cci_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate CCI."""
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)
        
        # Typical price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # SMA of typical price
        sma_tp = tp.rolling(window=self.period).mean()
        
        # Mean absolute deviation
        mad = tp.rolling(window=self.period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=False
        )
        
        # CCI
        cci = (tp - sma_tp) / (0.015 * mad)
        
        return cci


class WilliamsRMetric(MetricCalculator):
    """Williams %R."""
    
    def __init__(self, period=14):
        super().__init__(f'williams_r_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Williams %R."""
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)
        
        high_max = df['high'].rolling(window=self.period).max()
        low_min = df['low'].rolling(window=self.period).min()
        
        williams_r = -100 * (high_max - df['close']) / (high_max - low_min)
        
        return williams_r


class DMIMetric(MetricCalculator):
    """Directional Movement Index (DMI) - includes +DI and -DI."""
    
    def __init__(self, period=14):
        super().__init__(f'dmi_{period}d')
        self.period = period
    
    def calculate(self, df):
        """
        Calculate Directional Movement Index (+DI and -DI).
        
        +DI measures upward price movement strength
        -DI measures downward price movement strength
        """
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(index=df.index)
        
        # Calculate directional movements
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        # Positive and negative directional movements
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Calculate True Range (same as ATR calculation)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Smooth using Wilder's smoothing (exponential moving average)
        atr = tr.ewm(alpha=1/self.period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1/self.period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1/self.period, adjust=False).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        result = pd.DataFrame(index=df.index)
        result[f'{self.name}_plus'] = plus_di
        result[f'{self.name}_minus'] = minus_di
        
        return result


class ADXMetric(MetricCalculator):
    """Average Directional Index (ADX) - measures trend strength."""
    
    def __init__(self, period=14):
        super().__init__(f'adx_{period}d')
        self.period = period
    
    def calculate(self, df):
        """
        Calculate Average Directional Index (ADX).
        
        ADX measures the strength of a trend (regardless of direction):
        - ADX > 25: Strong trend
        - ADX < 20: Weak trend or ranging market
        """
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)
        
        # Calculate directional movements
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        # Positive and negative directional movements
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Smooth using Wilder's smoothing
        atr = tr.ewm(alpha=1/self.period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1/self.period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1/self.period, adjust=False).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        # Calculate DX (Directional Index)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=1/self.period, adjust=False).mean()
        
        return adx


# ============================================================================
# EHLERS INDICATORS
# Advanced indicators developed by John F. Ehlers for reduced lag and 
# better cycle/trend identification
# ============================================================================

class EhlersSuperSmootherMetric(MetricCalculator):
    """Ehlers SuperSmoother Filter - 2-pole Butterworth low-pass filter."""
    
    def __init__(self, period=10):
        super().__init__(f'ehlers_supersmoother_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Ehlers SuperSmoother."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close']
        n = len(prices)
        
        # Butterworth coefficients
        a1 = np.exp(-np.sqrt(2) * np.pi / self.period)
        b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / self.period)
        c2 = b1 + a1 * a1
        c3 = -(a1 * a1 + b1 * a1)
        c1 = 1 - c3 - c2
        
        # Initialize output array
        ss = np.zeros(n)
        ss[0] = prices.iloc[0]
        
        if n > 1:
            ss[1] = (c1 * prices.iloc[1] + c2 * ss[0]) / (1 + c3)
        
        # Apply filter
        for i in range(2, n):
            ss[i] = c1 * prices.iloc[i] + c2 * ss[i-1] + c3 * ss[i-2]
        
        return pd.Series(ss, index=df.index)


class EhlersMAMAMetric(MetricCalculator):
    """Ehlers MAMA (Mesa Adaptive Moving Average) and FAMA (Following Adaptive MA)."""
    
    def __init__(self, fast_limit=0.5, slow_limit=0.05):
        super().__init__('ehlers_mama_fama')
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
    
    def calculate(self, df):
        """Calculate MAMA and FAMA."""
        if 'close' not in df.columns:
            return pd.DataFrame(index=df.index)
        
        prices = df['close'].values
        n = len(prices)
        
        mama = np.zeros(n)
        fama = np.zeros(n)
        
        # High-pass filter the prices
        hp = np.zeros(n)
        hp[0] = prices[0]
        
        for i in range(1, n):
            if i >= 2:
                # 2-period high-pass filter
                hp[i] = 0.5 * (1 - 0.015 * i) * (prices[i] - prices[i-2]) + 0.79 * hp[i-1]
            else:
                hp[i] = prices[i] - prices[i-1]
        
        # Hilbert transform approximation via phasing
        phase = np.zeros(n)
        for i in range(2, n):
            if i >= 6:
                # Compute phase
                quad = (0.0962 * hp[i] + 0.5769 * hp[i-2] - 0.5769 * hp[i-4] - 0.0962 * hp[i-6])
                real = (0.0962 * hp[i-1] + 0.5769 * hp[i-3] - 0.5769 * hp[i-5] - 0.0962 * hp[i-7] if i >= 7 else 0)
                if real != 0:
                    phase[i] = np.arctan(quad / real) if quad / real > 0 else np.arctan(quad / real) + np.pi
                else:
                    phase[i] = phase[i-1]
        
        # Compute Adaptive MA
        dc_period = np.full(n, 20.0)  # Initialize dominant cycle
        smooth = np.full(n, 0.0)
        
        for i in range(1, n):
            if i >= 10:
                dc_period[i] = 2 * np.pi / (0.075 * phase[i] + 0.54) if phase[i] != 0 else dc_period[i-1]
            
            alpha = 2.0 / (dc_period[i] + 1.0)
            smooth[i] = 0.33 * prices[i] + 0.67 * smooth[i-1] if i > 0 else prices[i]
            
            # MAMA calculation
            mama[i] = (self.fast_limit * smooth[i]) + (1 - self.fast_limit) * (mama[i-1] if i > 0 else smooth[i])
            # FAMA calculation
            fama[i] = (self.slow_limit * mama[i]) + (1 - self.slow_limit) * (fama[i-1] if i > 0 else mama[i])
        
        result = pd.DataFrame(index=df.index)
        result['ehlers_mama'] = mama
        result['ehlers_fama'] = fama
        
        return result


class EhlersRoofingFilterMetric(MetricCalculator):
    """Ehlers Roofing Filter - High-Pass + SuperSmoother combination."""
    
    def __init__(self, period=10):
        super().__init__(f'ehlers_roofing_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Roofing Filter."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close'].values
        n = len(prices)
        
        # High-pass filter component
        hp = np.zeros(n)
        for i in range(2, n):
            hp[i] = 0.5 * (1.5 - 0.75 / self.period) * (prices[i] - prices[i-1]) + 0.79 * hp[i-1]
        
        # Apply SuperSmoother to high-pass output
        a1 = np.exp(-np.sqrt(2) * np.pi / self.period)
        b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / self.period)
        c2 = b1 + a1 * a1
        c3 = -(a1 * a1 + b1 * a1)
        c1 = 1 - c3 - c2
        
        roof = np.zeros(n)
        roof[0] = hp[0]
        if n > 1:
            roof[1] = (c1 * hp[1] + c2 * roof[0]) / (1 + c3)
        
        for i in range(2, n):
            roof[i] = c1 * hp[i] + c2 * roof[i-1] + c3 * roof[i-2]
        
        return pd.Series(roof, index=df.index)


class EhlersGaussianFilterMetric(MetricCalculator):
    """Ehlers Gaussian Filter - N-pole filter with bell-curve weighting."""
    
    def __init__(self, period=10):
        super().__init__(f'ehlers_gaussian_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Gaussian Filter."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close']
        
        # Create Gaussian weights
        weights = np.zeros(self.period)
        for i in range(self.period):
            weights[i] = np.exp(-((i - self.period/2) ** 2) / (2 * (self.period/4) ** 2))
        
        weights /= weights.sum()
        
        # Apply weighted convolution
        result = prices.rolling(window=self.period).apply(
            lambda x: np.dot(x.values, weights), raw=False
        )
        
        return result


class EhlersFisherTransformMetric(MetricCalculator):
    """Ehlers Fisher Transform - Normalizes price to Gaussian distribution."""
    
    def __init__(self, period=10):
        super().__init__(f'ehlers_fisher_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Fisher Transform."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close']
        
        # Normalize prices to -1 to 1 range
        min_price = prices.rolling(window=self.period).min()
        max_price = prices.rolling(window=self.period).max()
        
        normalized = 2 * (prices - min_price) / (max_price - min_price + 0.001) - 1
        
        # Apply Fisher Transform: 0.5 * ln((1+x)/(1-x))
        # Use fillna to handle edge cases
        normalized = normalized.clip(-0.999, 0.999)
        fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 0.001))
        
        return fisher


class EhlersLaguerreRSIMetric(MetricCalculator):
    """Ehlers Laguerre RSI - Smoother RSI using Laguerre polynomials."""
    
    def __init__(self, period=14, alpha=0.5):
        super().__init__(f'ehlers_laguerre_rsi_{period}d')
        self.period = period
        self.alpha = alpha
    
    def calculate(self, df):
        """Calculate Laguerre RSI."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close'].values
        n = len(prices)
        
        # Apply Laguerre filter
        L0 = np.zeros(n)
        L1 = np.zeros(n)
        L2 = np.zeros(n)
        L3 = np.zeros(n)
        
        for i in range(n):
            L0[i] = self.alpha * prices[i] + (1 - self.alpha) * (L0[i-1] if i > 0 else prices[i])
            L1[i] = -(1 - self.alpha) * L0[i] + (L0[i-1] if i > 0 else L0[i]) + (1 - self.alpha) * (L1[i-1] if i > 0 else 0)
            L2[i] = -(1 - self.alpha) * L1[i] + (L1[i-1] if i > 0 else L1[i]) + (1 - self.alpha) * (L2[i-1] if i > 0 else 0)
            L3[i] = -(1 - self.alpha) * L2[i] + (L2[i-1] if i > 0 else L2[i]) + (1 - self.alpha) * (L3[i-1] if i > 0 else 0)
        
        # Compute RSI from Laguerre values
        cu = np.zeros(n)
        cd = np.zeros(n)
        
        for i in range(1, n):
            diff = L3[i] - L3[i-1]
            if diff > 0:
                cu[i] = diff
            else:
                cd[i] = -diff
        
        cu_smooth = pd.Series(cu).rolling(window=self.period).mean().values
        cd_smooth = pd.Series(cd).rolling(window=self.period).mean().values
        
        laguerre_rsi = np.zeros(n)
        for i in range(n):
            if cu_smooth[i] + cd_smooth[i] > 0:
                laguerre_rsi[i] = cu_smooth[i] / (cu_smooth[i] + cd_smooth[i])
            else:
                laguerre_rsi[i] = 0
        
        return pd.Series(laguerre_rsi, index=df.index)


class EhlersCyberCycleMetric(MetricCalculator):
    """Ehlers Cyber Cycle - Isolates cyclic component of price action."""
    
    def __init__(self, period=10):
        super().__init__(f'ehlers_cyber_cycle_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Cyber Cycle."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close'].values
        n = len(prices)
        
        # Smooth prices
        smooth = np.zeros(n)
        for i in range(n):
            if i < 3:
                smooth[i] = np.mean(prices[:i+1])
            else:
                smooth[i] = (prices[i] + 2 * prices[i-1] + 2 * prices[i-2] + prices[i-3]) / 6.0
        
        # Apply band-pass filter
        cc = np.zeros(n)
        for i in range(3, n):
            # Cyber Cycle band-pass approximation
            cc[i] = (smooth[i] - 2 * smooth[i-1] + smooth[i-2]) * 0.25 + cc[i-1] * 0.5
        
        return pd.Series(cc, index=df.index)


class EhlersRVIMetric(MetricCalculator):
    """Ehlers Relative Vigor Index - Measures trend conviction."""
    
    def __init__(self, period=10):
        super().__init__(f'ehlers_rvi_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Relative Vigor Index."""
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)
        
        # Numerator: (Close - Open)
        numerator = (df['close'] - df['open']).rolling(window=self.period).sum()
        
        # Denominator: (High - Low)
        denominator = (df['high'] - df['low']).rolling(window=self.period).sum()
        
        # RVI = Numerator / Denominator
        rvi = numerator / (denominator + 0.001)
        
        return rvi


class EhlersReflexMetric(MetricCalculator):
    """Ehlers Reflex - Identifies cycle peaks and valleys."""
    
    def __init__(self, period=10):
        super().__init__(f'ehlers_reflex_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Reflex indicator."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close'].values
        n = len(prices)
        
        # Calculate momentum with zero lag
        reflex = np.zeros(n)
        for i in range(2, n):
            # Zero-lag momentum
            if i >= self.period:
                reflex[i] = prices[i] - prices[i - self.period]
            else:
                reflex[i] = prices[i] - prices[0]
        
        return pd.Series(reflex, index=df.index)


class EhlersTrendflexMetric(MetricCalculator):
    """Ehlers Trendflex - Identifies trend onset/exhaustion."""
    
    def __init__(self, period=10):
        super().__init__(f'ehlers_trendflex_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Trendflex indicator."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close'].values
        n = len(prices)
        
        # Compute detrended price
        trendflex = np.zeros(n)
        for i in range(n):
            if i >= self.period:
                # Slope of linear regression
                x = np.arange(self.period)
                y = prices[i-self.period+1:i+1]
                z = np.polyfit(x, y, 1)
                trendflex[i] = prices[i] - (z[0] * (self.period - 1) + z[1])
            else:
                trendflex[i] = prices[i] - np.mean(prices[:i+1])
        
        return pd.Series(trendflex, index=df.index)


class EhlersHilbertSineMetric(MetricCalculator):
    """Ehlers Hilbert Sine Wave - Lead and Sine lines for cycle identification."""
    
    def __init__(self, period=10):
        super().__init__('ehlers_hilbert_sine')
        self.period = period
    
    def calculate(self, df):
        """Calculate Hilbert Sine Wave (Lead Sine and Sine)."""
        if 'close' not in df.columns:
            return pd.DataFrame(index=df.index)
        
        prices = df['close'].values
        n = len(prices)
        
        # High-pass filter
        hp = np.zeros(n)
        for i in range(2, n):
            hp[i] = 0.962 * hp[i-1] + (prices[i] - prices[i-1]) * 0.5
        
        # Hilbert Transform approximation
        lead_sine = np.zeros(n)
        sine = np.zeros(n)
        
        for i in range(3, n):
            if i >= 6:
                # In-phase and Quadrature components
                inphase = 0.0962 * hp[i] + 0.5769 * hp[i-2] - 0.5769 * hp[i-4] - 0.0962 * hp[i-6]
                quad = 0.0962 * hp[i-1] + 0.5769 * hp[i-3] - 0.5769 * hp[i-5] - 0.0962 * hp[i-7] if i >= 7 else 0
                
                # Normalize
                magnitude = np.sqrt(inphase**2 + quad**2 + 0.001)
                inphase /= magnitude
                quad /= magnitude
                
                lead_sine[i] = inphase
                sine[i] = quad
        
        result = pd.DataFrame(index=df.index)
        result['ehlers_hilbert_lead_sine'] = lead_sine
        result['ehlers_hilbert_sine'] = sine
        
        return result


class EhlersDecyclerMetric(MetricCalculator):
    """Ehlers Decycler - High-pass filter that removes cycles."""
    
    def __init__(self, period=20):
        super().__init__(f'ehlers_decycler_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Decycler."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close'].values
        n = len(prices)
        
        # High-pass filter (removes cycles)
        decycler = np.zeros(n)
        for i in range(2, n):
            # 2-pole high-pass filter
            if i >= 2:
                decycler[i] = 0.5 * (1.5 - 0.75 / self.period) * (prices[i] - prices[i-1]) + 0.79 * decycler[i-1]
        
        return pd.Series(decycler, index=df.index)


class EhlersContinuationIndexMetric(MetricCalculator):
    """Ehlers Continuation Index - Binary signals for trend onset/exhaustion."""
    
    def __init__(self, period=10):
        super().__init__(f'ehlers_continuation_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Continuation Index."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close'].values
        n = len(prices)
        
        # Simple trend strength indicator
        ci = np.zeros(n)
        for i in range(1, n):
            # Compare current momentum to average
            if i >= self.period:
                recent_momentum = prices[i] - prices[i - self.period]
                avg_momentum = np.mean([prices[j] - prices[j - 1] for j in range(i - self.period + 1, i + 1)])
                ci[i] = 1 if recent_momentum > avg_momentum else -1
            else:
                ci[i] = 0
        
        return pd.Series(ci, index=df.index)


class EhlersVossPredictiveMetric(MetricCalculator):
    """Ehlers Voss Predictive Filter - Leads price action by predicted bars."""
    
    def __init__(self, period=10):
        super().__init__(f'ehlers_voss_predictive_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Voss Predictive Filter."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close'].values
        n = len(prices)
        
        # Trend-following with predictive component
        voss = np.zeros(n)
        
        for i in range(1, n):
            if i >= self.period:
                # Linear regression slope (predictive component)
                x = np.arange(self.period)
                y = prices[i - self.period + 1:i + 1]
                z = np.polyfit(x, y, 1)
                
                # Current price plus slope projection
                voss[i] = prices[i] + z[0]  # Add slope for ahead prediction
            else:
                voss[i] = prices[i]
        
        return pd.Series(voss, index=df.index)


class EhlersInstantaneousTrendlineMetric(MetricCalculator):
    """Ehlers Instantaneous Trendline - Follows trend, flat during cycles."""
    
    def __init__(self, period=10):
        super().__init__(f'ehlers_inst_trendline_{period}d')
        self.period = period
    
    def calculate(self, df):
        """Calculate Instantaneous Trendline."""
        if 'close' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        prices = df['close'].values
        n = len(prices)
        
        # Compute instantaneous trendline
        trendline = np.zeros(n)
        
        for i in range(n):
            if i == 0:
                trendline[i] = prices[i]
            elif i < self.period:
                trendline[i] = np.mean(prices[:i+1])
            else:
                # Mode detection: if oscillating, use midpoint; if trending, use median
                window = prices[i - self.period + 1:i + 1]
                midpoint = (np.max(window) + np.min(window)) / 2.0
                
                # Detect if trending or cycling
                swings = sum(1 for j in range(1, len(window)) if 
                           (window[j] > window[j-1] and (j > 1 and window[j-1] < window[j-2])) or
                           (window[j] < window[j-1] and (j > 1 and window[j-1] > window[j-2])))
                
                # If many swings, it's cycling - use midpoint; otherwise follow
                if swings >= self.period / 3:
                    trendline[i] = midpoint
                else:
                    trendline[i] = 0.5 * prices[i] + 0.5 * trendline[i-1]
        
        return pd.Series(trendline, index=df.index)


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
    
    def register_default_metrics(self, include_advanced=True):
        """
        Register a standard set of metrics.
        
        Parameters:
        -----------
        include_advanced : bool
            If True, includes advanced metrics (MACD, Stochastic, etc.)
        """
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
        
        # Advanced metrics (optional)
        if include_advanced:
            # MACD
            self.register_metric(MACDMetric(12, 26, 9))
            
            # Stochastic Oscillator
            self.register_metric(StochasticMetric(14))
            
            # Rate of Change
            self.register_metric(ROCMetric(12))
            self.register_metric(ROCMetric(21))
            
            # Volume-based
            self.register_metric(VolumeRatioMetric(20))
            self.register_metric(RVOLMetric(10))  # Short-term RVOL
            self.register_metric(RVOLMetric(20))  # Medium-term RVOL
            self.register_metric(RVOLMetric(50))  # Long-term RVOL
            self.register_metric(OBVMetric(normalize_window=50))
            self.register_metric(ADLineMetric(normalize_window=50))
            
            # VWAP Distance
            self.register_metric(VWAPDistanceMetric(20))
            
            # Risk metrics
            self.register_metric(SharpeRatioMetric(window=63))  # Quarterly
            self.register_metric(MaxDrawdownMetric(window=63))
            
            # Additional oscillators
            self.register_metric(CCIMetric(20))
            self.register_metric(WilliamsRMetric(14))
            
            # Directional indicators
            self.register_metric(DMIMetric(14))
            self.register_metric(ADXMetric(14))
            
            # Ehlers Core Smoothing & Trend Filters
            self.register_metric(EhlersSuperSmootherMetric(10))
            self.register_metric(EhlersMAMAMetric())
            self.register_metric(EhlersRoofingFilterMetric(10))
            self.register_metric(EhlersGaussianFilterMetric(10))
            
            # Ehlers Oscillators & Momentum
            self.register_metric(EhlersFisherTransformMetric(10))
            self.register_metric(EhlersLaguerreRSIMetric(14))
            self.register_metric(EhlersCyberCycleMetric(10))
            self.register_metric(EhlersRVIMetric(10))
            self.register_metric(EhlersReflexMetric(10))
            self.register_metric(EhlersTrendflexMetric(10))
            
            # Ehlers Cycle & Regime Identification
            self.register_metric(EhlersHilbertSineMetric(10))
            self.register_metric(EhlersDecyclerMetric(20))
            self.register_metric(EhlersContinuationIndexMetric(10))
            
            # Ehlers Forecasting & Predictive
            self.register_metric(EhlersVossPredictiveMetric(10))
            self.register_metric(EhlersInstantaneousTrendlineMetric(10))
        
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
