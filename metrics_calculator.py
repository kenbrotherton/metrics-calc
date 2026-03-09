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
import math
from abc import ABC, abstractmethod


class MetricCalculator(ABC):
    """Base class for metric calculators."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calculate(self, df):
        """Calculate metric for the given price dataframe."""
        pass


# ===== BASIC INDICATORS =====
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
        vwap = (typical_price * df['volume']).rolling(window=self.window).sum() / df['volume'].rolling(window=self.window).sum()

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


# ===== ADVANCED VOLUME INDICATORS =====
class RVOLMetric(MetricCalculator):
    """Relative Volume (Volume / SMA of Volume)."""

    def __init__(self, window):
        super().__init__(f'rvol_{window}d')
        self.window = window

    def calculate(self, df):
        """Calculate RVOL."""
        if 'volume' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        avg_vol = df['volume'].rolling(window=self.window).mean()
        rvol = df['volume'] / avg_vol

        return rvol


class OBVMetric(MetricCalculator):
    """On-Balance Volume."""

    def __init__(self):
        super().__init__('obv')

    def calculate(self, df):
        """Calculate OBV."""
        if 'close' not in df.columns or 'volume' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv


class OBVNormMetric(MetricCalculator):
    """On-Balance Volume (normalized over period)."""

    def __init__(self, window=50):
        super().__init__(f'obv_norm_{window}d')
        self.window = window

    def calculate(self, df):
        """Calculate normalized OBV."""
        if 'close' not in df.columns or 'volume' not in df.columns:
            return pd.Series(index=df.index, dtype=float)

        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        obv_norm = obv / df['volume'].rolling(window=self.window).sum()

        return obv_norm


class ADLineMetric(MetricCalculator):
    """Accumulation/Distribution Line (Chaikin A/D Line)."""

    def __init__(self):
        super().__init__('ad_line')

    def calculate(self, df):
        """Calculate A/D Line."""
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)

        mfv = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
        mfv = mfv.fillna(0)
        ad_line = (mfv * df['volume']).cumsum()

        return ad_line


class ADLineNormMetric(MetricCalculator):
    """Accumulation/Distribution Line (normalized)."""

    def __init__(self, window=50):
        super().__init__(f'ad_line_norm_{window}d')
        self.window = window

    def calculate(self, df):
        """Calculate normalized A/D Line."""
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)

        mfv = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
        mfv = mfv.fillna(0)
        ad_line = (mfv * df['volume']).cumsum()
        ad_line_norm = ad_line / df['volume'].rolling(window=self.window).sum()

        return ad_line_norm


# ===== DIRECTIONAL INDICATORS =====
class DMIMetric(MetricCalculator):
    """Directional Movement Index (+DI, -DI)."""

    def __init__(self, period=14):
        super().__init__(f'dmi_{period}d')
        self.period = period

    def calculate(self, df):
        """Calculate DMI components."""
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(index=df.index)

        up = df['high'].diff()
        down = -df['low'].diff()

        up = up.where((up > down) & (up > 0), 0)
        down = down.where((down > up) & (down > 0), 0)

        tr1 = df['high'] - df['low']
        tr2 = np.abs(df['high'] - df['close'].shift())
        tr3 = np.abs(df['low'] - df['close'].shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()

        plus_di = 100 * (up.rolling(window=self.period).mean() / atr)
        minus_di = 100 * (down.rolling(window=self.period).mean() / atr)

        result = pd.DataFrame(index=df.index)
        result[f'{self.name}_plus'] = plus_di
        result[f'{self.name}_minus'] = minus_di

        return result


class ADXMetric(MetricCalculator):
    """Average Directional Index."""

    def __init__(self, period=14):
        super().__init__(f'adx_{period}d')
        self.period = period

    def calculate(self, df):
        """Calculate ADX."""
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)

        up = df['high'].diff()
        down = -df['low'].diff()

        up = up.where((up > down) & (up > 0), 0)
        down = down.where((down > up) & (down > 0), 0)

        tr1 = df['high'] - df['low']
        tr2 = np.abs(df['high'] - df['close'].shift())
        tr3 = np.abs(df['low'] - df['close'].shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()

        plus_di = 100 * (up.rolling(window=self.period).mean() / atr)
        minus_di = 100 * (down.rolling(window=self.period).mean() / atr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.period).mean()

        return adx


# ===== EHLERS HELPER FUNCTIONS =====
def _safe_price_series(df):
    if 'close' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return pd.to_numeric(df['close'], errors='coerce').astype(float)


def _super_smoother_filter(price, period=10):
    price = pd.to_numeric(price, errors='coerce').astype(float)
    if len(price) == 0:
        return pd.Series(index=price.index, dtype=float)

    period = max(2, int(period))
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1
    c3 = -(a1 * a1)
    c1 = 1 - c2 - c3

    x = price.ffill().bfill()
    out = np.full(len(x), np.nan, dtype=float)

    if len(x) > 0:
        out[0] = x.iloc[0]
    if len(x) > 1:
        out[1] = x.iloc[1]

    for i in range(2, len(x)):
        out[i] = c1 * 0.5 * (x.iloc[i] + x.iloc[i - 1]) + c2 * out[i - 1] + c3 * out[i - 2]

    return pd.Series(out, index=price.index)


def _high_pass_filter(price, period=48):
    price = pd.to_numeric(price, errors='coerce').astype(float)
    if len(price) == 0:
        return pd.Series(index=price.index, dtype=float)

    period = max(3, int(period))
    arg = 0.707 * 2 * math.pi / period
    cos_arg = math.cos(arg)
    alpha = (cos_arg + math.sin(arg) - 1.0) / max(cos_arg, 1e-9)

    x = price.ffill().bfill()
    hp = np.zeros(len(x), dtype=float)

    for i in range(2, len(x)):
        hp[i] = ((1 - alpha / 2) ** 2) * (x.iloc[i] - 2 * x.iloc[i - 1] + x.iloc[i - 2]) + \
                2 * (1 - alpha) * hp[i - 1] - ((1 - alpha) ** 2) * hp[i - 2]

    return pd.Series(hp, index=price.index)


def _gaussian_filter(price, period=20, poles=3):
    price = pd.to_numeric(price, errors='coerce').astype(float)
    if len(price) == 0:
        return pd.Series(index=price.index, dtype=float)

    period = max(2, int(period))
    poles = max(1, int(poles))
    alpha = 2.0 / (period + 1.0)

    out = price.copy()
    for _ in range(poles):
        out = out.ewm(alpha=alpha, adjust=False).mean()
    return out


# ===== EHLERS FILTERS & TREND =====
class SuperSmootherMetric(MetricCalculator):
    def __init__(self, period=10):
        super().__init__(f'super_smoother_{period}d')
        self.period = period

    def calculate(self, df):
        return _super_smoother_filter(_safe_price_series(df), self.period)


class MAMAFAMAMetric(MetricCalculator):
    def __init__(self, fast_limit=0.5, slow_limit=0.05, vol_window=10):
        super().__init__('mama_fama')
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.vol_window = vol_window

    def calculate(self, df):
        price = _safe_price_series(df)
        if len(price) == 0:
            return pd.DataFrame(index=df.index)

        vol = price.diff().abs() / (price.diff().abs().rolling(self.vol_window).mean() + 1e-12)
        alpha = (self.slow_limit + (self.fast_limit - self.slow_limit) * vol.clip(0, 1)).fillna(self.slow_limit)

        mama = np.full(len(price), np.nan, dtype=float)
        fama = np.full(len(price), np.nan, dtype=float)
        x = price.ffill().bfill().values

        if len(x) > 0:
            mama[0] = x[0]
            fama[0] = x[0]

        for i in range(1, len(x)):
            a = float(alpha.iloc[i])
            mama[i] = a * x[i] + (1 - a) * mama[i - 1]
            fa = 0.5 * a
            fama[i] = fa * mama[i] + (1 - fa) * fama[i - 1]

        result = pd.DataFrame(index=df.index)
        result['mama'] = mama
        result['fama'] = fama
        return result


class RoofingFilterMetric(MetricCalculator):
    def __init__(self, high_pass_period=48, smooth_period=10):
        super().__init__('roofing_filter')
        self.high_pass_period = high_pass_period
        self.smooth_period = smooth_period

    def calculate(self, df):
        price = _safe_price_series(df)
        hp = _high_pass_filter(price, self.high_pass_period)
        return _super_smoother_filter(hp, self.smooth_period)


class GaussianFilterMetric(MetricCalculator):
    def __init__(self, period=20, poles=3):
        super().__init__(f'gaussian_filter_{period}d_{poles}p')
        self.period = period
        self.poles = poles

    def calculate(self, df):
        return _gaussian_filter(_safe_price_series(df), self.period, self.poles)


class UltimateSmootherMetric(MetricCalculator):
    def __init__(self, period=20):
        super().__init__(f'ultimate_smoother_{period}d')
        self.period = period

    def calculate(self, df):
        price = _safe_price_series(df)
        first_pass = _super_smoother_filter(price, self.period)
        second_pass = _super_smoother_filter(first_pass, self.period)
        return 2 * first_pass - second_pass


# ===== EHLERS OSCILLATORS =====
class FisherTransformMetric(MetricCalculator):
    def __init__(self, period=10):
        super().__init__(f'fisher_{period}d')
        self.period = period

    def calculate(self, df):
        price = _safe_price_series(df)
        if len(price) == 0:
            return pd.Series(index=df.index, dtype=float)

        low = price.rolling(self.period).min()
        high = price.rolling(self.period).max()
        x = 2 * ((price - low) / ((high - low) + 1e-12) - 0.5)
        x = x.clip(-0.999, 0.999).fillna(0)

        v = np.zeros(len(x), dtype=float)
        fisher = np.zeros(len(x), dtype=float)
        for i in range(1, len(x)):
            v[i] = 0.33 * x.iloc[i] + 0.67 * v[i - 1]
            v[i] = min(0.999, max(-0.999, v[i]))
            fisher[i] = 0.5 * math.log((1 + v[i]) / (1 - v[i])) + 0.5 * fisher[i - 1]

        return pd.Series(fisher, index=df.index)


class LaguerreRSIMetric(MetricCalculator):
    def __init__(self, gamma=0.5):
        super().__init__('laguerre_rsi')
        self.gamma = gamma

    def calculate(self, df):
        price = _safe_price_series(df).ffill().bfill()
        if len(price) == 0:
            return pd.Series(index=df.index, dtype=float)

        g = self.gamma
        l0 = np.zeros(len(price))
        l1 = np.zeros(len(price))
        l2 = np.zeros(len(price))
        l3 = np.zeros(len(price))
        out = np.full(len(price), np.nan)
        x = price.values

        l0[0] = x[0]
        for i in range(1, len(x)):
            l0[i] = (1 - g) * x[i] + g * l0[i - 1]
            l1[i] = -g * l0[i] + l0[i - 1] + g * l1[i - 1]
            l2[i] = -g * l1[i] + l1[i - 1] + g * l2[i - 1]
            l3[i] = -g * l2[i] + l2[i - 1] + g * l3[i - 1]

            cu = max(l0[i] - l1[i], 0) + max(l1[i] - l2[i], 0) + max(l2[i] - l3[i], 0)
            cd = max(l1[i] - l0[i], 0) + max(l2[i] - l1[i], 0) + max(l3[i] - l2[i], 0)
            out[i] = cu / (cu + cd) if (cu + cd) != 0 else 0.5

        return pd.Series(out, index=df.index)


class EhlersStochasticMetric(MetricCalculator):
    def __init__(self, period=14, smooth_period=10):
        super().__init__(f'ehlers_stoch_{period}d')
        self.period = period
        self.smooth_period = smooth_period

    def calculate(self, df):
        price = _super_smoother_filter(_safe_price_series(df), self.smooth_period)
        low = price.rolling(self.period).min()
        high = price.rolling(self.period).max()
        return 100 * (price - low) / ((high - low) + 1e-12)


class CyberCycleMetric(MetricCalculator):
    def __init__(self, alpha=0.07):
        super().__init__('cyber_cycle')
        self.alpha = alpha

    def calculate(self, df):
        price = _safe_price_series(df).ffill().bfill()
        if len(price) == 0:
            return pd.Series(index=df.index, dtype=float)

        a = self.alpha
        x = price.values
        cycle = np.zeros(len(x), dtype=float)

        for i in range(2, len(x)):
            cycle[i] = ((1 - 0.5 * a) ** 2) * (x[i] - 2 * x[i - 1] + x[i - 2]) + \
                       2 * (1 - a) * cycle[i - 1] - ((1 - a) ** 2) * cycle[i - 2]

        return pd.Series(cycle, index=df.index)


class RVIMetric(MetricCalculator):
    def __init__(self, period=10):
        super().__init__(f'rvi_{period}d')
        self.period = period

    def calculate(self, df):
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(index=df.index, dtype=float)

        numerator = (df['close'] - df['open']).rolling(self.period).mean()
        denominator = (df['high'] - df['low']).rolling(self.period).mean()
        return numerator / (denominator + 1e-12)


class ReflexTrendflexMetric(MetricCalculator):
    def __init__(self, period=20):
        super().__init__(f'reflex_trendflex_{period}d')
        self.period = period

    def calculate(self, df):
        price = _safe_price_series(df)
        smooth = _super_smoother_filter(price, max(5, self.period // 2))
        trend = smooth - smooth.shift(self.period)

        reflex = (price - smooth) / (price.rolling(self.period).std() + 1e-12)
        trendflex = trend / (trend.rolling(self.period).std() + 1e-12)

        result = pd.DataFrame(index=df.index)
        result['reflex'] = reflex
        result['trendflex'] = trendflex
        return result


# ===== EHLERS CYCLE & REGIME =====
class HilbertSineWaveMetric(MetricCalculator):
    def __init__(self):
        super().__init__('hilbert_sine_wave')

    def calculate(self, df):
        price = _safe_price_series(df)
        if len(price) == 0:
            return pd.DataFrame(index=df.index)

        detrender = price - price.shift(7)
        i1 = detrender
        q1 = detrender.shift(2)

        phase = np.degrees(np.arctan2(q1.fillna(0), i1.fillna(0)))
        phase = pd.Series(np.unwrap(np.radians(phase)), index=df.index)
        phase_deg = np.degrees(phase)

        sine = np.sin(np.radians(phase_deg))
        lead_sine = np.sin(np.radians(phase_deg + 45))

        result = pd.DataFrame(index=df.index)
        result['hilbert_sine'] = sine
        result['hilbert_lead_sine'] = lead_sine
        return result


class HilbertTransformDiscriminatorMetric(MetricCalculator):
    def __init__(self):
        super().__init__('hilbert_discriminator')

    def calculate(self, df):
        price = _safe_price_series(df)
        if len(price) == 0:
            return pd.DataFrame(index=df.index)

        detrender = price - price.shift(7)
        phase = np.degrees(np.arctan2(detrender.shift(2).fillna(0), detrender.fillna(0)))
        phase = pd.Series(np.unwrap(np.radians(phase)), index=df.index)
        phase_deg = np.degrees(phase)

        phase_rate = phase_deg.diff().abs()
        mode = (phase_rate.rolling(10).std() < 12).astype(float)

        result = pd.DataFrame(index=df.index)
        result['hilbert_phase_deg'] = phase_deg
        result['hilbert_trend_mode'] = mode
        return result


class AutocorrelationPeriodogramMetric(MetricCalculator):
    def __init__(self, min_period=10, max_period=48, window=96):
        super().__init__('autocorr_periodogram')
        self.min_period = min_period
        self.max_period = max_period
        self.window = window

    def calculate(self, df):
        price = _safe_price_series(df).ffill().bfill()
        if len(price) == 0:
            return pd.DataFrame(index=df.index)

        dom_cycle = np.full(len(price), np.nan)
        strength = np.full(len(price), np.nan)
        x = price.values

        for i in range(self.window, len(x)):
            segment = pd.Series(x[i - self.window:i])
            corrs = []
            lags = range(self.min_period, min(self.max_period, self.window - 1) + 1)
            for lag in lags:
                corrs.append(segment.autocorr(lag=lag))

            if len(corrs) > 0 and np.any(np.isfinite(corrs)):
                corrs_arr = np.array(corrs, dtype=float)
                idx = int(np.nanargmax(np.abs(corrs_arr)))
                dom_cycle[i] = list(lags)[idx]
                strength[i] = corrs_arr[idx]

        result = pd.DataFrame(index=df.index)
        result['dominant_cycle_period'] = dom_cycle
        result['dominant_cycle_strength'] = strength
        return result


class EhlersDecyclerMetric(MetricCalculator):
    def __init__(self, period=30):
        super().__init__(f'decycler_{period}d')
        self.period = period

    def calculate(self, df):
        price = _safe_price_series(df)
        hp = _high_pass_filter(price, self.period)
        return price - hp


class ContinuationIndexMetric(MetricCalculator):
    def __init__(self, period=20, threshold=0.0):
        super().__init__('continuation_index')
        self.period = period
        self.threshold = threshold

    def calculate(self, df):
        price = _safe_price_series(df)
        smooth = _super_smoother_filter(price, self.period)
        slope = smooth.diff()
        ci = np.where(slope > self.threshold, 1, -1)
        return pd.Series(ci, index=df.index, dtype=float)


# ===== EHLERS FORECASTING =====
class VossPredictiveFilterMetric(MetricCalculator):
    def __init__(self, lookback=3):
        super().__init__('voss_predictive')
        self.lookback = lookback

    def calculate(self, df):
        price = _safe_price_series(df)
        lead = price + (price - price.shift(self.lookback))
        return _super_smoother_filter(lead, max(5, self.lookback * 2))


class InstantaneousTrendlineMetric(MetricCalculator):
    def __init__(self, alpha=0.07):
        super().__init__('instantaneous_trendline')
        self.alpha = alpha

    def calculate(self, df):
        price = _safe_price_series(df).ffill().bfill()
        if len(price) == 0:
            return pd.Series(index=df.index, dtype=float)

        a = self.alpha
        x = price.values
        itrend = np.full(len(x), np.nan, dtype=float)

        if len(x) > 0:
            itrend[0] = x[0]
        if len(x) > 1:
            itrend[1] = x[1]

        for i in range(2, len(x)):
            itrend[i] = (a - (a * a) / 4) * x[i] + 0.5 * a * a * x[i - 1] - \
                        (a - 0.75 * a * a) * x[i - 2] + 2 * (1 - a) * itrend[i - 1] - \
                        ((1 - a) ** 2) * itrend[i - 2]

        return pd.Series(itrend, index=df.index)


class MetricsEngine:
    """Orchestrates calculation of multiple metrics."""

    def __init__(self):
        self.calculators = []

    def register_metric(self, calculator):
        """Register a metric calculator."""
        if not isinstance(calculator, MetricCalculator):
            raise TypeError("Calculator must be instance of MetricCalculator")
        self.calculators.append(calculator)
        return self

    def register_default_metrics(self):
        """Register all metrics (basic and advanced)."""
        # Basic metrics
        self.register_metric(VWAPMetric(20))
        self.register_metric(VWAPMetric(50))
        self.register_metric(VWAPMetric(200))

        self.register_metric(SMAMetric(20))
        self.register_metric(SMAMetric(50))
        self.register_metric(SMAMetric(200))
        self.register_metric(EMAMetric(12))
        self.register_metric(EMAMetric(26))

        self.register_metric(VolatilityMetric(20))
        self.register_metric(VolatilityMetric(50))

        self.register_metric(MomentumMetric(21))
        self.register_metric(MomentumMetric(63))

        self.register_metric(RSIMetric(14))
        self.register_metric(VolumeMetric(20))
        self.register_metric(BollingerBandsMetric(20))
        self.register_metric(ATRMetric(14))

        # Advanced metrics
        self.register_metric(RVOLMetric(20))
        self.register_metric(RVOLMetric(50))
        self.register_metric(RVOLMetric(200))

        self.register_metric(OBVMetric())
        self.register_metric(OBVNormMetric(50))

        self.register_metric(ADLineMetric())
        self.register_metric(ADLineNormMetric(50))

        self.register_metric(DMIMetric(14))
        self.register_metric(ADXMetric(14))

        # Ehlers filters & trend
        self.register_metric(SuperSmootherMetric(10))
        self.register_metric(MAMAFAMAMetric())
        self.register_metric(RoofingFilterMetric(48, 10))
        self.register_metric(GaussianFilterMetric(20, 3))
        self.register_metric(UltimateSmootherMetric(20))

        # Ehlers oscillators
        self.register_metric(FisherTransformMetric(10))
        self.register_metric(LaguerreRSIMetric(0.5))
        self.register_metric(EhlersStochasticMetric(14, 10))
        self.register_metric(CyberCycleMetric(0.07))
        self.register_metric(RVIMetric(10))
        self.register_metric(ReflexTrendflexMetric(20))

        # Ehlers cycle & regime
        self.register_metric(HilbertSineWaveMetric())
        self.register_metric(HilbertTransformDiscriminatorMetric())
        self.register_metric(AutocorrelationPeriodogramMetric(10, 48, 96))
        self.register_metric(EhlersDecyclerMetric(30))
        self.register_metric(ContinuationIndexMetric(20, 0.0))

        # Ehlers forecasting
        self.register_metric(VossPredictiveFilterMetric(3))
        self.register_metric(InstantaneousTrendlineMetric(0.07))

        return self

    def calculate_all(self, df):
        """Calculate all registered metrics."""
        if df is None or len(df) == 0:
            return None

        result = df.copy()

        for calc in self.calculators:
            try:
                metric_result = calc.calculate(df)

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
    """Query a specific metric value for a specific date."""
    try:
        date = pd.to_datetime(date)
        if date in metrics_df.index and metric_name in metrics_df.columns:
            return metrics_df.loc[date, metric_name]
        return None
    except Exception:
        return None
