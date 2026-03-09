# PairedSwitching Project - Research Summary
**Date:** March 8, 2026  
**Purpose:** Understand current implementation, metrics, and gaps

---

## 1. PROJECT OVERVIEW

**PairedSwitching** is a QuantConnect LEAN algorithm that classifies stocks into market regimes and analyzes correlations between groups for potential paired trading strategies.

**Status:** Exploratory/Prototype - Core framework built, clustering working with fixed regimes, ML infrastructure in place but underutilized.

---

## 2. CURRENT IMPLEMENTATION

### 2.1 Main Algorithm (`main.py` - 636 lines)

**Purpose:** QuantConnect LEAN algorithm with live backtesting capabilities

**Key Components:**

| Component | Implementation | Status |
|-----------|-----------------|--------|
| **Universe Selection** | Coarse filter (top 1000 by $volume) → Fine filter (top 5 sectors, 100 stocks each) | ✅ Fully implemented |
| **Warmup Period** | 3 years (756 days) of historical data | ✅ Configured |
| **Metric Collection** | Manual calculation: price, price_change(3yr), momentum(21d), volatility, volume, direction | ✅ Implemented |
| **Clustering Strategy** | Fixed 6 Market Regimes (not k-means) | ✅ Implemented |
| **Regime Classification** | Based on momentum (±2% threshold) + volatility (1.5% threshold) | ✅ Implemented |
| **Correlation Analysis** | Cross-group correlation matrix calculation | ✅ Implemented |
| **Regression Analysis** | LinearRegression: metric vs group membership (R² scores) | ✅ Implemented |
| **Rebalancing** | Monthly schedule on month start | ✅ Implemented |

**Regime Definitions (Fixed):**
```
1. Calm Bull      → Momentum > 2%, Volatility < 1.5%
2. Volatile Bull  → Momentum > 2%, Volatility > 1.5%
3. Calm Bear      → Momentum < -2%, Volatility < 1.5%
4. Volatile Bear  → Momentum < -2%, Volatility > 1.5%
5. Calm Sideways  → -2% < Momentum < 2%, Volatility < 1.5%
6. Volatile Sideways → -2% < Momentum < 2%, Volatility > 1.5%
```

---

### 2.2 Metrics Processor System

#### A. **metrics_processor.py** (314 lines)
**Purpose:** Standalone script to pre-calculate rolling metrics from local data

**Capabilities:**
- Load symbols from local ZIP files (`/Lean/Data/equity/usa/daily/`)
- Parse OHLCV data (handles headerless CSV format)
- Calculate 30+ rolling metrics using MetricsEngine
- Save metrics per-symbol to timestamped run directories
- Create index and metadata files
- Query API for historical metrics

**Output Structure:**
```
processed_metrics/
├── run_20251018_20260309_004117/
│   ├── aapl_metrics.csv       (date-indexed with 30+ metric columns)
│   ├── msft_metrics.csv
│   ├── [500+ symbols]
│   ├── metadata.json          (symbols, date range, metric list)
│   └── index.csv              (symbol → file mapping)
```

**Current Runs:** 6 runs saved (400-560 symbols each), latest has 500+ symbols

#### B. **metrics_calculator.py** (460 lines)
**Purpose:** Modular metric calculation framework

**Metrics Implemented (30+):**

| Category | Metrics | Window/Period |
|----------|---------|----------------|
| **Basic Averages** | VWAP, SMA, EMA | 20d, 50d, 200d (varies) |
| **Momentum** | Momentum, RSI | 21d, 63d (momentum), 14d (RSI) |
| **Volatility** | Volatility (StdDev) | 20d, 50d |
| **Volume** | Volume, Avg Volume | 20d |
| **Bollinger Bands** | Upper, Middle, Lower | 20d window, 2σ |
| **Average True Range** | ATR | 14d |
| **Relative Volume** | RVOL | 20d, 50d, 200d |
| **On-Balance Volume** | OBV, OBV Normalized | cumulative, 50d norm |
| **Accumulation/Distribution** | A/D Line, A/D Normalized | cumulative, 50d norm |
| **Directional Movement** | DMI (+DI, -DI), ADX | 14d |

**Architecture:**
- Abstract base class `MetricCalculator` with `calculate(df)` method
- `MetricsEngine` orchestrates multi-metric calculation
- Results stored as date-indexed DataFrames per symbol
- Query API: `query_metric(symbol, date, metric_name)`

---

### 2.3 Research Notebooks

#### A. **research_local.ipynb** (13 cells)
**Status:** Exploratory/In-progress

**Content:**
1. Local data loading from ZIP files
2. Symbol property loading
3. Metric collection (manual calculation)
4. 6-regime fixed clustering
5. Visualization (scatter plots, histograms)
6. Group statistics and member listings
7. Export to CSV

**Key Finding:** Notebooks contain heavy lifting for data loading and clustering, notebooks validate the 6-regime approach

#### B. **metrics_processor_research.ipynb** (42 cells)
**Status:** Exploratory/Validation notebook

**Content:**
1. Test metrics_processor module
2. Validate metric outputs across symbols
3. Test visualization and data export
4. Check for data quality issues
5. Performance profiling

**Key Finding:** Comprehensive testing of metrics pipeline, validates 30+ metric calculations

---

## 3. DATA STRUCTURE

### 3.1 Available Data Sources
```
C:\Users\kenbr\QC\data\
├── equity/usa/daily/              → 500+ stock ZIP files (main trading data)
├── crypto/                         → Binance, Bitfinex, Bybit, Coinbase
├── forex/                          → FXCM, OANDA
├── futures/                        → CME, NYMEX, CBOT, ICE, EUREX, SGX, etc.
├── options/                        → US options
├── alternative/                    → AlphaStreams, Estimize, SEC, Trading Economics
├── market-hours/                   → market-hours-database.json
└── symbol-properties/              → symbol-properties-database.csv
```

### 3.2 Data Format
- **Format:** ZIP-compressed CSV files (headerless or with headers)
- **Columns:** time/date, open, high, low, close, volume, [openinterest]
- **Index:** Date (parsed from first column or 'time'/'date')
- **Frequency:** Daily OHLCV

### 3.3 Processed Metrics Output
- **Location:** `processed_metrics/run_[DATE]_[TIME]/`
- **Per-symbol files:** `{symbol}_metrics.csv` with date index + 30+ metric columns
- **Metadata:** `metadata.json`, `index.csv`
- **Current coverage:** 500-560 stocks per run

---

## 4. CONFIGURATION

### 4.1 **config.json** (Cloud)
- Cloud ID: 28604153
- Organization ID: 175306019d9b7d2de260c12740704abe
- Backtesting grid with 7 charts for results visualization
- Charts: Group Momentum, Group Sizes, Inter-Group Correlations, Group Divergence, Clustering Performance, etc.

### 4.2 **local_config.json** (Local)
```json
{
    "data-folder": "C:\\Users\\kenbr\\QC\\data",
    "algorithm-location": "main.py",
    "algorithm-type-name": "PairedSwitching",
    "environment": "backtesting"
}
```

---

## 5. SK-LEARN / TENSORFLOW USAGE

### Current Usage
| Library | Import | Usage Status | Location |
|---------|--------|--------|----------|
| `sklearn.cluster.KMeans` | ✅ Imported | ⚠️ **Not actively used** (infrastructure only) | main.py:11 |
| `sklearn.linear_model.LinearRegression` | ✅ Imported | ✅ **Active** - R² regression between groups/metrics | main.py:12, lines 551-565 |
| `sklearn.metrics.silhouette_score` | ✅ Imported | ⚠️ **Not used** (infrastructure only) | main.py:13 |
| `sklearn.preprocessing` | ✅ Imported | ⚠️ **Not used** (infrastructure only) | main.py:11 |

### Why KMeans not used?
Current implementation uses **fixed 6 regimes** instead of dynamic k-means clustering:
- Regime assignment deterministic (momentum + volatility thresholds)
- KMeans infrastructure present but dormant
- Could enable dynamic clustering with silhouette optimization

---

## 6. WHAT'S FULLY IMPLEMENTED ✅

| Feature | Evidence | Notes |
|---------|----------|-------|
| Data loading from local ZIP files | metrics_processor.py, research_local.ipynb | Handles 500+ symbols |
| 30+ rolling metrics calculation | metrics_calculator.py (460 lines) | Comprehensive technical analysis suite |
| Fixed 6-regime clustering | main.py lines 249-310 | Deterministic classification |
| Group statistics (avg metrics per regime) | main.py lines 312-364 | Calculates mean/median per group |
| Correlation matrix analysis | main.py lines 387-450 | Cross-group price/volume correlations |
| Linear regression analysis | main.py lines 451-565 | R² scores for metric importance |
| Monthly rebalancing | main.py line 48 | Scheduled event |
| Visualization & charting | main.py (O/L/R 50+), config.json | Backtesting grid configured |
| Metrics persistence | processed_metrics_dir (6 runs) | 500+ symbols × 30+ metrics stored |
| Local backtesting setup | local_config.json, research_local.ipynb | Docker + local data mounted |

---

## 7. WHAT'S IN-PROGRESS / EXPLORATORY 🔄

| Feature | Current State | Next Steps |
|---------|---------------|----|
| K-means dynamic clustering | Infrastructure present, not active | Implement elbow method + silhouette score optimization |
| Event triggering | Not implemented | Define trading signals from regime changes/divergences |
| Cross-group pair trading | Analysis exists, no trade logic | Implement reversion strategy when divergence detected |
| Feature engineering | Basic technical metrics only | Add correlation distance, divergence scores, entropy metrics |
| ML feature classification | Not started | Predict regime changes or correlation shifts |
| Risk management | Minimal (no position sizing) | Add kelly criterion, drawdown management |

---

## 8. GAPS BETWEEN CURRENT STATE & GOALS

### Gap 1: Correlation Analysis
**Current:** Manual group averaging + basic correlation matrix
**Goal:** Advanced correlation-based pair identification
```python
# Current: Simple group mean calculation
group_series[group_name] = data_df[group_syms].mean(axis=1)

# Potential: Correlation distance matrix
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
```
**Missing:** 
- Correlation distance-based clustering (not just momentum/volatility)
- Hierarchy clustering for pair identification
- Dynamic correlation regimes

---

### Gap 2: K-Means Clustering
**Current:** Fixed 6 regimes (deterministic thresholds)
**Goal:** Dynamic k-means with optimal k selection
```python
# Ready but inactive:
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Implementation needed:
# - Normalize features (preprocessing.StandardScaler)
# - Elbow method + silhouette analysis
# - Dynamic k selection based on data
```
**Missing:**
- Feature scaling before k-means
- Silhouette score optimization loop
- Comparison: fixed 6 vs. optimal k

---

### Gap 3: Event Triggering
**Current:** None - only monthly rebalancing
**Goal:** Detect regime changes, correlation breaks, divergence events
**Missing:**
- Month-to-month regime change detection
- Correlation breakdown alerts
- Volatility spike identification
- Inter-group divergence events

---

### Gap 4: Group Analysis
**Current:** Basic group averages + R² regression scores
**Goal:** Rich statistical group profiles
```python
# Implemented:
- avg price, momentum, volatility, volume (simple means)
- R² correlation with group membership
- Basic correlation matrix

# Missing:
- Correlation distance within/between groups
- Group stability metrics (how often do stocks move between regimes)
- Regime persistence probability
- Beta relative to SPY per group
- Entropy/diversity within groups
- Centroid-distance metrics for classification confidence
```

---

### Gap 5: Advanced ML
**Current:** LinearRegression only, no predictive models
**Goal:** Predictive ML for regime/correlation
**Missing:**
- Classification models (regime prediction)
- Time series models (ARIMA, Prophet for regime duration)
- Neural networks (LSTM for sequence prediction)
- Feature interaction analysis
- Hyperparameter tuning

---

## 9. RECOMMENDATIONS FOR NEXT STEPS

### Phase 1: Quick Wins (1-2 weeks)
1. **Activate K-Means with silhouette optimization**
   - Compare fixed 6 vs. optimal k
   - Visualize elbow curve
   
2. **Add correlation distance clustering**
   - Use scipy.spatial.distance for correlation-based pairs
   - Identify most correlated stock pairs per regime

3. **Event trigger detection**
   - Flag month-to-month regime changes
   - Detect correlation divergence (inter-group > intra-group)

### Phase 2: Medium Term (3-4 weeks)
4. **Group stability analysis**
   - Probability of regime persistence
   - Classification confidence scores
   - Volatility of group assignments

5. **Machine learning baseline**
   - Random Forest for regime classification
   - Feature importance analysis
   - Cross-validation on historical data

### Phase 3: Long Term (ongoing)
6. **Predictive models**
   - LSTM for regime duration prediction
   - GRU for correlation shift forecasting
   - Anomaly detection for regime breaks

7. **Trading strategy**
   - Implement actual pair trades from divergences
   - Position sizing (Kelly criterion)
   - Risk management

---

## 10. KEY FILES REFERENCE

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `main.py` | Main LEAN algorithm | 636 | ✅ Production-ready |
| `metrics_processor.py` | Batch metrics calculation | 314 | ✅ Functional |
| `metrics_calculator.py` | Metric definitions (30+) | 460 | ✅ Complete |
| `metrics_processor_research.ipynb` | Validation notebook | 42 cells | 🔄 Exploratory |
| `research_local.ipynb` | Local research notebook | 13 cells | 🔄 Exploratory |
| `config.json` | Cloud backtest config | 303 lines | ✅ Configured |
| `local_config.json` | Local config | 7 lines | ✅ Configured |
| `test_ehlers_metrics.py` | (empty) | 2 lines | ❌ Not started |

---

## 11. DATA SCIENCE INSIGHTS

### Current Metrics Quality
- **Technical Analysis:** Comprehensive (30+ indicators)
- **Fundamental Analysis:** None (not in scope for technical strategy)
- **Alternative Data:** Available but not used (crypto, forex in data folder)

### Clustering Observations
- **Fixed 6 regimes:** Simple, interpretable, deterministic
- **Limitation:** May not reflect actual market structure (e.g., tech vs financials behave differently)
- **Opportunity:** K-means could discover natural clusters (e.g., 4-5 clusters may be optimal)

### Correlation Findings (from viz in main.py)
- Inter-group correlations tracked monthly
- Group divergence calculated
- Baseline (SPY) correlations monitored
- **Note:** Currently just exploratory, no trading signals yet

---

## 12. PROJECT MATURITY ASSESSMENT

| Aspect | Maturity | Assessment |
|--------|----------|------------|
| Data Pipeline | 🟢 Production | 500+ symbols, metrics stored, query API |
| Clustering | 🟡 Prototype | Fixed 6 regimes, but KMeans ready |
| Correlation Analysis | 🟡 Research | Basic implementation, needs depth |
| Trading Logic | 🔴 Not Started | Framework present, no actual trades |
| Risk Management | 🔴 Minimal | No position sizing, no drawdown controls |
| Backtesting | 🟡 Setup | LEAN configured, needs real strategies |
| ML/AI | 🔴 Dormant | Imports present, not utilized |

---

## SUMMARY TABLE

```
┌─────────────────────────────────────────────────────────────────┐
│               PAIREDSWITCHING PROJECT STATUS                    │
├─────────────────────────────────────────────────────────────────┤
│ Architecture:         ✅ Solid (QuantConnect LEAN + Python)    │
│ Data Pipeline:        ✅ Complete (500+ symbols, 30+ metrics)  │
│ Clustering:           🟡 Basic (6 fixed regimes, K-means ready)│
│ Correlation Analysis: 🟡 Functional (basic matrices, no signals)│
│ Trading Strategy:     🔴 Missing (framework only, no trades)   │
│ ML/AI Integration:    🔴 Dormant (infrastructure, not active)  │
│ Research Status:      🟡 Exploratory (2 notebooks active)      │
│ Production Readiness: 🔴 Not ready for live trading yet       │
└─────────────────────────────────────────────────────────────────┘

Total Code: ~1800 lines Python (main + metrics + processor)
Data Coverage: 500-560 equities per run, 3 years history
Metrics: 30+ technical indicators per stock
Clustering: 6 market regimes identified

Recommended Focus: K-means optimization → Event triggers → ML models
```

---

*Research compiled: March 8, 2026*  
*Data: 500+ equity symbols, 3-year history, 30+ technical metrics*
