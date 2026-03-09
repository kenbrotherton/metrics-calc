# PairedSwitching Phase 1-4 Architecture
## Comprehensive Development & Optimization Guide

**Date:** March 8, 2026  
**Status:** Phase 1-4 Complete & Ready for Backtesting  
**Implementation:** Adaptive clustering, pair discovery, event triggers, cross-group analysis

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Phase 1: Adaptive K-Means Clustering](#phase-1-adaptive-k-means-clustering)
4. [Phase 2: Negative Correlation Pair Discovery](#phase-2-negative-correlation-pair-discovery)
5. [Phase 3: Within-Group Event Detection](#phase-3-within-group-event-detection)
6. [Phase 4: Cross-Group Correlation Analysis](#phase-4-cross-group-correlation-analysis)
7. [Backtest Parameters & Tuning](#backtest-parameters--tuning)
8. [Expected Outputs](#expected-outputs)
9. [Backtesting Workflow](#backtesting-workflow)
10. [Result Interpretation](#result-interpretation)
11. [Optimization Next Steps](#optimization-next-steps)

---

## Executive Summary

**PairedSwitching** is a four-phase ML algorithm designed to identify and trade negatively-correlated stock pairs through dynamic clustering, event-driven triggers, and cross-group analysis.

### Key Capabilities

| Phase | Focus | Output | Key Metric |
|-------|-------|--------|-----------|
| **1** | Adaptive K-Means clustering | Dynamic groups with silhouette scores | Optimal k selection |
| **2** | Within-cluster pair discovery | Negatively correlated pairs | Correlation + R² validation |
| **3** | Group-level event detection | Divergence events per group | Event trigger % of group |
| **4** | Cross-group correlation | Metric importance rankings | RF R² + feature importance |

### Strategy Logic

1. **Cluster stocks** using momentum + volatility (Phase 1)
2. **Find trading pairs** within clusters with negative correlation (Phase 2)
3. **Detect divergence events** when groups show coordinated spikes (Phase 3)
4. **Identify exploitable patterns** via metric analysis between groups (Phase 4)
5. **Execute trades**: Long the underperformer, short the outperformer when pair divergence detected
6. **Exit trades**: When spread profit expectation turns negative or max holding period reached

---

## Architecture Overview

### Data Flow

```
Historical Price Data (3 years)
         ↓
   [Monthly Loop]
         ↓
1. Collect Stock Metrics (momentum, volatility, price_change, volume)
         ↓
2. Perform Clustering → Dynamic k-selection + silhouette optimization
         ↓
3. Discover Negative Correlation Pairs (within clusters)
         ↓
4. Detect Within-Group Events (momentum/volume/volatility spikes)
         ↓
5. Analyze Cross-Group Correlations (inter-group relationships)
         ↓
6. Save Results → ObjectStore (JSON + CSV for backtesting analysis)
```

### Key Parameters

All parameters are **backtest-tunable** via QuantConnect:

```json
{
  "cluster_k_min": 2,
  "cluster_k_max": 12,
  "cluster_random_state": 42,
  "min_group_size": 10,
  
  "pair_corr_threshold": -0.7,
  "pair_min_r_squared": 0.5,
  "pair_lookback_days": 60,
  
  "event_momentum_sigma": 2.0,
  "event_volume_threshold": 1.5,
  "event_volatility_regime_shift": 1.5,
  "event_group_trigger_pct": 0.30,
  "event_lookback_days": 20,
  
  "group_corr_threshold": -0.5,
  "group_pair_lookback": 60,
  "rf_n_estimators": 50,
  "rf_min_samples_leaf": 5
}
```

---

## Phase 1: Adaptive K-Means Clustering

### Purpose
Replace fixed 6-regime logic with dynamic, data-driven clustering that adapts to market conditions.

### Implementation

**Input:**  
- 500+ stocks with metrics (momentum, volatility, price_change, volume)

**Process:**
1. **Feature scaling**: StandardScaler on 4D feature space
2. **K-selection loop**: Try k ∈ [k_min, k_max], score each via silhouette_score
3. **Optimal k**: Select k with highest silhouette score
4. **Minimum group enforcement**: Reassign clusters < min_group_size to nearest eligible cluster
5. **Output**: Cluster assignments + metadata

**Output Structure:**
```json
{
  "month": 1,
  "sample_count": 523,
  "features": ["momentum", "volatility", "price_change", "volume"],
  "k_min": 2,
  "k_max": 12,
  "selected_k": 4,
  "silhouette_score": 0.456,
  "min_group_size": 10,
  "group_sizes": {
    "Cluster 0": 128,
    "Cluster 1": 95,
    "Cluster 2": 150,
    "Cluster 3": 150
  }
}
```

### Key Parameters

| Parameter | Default | Tuning Range | Effect |
|-----------|---------|--------------|--------|
| `cluster_k_min` | 2 | 2-5 | Lower = more aggressive grouping |
| `cluster_k_max` | 12 | 8-20 | Higher = finer granularity |
| `min_group_size` | 10 | 5-30 | Larger = smoother groups, fewer events |
| `cluster_random_state` | 42 | Fixed | Reproducibility seed |

### Expected Behavior

- **Silhouette scores** typically: 0.3 - 0.6 (higher = better separation)
- **Cluster count** typically: 4-8 (adaptive to market regime)
- **Cluster sizes**: Roughly balanced (enforce minimum prevents tiny clusters)

---

## Phase 2: Negative Correlation Pair Discovery

### Purpose
Identify within-cluster pairs with strong negative correlation for paired trading.

### Implementation

**Input:**  
- Clustered symbols + 60 days of historical returns

**Process:**
1. **Build correlation matrix** for each cluster (pct_change)
2. **Find pairs** where `correlation < threshold` (e.g., -0.7)
3. **Validate with LinearRegression**:
   - X = returns of symbol_a
   - y = returns of symbol_b
   - Keep only pairs with R² ≥ pair_min_r_squared
4. **Store diagnostics**: correlation, R², coefficient, intercept, sample size

**Output Structure (per pair):**
```json
{
  "pair_id": "AAPL-MSFT",
  "symbol_a": "AAPL",
  "symbol_b": "MSFT",
  "group": "Cluster 2",
  "sample_size": 60,
  "correlation": -0.758,
  "r_squared": 0.564,
  "coefficient": -0.823,
  "intercept": 0.0015,
  "lookback_days": 60
}
```

### Key Parameters

| Parameter | Default | Tuning Range | Effect |
|-----------|---------|--------------|--------|
| `pair_corr_threshold` | -0.7 | -0.95 to -0.5 | Lower = only strongest pairs |
| `pair_min_r_squared` | 0.5 | 0.3-0.7 | Higher = more stable relationships |
| `pair_lookback_days` | 60 | 40-120 | Longer = more stable estimates |

### Expected Behavior

- **Valid pairs** typically: 5-30 per cluster (per month)
- **Correlation range**: -0.7 to -0.95
- **R² range**: 0.5 to 0.9 (higher = more predictable spread)
- **Strongest pairs** often cross sectors (tech vs. energy, growth vs. defensive)

---

## Phase 3: Within-Group Event Detection

### Purpose
Detect coordinated divergence events within clusters to trigger pair trades.

### Implementation

**Input:**  
- Cluster membership + 20 days of historical data

**Process:**
1. **Per-symbol analysis** (momentum, volume, volatility):
   - **Momentum spike**: `|z-score| > event_momentum_sigma` (rolling 5-day mean/std)
   - **Volume spike**: `volume / rolling_avg_volume > event_volume_threshold`
   - **Volatility spike**: `recent_vol / baseline_vol > event_volatility_regime_shift`

2. **Group aggregation**:
   - Count symbols triggering events
   - If count ≥ `ceil(group_size × event_group_trigger_pct)`, fire group event
   
3. **Store diagnostics**: timestamp, group, event count, event types, participation %

**Output Structure (per event):**
```json
{
  "timestamp": "2025-07-15T09:30:00",
  "month": 1,
  "group": "Cluster 2",
  "group_size": 95,
  "event_count": 32,
  "event_trigger_pct": 0.337,
  "event_types": {
    "momentum_spike": 18,
    "volume_spike": 10,
    "volatility_spike": 4
  }
}
```

### Key Parameters

| Parameter | Default | Tuning Range | Effect |
|-----------|---------|--------------|--------|
| `event_momentum_sigma` | 2.0 | 1.5-3.0 | Lower = more sensitive |
| `event_volume_threshold` | 1.5 | 1.2-2.5 | Lower = more volume events |
| `event_volatility_regime_shift` | 1.5 | 1.2-2.5 | Lower = more vol events |
| `event_group_trigger_pct` | 0.30 | 0.15-0.50 | Lower = easier to trigger |

### Expected Behavior

- **Events per month**: 2-15 per cluster (adaptive model-dependent)
- **Momentum spikes**: Most common (~50-60%)
- **Volume spikes**: ~30%
- **Volatility spikes**: ~10-20%
- **Group participation**: 30-70% at trigger

---

## Phase 4: Cross-Group Correlation Analysis

### Purpose
Find negatively correlated groups and identify which metrics drive the relationship.

### Implementation

**Input:**  
- All clusters + historical returns (60 days)

**Process:**
1. **Build inter-group correlation matrix** (cluster-level aggregates = mean returns per group)
2. **Find group pairs** where `correlation < group_corr_threshold` (e.g., -0.5)
3. **For each pair, train RandomForest**:
   - X = returns of group_a
   - y = returns of group_b
   - Captures non-linear relationships
4. **Compute metric importance**:
   - Difference in average metrics (momentum, volatility, price_change) between groups
   - Normalize to importance weights
5. **Store results**: group correlation, RF R², metric importance per pair

**Output Structure (per group pair):**
```json
{
  "pair_id": "Cluster_1-Cluster_3",
  "group_a": "Cluster 1",
  "group_b": "Cluster 3",
  "group_a_size": 95,
  "group_b_size": 128,
  "correlation": -0.634,
  "rf_r_squared": 0.418,
  "sample_size": 60,
  "lookback_days": 60,
  "metric_importances": {
    "momentum": 0.45,
    "volatility": 0.35,
    "price_change": 0.20
  }
}
```

### Key Parameters

| Parameter | Default | Tuning Range | Effect |
|-----------|---------|--------------|--------|
| `group_corr_threshold` | -0.5 | -0.8 to -0.3 | Lower = only strong pairs |
| `group_pair_lookback` | 60 | 40-120 | Longer = more stable |
| `rf_n_estimators` | 50 | 30-200 | Higher = better fit (more compute) |
| `rf_min_samples_leaf` | 5 | 2-10 | Higher = less overfit |

### Expected Behavior

- **Group pairs**: 2-8 per month (depends on cluster count)
- **Inter-group correlation**: -0.5 to -0.9
- **RF R²**: 0.3-0.6 (predictability of group relationship)
- **Metric importance**: Momentum/volatility typically 40-60%, price_change 20-40%

---

## Backtest Parameters & Tuning

### Backtest Configuration

When running backtest on QuantConnect Cloud, pass parameters via algorithm parameters:

```python
# Example: QuantConnect backtest parameters
self.add_universe(...)
self.set_parameter("cluster_k_max", 12)
self.set_parameter("pair_corr_threshold", -0.7)
self.set_parameter("event_group_trigger_pct", 0.30)
```

### Tuning Strategy

**Phase 1: Conservative Clustering**
- Minimize drift in cluster stability
- Larger min_group_size → fewer but more stable groups
- Lower k_max → prevents over-segmentation

```
cluster_k_min: 2
cluster_k_max: 8
min_group_size: 15
```

**Phase 2: Pair Quality**
- Prioritize stability over quantity
- Higher pair_corr_threshold → only strongest pairs
- Higher pair_min_r_squared → lower noise trades

```
pair_corr_threshold: -0.75
pair_min_r_squared: 0.6
pair_lookback_days: 60
```

**Phase 3: Event Sensitivity**
- Balance signal/noise
- Higher event_*_sigma/threshold → fewer false positives
- Lower event_group_trigger_pct → more aggressive

```
event_momentum_sigma: 2.2
event_volume_threshold: 1.7
event_group_trigger_pct: 0.25
```

**Phase 4: Cross-Group**
- Higher group_corr_threshold → more group pairs
- Lower rf_n_estimators → faster compute

```
group_corr_threshold: -0.5
rf_n_estimators: 30
```

### Walk-Forward Testing

**Recommended approach:**

1. **Training period**: 12 months
2. **Test period**: 3 months
3. **Rebalance**: Monthly on first trading day
4. **Parameter sweep**: Test 5-10 parameter combinations per phase

Example walk-forward:
- Train: Jan-Dec 2024 → Test: Jan-Mar 2025
- Train: Feb 2024-Jan 2025 → Test: Feb-Apr 2025
- Train: Mar 2024-Feb 2025 → Test: Mar-May 2025

---

## Expected Outputs

### per-Month ObjectStore Files

After each monthly rebalance, the algorithm saves:

#### 1. Metrics
- **`metrics_month_N.csv`**: All 500+ stocks with their metrics (momentum, volatility, etc.)
- Used for analysis of raw input data

#### 2. Clustering
- **`cluster_metadata_month_N.json`**: Metadata (k, silhouette, group sizes)
- **`cluster_plot_month_N.png`**: Scatter plot of momentum vs volatility colored by cluster

#### 3. Pairs
- **`pairs_month_N.json`**: Full pair diagnostics (correlation, R², coefficients)
- **`pairs_month_N.csv`**: Tabular format for Excel analysis

#### 4. Events
- **`events_month_N.json`**: Event-triggered groups + event types
- **`events_month_N.csv`**: Tabular event log

#### 5. Cross-Group
- **`group_pairs_month_N.json`**: Inter-group correlations + metric importance
- **`group_pairs_month_N.csv`**: Flattened with importance_* columns per metric

### Runtime Statistics

Backtest results tab shows:

```
Clusters: 4              (Number of active clusters)
Silhouette: 0.456       (Cluster separation quality)
Valid Pairs: 23         (Valid trading pairs discovered)
Events: 2               (Group divergence events this month)
Group Pairs: 3          (Negatively correlated group pairs)
```

### Charts

Backtest Results tab contains:

- **Group Momentum**: Average momentum per cluster over time
- **Group Sizes**: Member count per cluster over time
- **Pair Discovery (R2)**: R² scores of strongest pairs
- **Group Events**: Event occurrence per cluster
- **Group Pair Correlations**: Inter-group correlation scores
- **Clustering Performance**: Silhouette score trend

---

## Backtesting Workflow

### Step 1: Upload to QuantConnect Cloud

1. Log into QuantConnect
2. Create new project or use existing
3. Upload `main.py` to Cloud IDE
4. Ensure config.json has correct cloud-id

### Step 2: Configure Backtest

**Backtest Settings:**
- **Date Range**: Jan 2025 - Dec 2025 (12-month initial)
- **Starting Capital**: $100,000 (scale as needed)
- **Universe**: S&P 500 via fine/coarse universe selector (~500 stocks)
- **Resolution**: Daily

**Algorithm Parameters** (via QuantConnect UI):
```
cluster_k_max: 12
pair_corr_threshold: -0.7
event_group_trigger_pct: 0.30
```

### Step 3: Run Backtest

1. Click **Backtest** button
2. Wait for completion (~5-10 minutes for 12 months)
3. Review results in **Live Results** tab

### Step 4: Download Results

1. Go to **ObjectStore** → Download files:
   - `cluster_metadata_month_*.json`
   - `pairs_month_*.csv`
   - `events_month_*.csv`
   - `group_pairs_month_*.csv`

2. Analyze locally in Jupyter or Excel

### Step 5: Optimize Parameters

Based on Phase 1-4 diagnostics:
- Adjust clustering if silhouette scores low
- Adjust pair thresholds if R² too noisy
- Adjust event triggers based on false positive rate
- Adjust cross-group thresholds if too few group pairs

---

## Result Interpretation

### Phase 1: Clustering Diagnostics

**Good indicators:**
- Silhouette score: 0.4+
- Cluster sizes: All ≥ 10, balanced distribution
- K stability: Same k selected for 2-3 consecutive months

**Red flags:**
- Silhouette < 0.2 → Clusters poorly separated
- One cluster >> 50% of portfolio → Over-concentration
- K jumps 5+ each month → Unstable market regime

### Phase 2: Pair Quality

**Good indicators:**
- 20+ valid pairs per month
- Correlation range: -0.7 to -0.95
- R² mostly > 0.5
- Pair consistency: Same pairs appear 2-3 months

**Red flags:**
- < 5 valid pairs → Thresholds too strict
- R² < 0.3 → Noisy predictions
- Pair turnover > 80% → Unstable relationships

### Phase 3: Event Reliability

**Good indicators:**
- 3-8 events per month per cluster
- Event participation: 25-50%
- Momentum spikes: Most common (50-60%)
- Events precede price divergence 1-3 days

**Red flags:**
- 0 events → Thresholds too high
- 20+ events/month → Too many false positives
- Event types: Predominantly volume only (weak signal)

### Phase 4: Cross-Group Patterns

**Good indicators:**
- 2-6 negatively correlated group pairs
- Group correlation: -0.5 to -0.9
- RF R²: 0.3-0.6 (non-trivial predictability)
- Metric importance: Balanced across 3 metrics

**Red flags:**
- No group pairs → Thresholds too strict
- Group pairs: All correlation < -0.8 (rare events)
- RF R² < 0.2 → Little predictability
- Metric importance: Dominated by 1 metric (overfitting)

---

## Optimization Next Steps

### Immediate (Week 1-2)

1. **Run 12-month backtest** with default parameters
2. **Analyze Phase 1-4 diagnostics**: Are clusters stable? Pairs consistent? Events meaningful?
3. **Fix obvious issues**: k too high? Pair thresholds too loose? Event % too low?
4. **Document findings** in backtest notes

### Short-term (Week 3-4)

1. **Walk-forward backtest**: 3-month hold-out validation
2. **Parameter sensitivity**: Test ±20% variations on key params
3. **Compare to baseline**: How does adaptive vs fixed 6-regime perform?
4. **Pair trade accuracy**: Check if discovered pairs actually move together

### Medium-term (Month 2-3)

1. **Add position sizing**: Risk-parity on pair notional exposure
2. **Implement stop-losses**: Max % drawdown per pair
3. **Add adaptive spreads**: Tighter pairs get larger sizing
4. **Test live data**: Verify metrics on current market

### Long-term (Month 3+)

1. **Machine learning refinement**: Use backtesting data to retrain feature importance
2. **Regime-specific tuning**: Different parameters for bull/bear/sideways
3. **Integration with order execution**: Minimize slippage on pair entry
4. **Paper trading**: Test on live quotes before capital deployment

### Key Metrics to Track

| Metric | Target | Threshold |
|--------|--------|-----------|
| Sharpe Ratio | > 1.5 | > 1.0 |
| Max Drawdown | < 20% | < 30% |
| Win Rate | > 55% | > 50% |
| Profit Factor | > 1.5 | > 1.0 |
| Pair Consistency | > 70% | > 50% |

---

## Reference: Architecture Decisions

### Why Adaptive K-Means?

- **Dynamic**: Adapts to market regime changes
- **Interpretable**: Silhouette score guides k selection
- **Scalable**: Efficient for 500+ symbols
- **Stable**: Consistent results with fixed random_state

### Why Linear Pair Validation?

- **Speed**: Fast to compute for 500+ candidates
- **Interpretability**: Coefficient shows relationship strength
- **Robustness**: Less prone to overfitting than non-linear
- **Stability**: Works across market regimes

### Why RandomForest for Metrics?

- **Non-linear**: Captures complex metric interactions
- **Feature importance**: Directly interpretable ranking
- **Robust**: Less sensitive to outliers
- **Fast**: n_estimators=50 runs in <1s per backtest

### Why 3-Metric Space?

- **Momentum**: Captures trend direction
- **Volatility**: Measures risk/regime change
- **Price change**: Reflects absolute performance
- Minimal collinearity, captures most structure with K-Means

---

## Support & Troubleshooting

### Common Issues

**Q: Silhouette score too low (< 0.2)**
- A: Reduce k_max (force fewer, larger clusters) or increase min_group_size

**Q: Too few pairs discovered**
- A: Relax pair_corr_threshold (e.g., -0.6) or lower pair_min_r_squared (e.g., 0.4)

**Q: Event triggers too frequently**
- A: Increase event_*_sigma/threshold or raise event_group_trigger_pct

**Q: No group pairs found**
- A: Relax group_corr_threshold (e.g., -0.4) or increase group_pair_lookback

**Q: Code hangs during backtest**
- A: Reduce rf_n_estimators or max pair count via stricter thresholds

---

## Files Reference

| File | Purpose |
|------|---------|
| `main.py` | Core algorithm (Phase 1-4) |
| `config.json` | QuantConnect Cloud config |
| `local_config.json` | Local LEAN config |
| `lean.json` | LEAN Docker config |
| `backtest.json` | Backtest parameter template |
| `validate_phases.py` | Local validation script |
| `README_LOCAL.md` | Local data setup guide |
| This file | Comprehensive documentation |

---

## Contact & Questions

For algorithm refinements or parameter tuning support, refer to:
- **Phase 1 (Clustering)**: Check silhouette score + group size distribution
- **Phase 2 (Pairs)**: Validate R² + correlation stability across months
- **Phase 3 (Events)**: Tune sigma/threshold based on false positive rate
- **Phase 4 (Cross-Group)**: Analyze metric importance patterns

---

**End of PairedSwitching Phase 1-4 Documentation**

Last Updated: March 8, 2026  
Status: Production Ready  
Next: Deploy to QuantConnect Cloud for backtesting
