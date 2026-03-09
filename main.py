#region imports
from AlgorithmImports import *

# Data manipulation and numerical computing
import pandas as pd
import numpy as np

# Scikit-learn for clustering and regression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
from pathlib import Path
#endregion


class PairedSwitching(QCAlgorithm):
    
    def initialize(self):
        """Initialize the algorithm with settings and universe."""
        # Initialize log buffer
        self._log_content = "Algorithm Initialized\n"
        
        self.set_start_date(2025, 3, 15)
        self.set_cash(100000)

        # Local backtest mode (price-only universe, no coarse/fine fundamental dependency)
        self._local_data_mode = (self.get_parameter("local_data_mode") or "false").lower() == "true"
        self._local_symbols = [
            s.strip().upper()
            for s in (self.get_parameter("local_symbols") or "SPY,AAPL,MSFT,AMZN,NVDA,META,TSLA,JPM,XOM,UNH").split(",")
            if s and s.strip()
        ]
        self._include_agg = (self.get_parameter("include_agg") or "false").lower() == "true"

        # Auto-fallback for local daily-only datasets (no coarse fundamental files)
        if not self._local_data_mode:
            coarse_path = Path("/Lean/Data/equity/usa/fundamental/coarse")
            if (not coarse_path.exists()) or (not any(coarse_path.glob("*.csv"))):
                self._local_data_mode = True
        
        # Primary assets
        self.spy = self.add_equity("SPY", Resolution.DAILY)
        self.agg = self.add_equity("AGG", Resolution.DAILY) if self._include_agg else None
        
        # Warmup period (3 years of historical data)
        self.set_warmup(timedelta(days=756))
        
        # Data storage
        self._price_data = {}
        self._market_cap_data = {}
        self._current_universe = set()
        self._correlation_groups = {}
        self._group_labels = {}
        self._months = 0
        self._last_clustering_metadata = {}

        # Adaptive clustering parameters (phase 1), overridable via backtest parameters
        self._cluster_k_min = int(self.get_parameter("cluster_k_min") or 2)
        self._cluster_k_max = int(self.get_parameter("cluster_k_max") or 12)
        self._cluster_random_state = int(self.get_parameter("cluster_random_state") or 42)
        self._cluster_n_init = int(self.get_parameter("cluster_n_init") or 10)
        self._min_group_size = int(self.get_parameter("min_group_size") or 10)

        # Pair discovery parameters (phase 2)
        self._pair_corr_threshold = float(self.get_parameter("pair_corr_threshold") or -0.7)
        self._pair_min_r_squared = float(self.get_parameter("pair_min_r_squared") or 0.5)
        self._pair_lookback_days = int(self.get_parameter("pair_lookback_days") or 60)
        self._discovered_pairs = []

        # Pair trading execution controls (minimal first pass)
        self._max_pair_positions = int(self.get_parameter("max_pair_positions") or 3)
        self._pair_position_size = float(self.get_parameter("pair_position_size") or 0.10)
        self._min_hold_days = int(self.get_parameter("min_hold_days") or 3)
        self._max_hold_days = int(self.get_parameter("max_hold_days") or 30)
        self._pending_pair_candidates = []
        self._active_pair_positions = {}
        self._enable_test_pair_fallback = (self.get_parameter("enable_test_pair_fallback") or "true").lower() == "true"

        # Event detection parameters (phase 3)
        self._event_momentum_sigma = float(self.get_parameter("event_momentum_sigma") or 2.0)
        self._event_volume_threshold = float(self.get_parameter("event_volume_threshold") or 1.5)  # multiplier of rolling avg
        self._event_volatility_regime_shift = float(self.get_parameter("event_volatility_regime_shift") or 1.5)
        self._event_group_trigger_pct = float(self.get_parameter("event_group_trigger_pct") or 0.30)  # 30% of members
        self._event_lookback_days = int(self.get_parameter("event_lookback_days") or 20)
        self._monthly_events = []

        # Cross-group analysis parameters (phase 4)
        self._group_corr_threshold = float(self.get_parameter("group_corr_threshold") or -0.5)  # inter-group neg correlation threshold
        self._group_pair_lookback = int(self.get_parameter("group_pair_lookback") or 60)
        self._rf_n_estimators = int(self.get_parameter("rf_n_estimators") or 50)
        self._rf_min_samples_leaf = int(self.get_parameter("rf_min_samples_leaf") or 5)
        self._group_pair_results = []

        # Add SPY as baseline to universe
        self._current_universe.add(self.spy.symbol)
        self.universe_settings.resolution = Resolution.DAILY

        if self._local_data_mode:
            # Explicit local universe for daily price-only datasets
            for ticker in self._local_symbols:
                symbol = self.add_equity(ticker, Resolution.DAILY).symbol
                self._current_universe.add(symbol)
        else:
            # Use coarse/fine universe selection for S&P 500 stocks
            self.add_universe(self._coarse_filter, self._fine_filter)
        
        # Schedule daily rebalancing
        self.schedule.on(
            self.date_rules.every_day("SPY"),
            self.time_rules.after_market_open("SPY", 1),
            self._daily_rebalance
        )


    
    def _coarse_filter(self, coarse):
        """Filter stocks by price and volume."""
        # Select stocks with fundamental data and price > 1
        filtered = [x for x in coarse if x.has_fundamental_data and x.price > 1]
        
        # Sort by Dollar Volume to get the most liquid stocks
        # We take top 1000 to ensure we have enough candidates for each sector in fine selection
        sorted_by_dollar_vol = sorted(filtered, key=lambda x: x.dollar_volume, reverse=True)
        selected = [x.symbol for x in sorted_by_dollar_vol[:1000]]
        
        # self.debug(f"Coarse filter: {len(selected)} symbols selected")
        return selected
    
    def _fine_filter(self, fine):
        """Select top 100 stocks by market cap in each sector."""
        # Group stocks by Morningstar Sector Code
        sector_dict = {}
        sector_mcap = {}
        
        for f in fine:
            if f.market_cap == 0: continue
            sector = f.asset_classification.morningstar_sector_code
            if sector not in sector_dict:
                sector_dict[sector] = []
                sector_mcap[sector] = 0
            sector_dict[sector].append(f)
            sector_mcap[sector] += f.market_cap
        
        # Select Top 5 Sectors by total market cap
        top_5_sectors = sorted(sector_mcap, key=lambda x: sector_mcap[x], reverse=True)[:5]
        
        selected = []
        # Select top 100 by market cap for each of the top 5 sectors
        for sector in top_5_sectors:
            sorted_sector = sorted(sector_dict[sector], key=lambda x: x.market_cap, reverse=True)
            selected.extend([x.symbol for x in sorted_sector[:100]])
        
        # self.debug(f"Fine filter: {len(selected)} stocks selected across {len(sector_dict)} sectors")
        return selected
    
    def on_securities_changed(self, changes):
        """Track universe additions and removals."""
        if len(changes.added_securities) > 0:
            # self.debug(f"Adding {len(changes.added_securities)} securities to universe")
            for security in changes.added_securities:
                self._current_universe.add(security.symbol)
        
        if len(changes.removed_securities) > 0:
            # self.debug(f"Removing {len(changes.removed_securities)} securities from universe")
            for security in changes.removed_securities:
                self._current_universe.discard(security.symbol)
    
    def _daily_rebalance(self):
        """Daily event handler to retrain clustering model."""
        self._months += 1
        
        # Skip during warmup
        if self.is_warming_up:
            return
        
        # Perform analysis after warmup (skip first 7 days)
        if self._months >= 7:
            # self.debug(f"\n{'=' * 100}")
            # self.debug(f"DAILY REBALANCING - Day {self._months}")
            # self.debug(f"{'=' * 100}")
            
            # Collect stock metrics
            metrics_data = self._collect_stock_metrics()
            
            if metrics_data is not None and len(metrics_data) > 0:
                # Perform clustering
                self._perform_clustering(metrics_data)
                
                # Generate group report
                self._generate_group_report(metrics_data)
                
                # Perform regression analysis
                self._perform_regression_analysis(metrics_data)
                
                # Discover negative correlation pairs (Phase 2)
                self._discover_negative_correlation_pairs(metrics_data)
                
                # Detect within-group events (Phase 3)
                self._detect_group_events(metrics_data)

                # Build trade candidates from discovered pairs/events
                self._refresh_pair_candidates(metrics_data)
                
                # Cross-group correlation analysis (Phase 4)
                self._analyze_cross_group_metrics(metrics_data)
                
                # Analyze correlations BETWEEN groups (The Grid)
                self._analyze_group_correlations()
                
                # Save visual and data outputs
                self._save_monthly_outputs(metrics_data)
    
    def _collect_stock_metrics(self):
        """Collect price, momentum, and other metrics for all stocks in universe."""
        # Ensure we only pass valid Symbol objects to history
        symbols = [s for s in self._current_universe if isinstance(s, Symbol)]
        
        if not symbols:
            # self.debug("No symbols in universe")
            return None
        
        # self.debug(f"Universe contains {len(symbols)} symbols")
        # self.debug(f"Collecting metrics for {len(symbols)} stocks...")
        
        try:
            # Fetch historical data (1 year for momentum calculation)
            history = self.history(symbols, 252, Resolution.DAILY)

            
            if history.empty:
                # self.debug("No historical data retrieved")
                return None
            
            metrics = {}
            
            # Create mapping for Ticker -> Symbol to handle potential string keys in history
            ticker_to_symbol = {str(s.value): s for s in symbols}
            
            # Iterate through the data we actually have using groupby
            # This avoids checking 'if symbol in available_symbols' and handling KeyErrors manually
            # level=0 is the Symbol index
            for identifier, sym_history in history.groupby(level=0):
                
                # Resolve the identifier to a Symbol object
                symbol_obj = None
                if isinstance(identifier, Symbol):
                    symbol_obj = identifier
                elif isinstance(identifier, str):
                    symbol_obj = ticker_to_symbol.get(identifier)
                
                # If we can't map it to a known symbol, skip
                if symbol_obj is None:
                    continue
                
                # Need sufficient data
                if len(sym_history) < 252 * 0.5:
                    continue
                
                # Extract data safely
                if 'close' not in sym_history.columns:
                    continue
                    
                close_prices = np.array(sym_history['close'].values, dtype=np.float64)
                
                # Handle volume if present
                if 'volume' in sym_history.columns:
                    volumes = np.array(sym_history['volume'].values, dtype=np.float64)
                    avg_volume = np.mean(volumes)
                else:
                    avg_volume = 0
                
                # Calculate metrics
                current_price = close_prices[-1]
                
                # Price Change
                start_price = close_prices[0]
                price_change = (current_price - start_price) / start_price if start_price != 0 else 0
                
                # Momentum (21 days)
                if len(close_prices) > 21:
                    prev_price = close_prices[-21]
                    momentum = (current_price - prev_price) / prev_price if prev_price != 0 else 0
                else:
                    momentum = 0
                
                # Volatility
                if len(close_prices) > 1:
                    prices_t = close_prices[1:]
                    prices_t_minus_1 = close_prices[:-1]
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        returns = (prices_t - prices_t_minus_1) / prices_t_minus_1
                    
                    returns = returns[np.isfinite(returns)]
                    
                    if len(returns) > 0:
                        volatility = np.std(returns)
                    else:
                        volatility = 0
                else:
                    volatility = 0
                
                direction = 1 if price_change > 0 else -1
                
                metrics[symbol_obj] = {
                    'price': current_price,
                    'price_change': price_change,
                    'momentum': momentum,
                    'volatility': volatility,
                    'volume': avg_volume,
                    'direction': direction
                }
            
            # self.debug(f"Collected metrics for {len(metrics)} stocks")
            
            if len(metrics) > 0:
                return pd.DataFrame(metrics).T
            else:
                return None
            
        except Exception as e:
            # self.debug(f"Error collecting metrics: {str(e)}")
            return None
    
    def _perform_clustering(self, metrics_df):
        """Cluster stocks using adaptive k-selection on standardized feature space."""
        if metrics_df is None or len(metrics_df) < 3:
            return

        try:
            feature_cols = ['momentum', 'volatility', 'price_change', 'volume']
            available_features = [c for c in feature_cols if c in metrics_df.columns]
            if len(available_features) < 2:
                return

            feature_df = metrics_df[available_features].copy().replace([np.inf, -np.inf], np.nan).dropna()
            if len(feature_df) < 3:
                return

            scaler = preprocessing.StandardScaler()
            X = scaler.fit_transform(feature_df.values)

            n_samples = len(feature_df)
            k_min = max(2, min(self._cluster_k_min, n_samples - 1))
            k_max = max(k_min, min(self._cluster_k_max, n_samples - 1))

            best_score = -1.0
            best_k = k_min
            best_labels = None
            best_centers = None

            for k in range(k_min, k_max + 1):
                model = KMeans(
                    n_clusters=k,
                    random_state=self._cluster_random_state,
                    n_init=self._cluster_n_init
                )
                labels = model.fit_predict(X)

                unique_labels = np.unique(labels)
                if len(unique_labels) < 2:
                    score = -1.0
                else:
                    score = float(silhouette_score(X, labels))

                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels.copy()
                    best_centers = model.cluster_centers_.copy()

            if best_labels is None:
                return

            labels = best_labels.copy()
            centers = best_centers.copy()

            # Enforce minimum group size by reassigning small clusters to nearest eligible cluster
            cluster_sizes = pd.Series(labels).value_counts().to_dict()
            small_clusters = [cluster_id for cluster_id, size in cluster_sizes.items() if size < self._min_group_size]

            if small_clusters:
                for cluster_id in small_clusters:
                    member_idx = np.where(labels == cluster_id)[0]
                    if len(member_idx) == 0:
                        continue

                    eligible_targets = [
                        cid for cid, size in cluster_sizes.items()
                        if cid != cluster_id and size >= self._min_group_size
                    ]

                    if not eligible_targets:
                        continue

                    source_center = centers[cluster_id]
                    target_centers = centers[eligible_targets]
                    distances = np.linalg.norm(target_centers - source_center, axis=1)
                    target_cluster = eligible_targets[int(np.argmin(distances))]

                    labels[member_idx] = target_cluster
                    cluster_sizes[target_cluster] += len(member_idx)
                    cluster_sizes[cluster_id] = 0

            # Build groups and drop empty clusters
            groups_temp = {}
            symbols = feature_df.index.tolist()
            for idx, cluster_id in enumerate(labels):
                group_name = f"Cluster {int(cluster_id)}"
                if group_name not in groups_temp:
                    groups_temp[group_name] = []
                groups_temp[group_name].append(symbols[idx])

            self._correlation_groups = {k: v for k, v in groups_temp.items() if len(v) > 0}

            final_sizes = {group_name: len(members) for group_name, members in self._correlation_groups.items()}
            self._last_clustering_metadata = {
                "month": int(self._months),
                "sample_count": int(n_samples),
                "features": available_features,
                "k_min": int(k_min),
                "k_max": int(k_max),
                "selected_k": int(best_k),
                "silhouette_score": float(best_score),
                "min_group_size": int(self._min_group_size),
                "group_sizes": final_sizes
            }

            self.set_runtime_statistic("Clusters", len(self._correlation_groups))
            self.set_runtime_statistic("Silhouette", round(float(best_score), 3))
            self.plot("Clustering Performance", "Cluster Count", len(self._correlation_groups))
            self.plot("Clustering Performance", "Silhouette Score", float(best_score))

        except Exception:
            pass
    
    def _generate_group_report(self, metrics_df):
        """Generate and print group report."""
        if not self._correlation_groups:
            # self.debug("No groups to report")
            return
        
        # self.debug("\n" + "=" * 100)
        # self.debug("GROUP REPORT")
        # self.debug("=" * 100)
        
        self._group_labels = {}
        
        for group_name, symbols in sorted(self._correlation_groups.items()):
            if not symbols:
                continue
            
            # self.debug(f"\n{'─' * 100}")
            # self.debug(f"GROUP {group_id} - {len(symbols)} MEMBERS")
            # self.debug(f"{'─' * 100}")
            
            # Group statistics
            group_metrics = metrics_df.loc[symbols]
            
            avg_price = group_metrics['price'].mean()
            avg_price_change = group_metrics['price_change'].mean()
            avg_momentum = group_metrics['momentum'].mean()
            avg_volatility = group_metrics['volatility'].mean()
            avg_volume = group_metrics['volume'].mean()
            
            # Plot group momentum to the Results tab
            self.plot("Group Momentum", f"Group {group_name}", avg_momentum)
            
            # Plot group size
            self.plot("Group Sizes", f"Group {group_name}", len(symbols))
            
            # self.debug(f"GROUP METRICS:")
            # self.debug(f"  Average Price: ${avg_price:.2f}")
            # self.debug(f"  Average Price Change: {avg_price_change*100:.2f}%")
            # self.debug(f"  Average Momentum (21d): {avg_momentum*100:.2f}%")
            # self.debug(f"  Average Volatility: {avg_volatility*100:.2f}%")
            # self.debug(f"  Average Volume: {avg_volume:,.0f}")
            
            # Characterize the group
            # Use fixed label
            self._group_labels[group_name] = group_name
            
            # Member list
            # self.debug(f"\nMEMBERS ({len(symbols)} stocks):")
            sorted_symbols = sorted(symbols)
            
            for i in range(0, len(sorted_symbols), 5):
                row = sorted_symbols[i:i+5]
                # self.debug(f"  {', '.join(row)}")
        
        # self.debug("\n" + "=" * 100)
    
    def _perform_regression_analysis(self, metrics_df):
        """Perform regression analysis between groups and metrics."""
        if not self._correlation_groups:
            # self.debug("No groups for regression analysis")
            return
        
        try:
            # self.debug(f"\nPerforming regression analysis between groups...")
            
            # Create group assignment series
            group_assignments = {}
            
            # Map names to integers for regression
            unique_groups = sorted(self._correlation_groups.keys())
            name_to_id = {name: i for i, name in enumerate(unique_groups)}
            
            for group_name, symbols in self._correlation_groups.items():
                for symbol in symbols:
                    group_assignments[symbol] = name_to_id[group_name]
            
            group_series = pd.Series(group_assignments)
            
            # Perform regression for each metric
            results = {}
            
            for metric_col in ['price', 'price_change', 'momentum', 'volatility', 'volume']:
                try:
                    # Align data
                    metric_data = metrics_df[metric_col]
                    common_symbols = group_series.index.intersection(metric_data.index)
                    
                    X = group_series[common_symbols].values.reshape(-1, 1)
                    y = metric_data[common_symbols].values
                    
                    # Fit regression
                    model = LinearRegression()
                    model.fit(X, y)
                    r_squared = model.score(X, y)
                    
                    results[metric_col] = r_squared
                    
                    # Plot the R-squared value to visualize correlation strength over time
                    self.plot("Cluster Feature Importance (R2)", metric_col, float(r_squared))
                    
                except Exception as e:
                    results[metric_col] = 0.0
            
            # Generate regression report
            self._generate_regression_report(results)
            
        except Exception as e:
            # self.debug(f"Error in regression analysis: {str(e)}")
            pass
    
    def _generate_regression_report(self, results):
        """Generate and print regression report."""
        # self.debug("\n" + "=" * 100)
        # self.debug("REGRESSION ANALYSIS REPORT - Group Metrics Correlation")
        # self.debug("=" * 100)
        
        # self.debug(f"\nR-squared values indicate how well each metric correlates with group membership:")
        # self.debug(f"(Higher = stronger correlation between metric and group)\n")
        
        # Sort by R-squared descending
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        for metric, r_squared in sorted_results:
            strength = "STRONG" if r_squared > 0.5 else "MODERATE" if r_squared > 0.25 else "WEAK"
            # self.debug(f"  {metric:<20} R² = {r_squared:.4f}  [{strength}]")
        
        # Summary
        # self.debug("\nINTERPRETATION:")
        high_correlation = [m for m, r2 in results.items() if r2 > 0.5]
        
        if high_correlation:
            # self.debug(f"  Strong group distinction factors: {', '.join(high_correlation)}")
            pass
        else:
            # self.debug(f"  Groups show weak metric-based distinction - may indicate alternative grouping factors")
            pass
        
        # self.debug("\n" + "=" * 100)
    
    def _discover_negative_correlation_pairs(self, metrics_df):
        """
        Phase 2: Discover negative correlation pairs within clusters.
        For each cluster:
        1. Build correlation matrix from historical returns
        2. Find pairs with correlation < threshold (default -0.7)
        3. Validate with LinearRegression to ensure stable relationship
        4. Store pair results with ID, correlation, R², coefficients
        """
        if not self._correlation_groups or len(self._correlation_groups) == 0:
            return

        try:
            pairs_results = []
            
            # Fetch historical data for all symbols (for price-based correlation)
            all_symbols = []
            group_map = {}  # map symbol -> group_name
            
            for group_name, symbols in self._correlation_groups.items():
                for sym in symbols:
                    if isinstance(sym, Symbol):
                        all_symbols.append(sym)
                        group_map[sym] = group_name
            
            if len(all_symbols) < 3:
                return
            
            # Get historical prices for correlation calculation
            history = self.history(all_symbols, self._pair_lookback_days, Resolution.DAILY)
            if history.empty:
                return
            
            closes = history['close'].unstack(level=0) if 'close' in history.columns else None
            if closes is None or closes.empty:
                return
            
            # Normalize columns to Symbol objects
            ticker_to_symbol = {}
            for s in all_symbols:
                ticker_to_symbol[str(s.value)] = s
                ticker_to_symbol[str(s)] = s
            
            def normalize_columns(df):
                if df.empty: return df
                new_cols = {}
                for c in df.columns:
                    if isinstance(c, str):
                        sym = ticker_to_symbol.get(c)
                        if sym:
                            new_cols[c] = sym
                return df.rename(columns=new_cols)
            
            closes = normalize_columns(closes)
            returns = closes.pct_change().dropna()
            
            if returns.empty or len(returns) < 10:
                return
            
            # For each cluster, find negatively correlated pairs
            for group_name, symbols in self._correlation_groups.items():
                group_symbols = [s for s in symbols if isinstance(s, Symbol) and s in returns.columns]
                
                if len(group_symbols) < 2:
                    continue
                
                # Build correlation matrix for this cluster
                cluster_returns = returns[group_symbols]
                corr_matrix = cluster_returns.corr()
                
                # Find pairs with correlation < threshold (negative correlation)
                for i in range(len(group_symbols)):
                    for j in range(i + 1, len(group_symbols)):
                        sym_a, sym_b = group_symbols[i], group_symbols[j]
                        corr_value = float(corr_matrix.at[sym_a, sym_b])
                        
                        if corr_value >= self._pair_corr_threshold:
                            continue  # Not negative enough
                        
                        try:
                            # Validate pair relationship with LinearRegression
                            X = cluster_returns[sym_a].values.reshape(-1, 1)
                            y = cluster_returns[sym_b].values
                            
                            model = LinearRegression()
                            model.fit(X, y)
                            r_squared = float(model.score(X, y))
                            coefficient = float(model.coef_[0])
                            intercept = float(model.intercept_)
                            
                            # Only keep pairs with sufficient R² (predictive power)
                            if r_squared >= self._pair_min_r_squared:
                                pair_result = {
                                    "pair_id": f"{str(sym_a)}-{str(sym_b)}",
                                    "symbol_a": str(sym_a),
                                    "symbol_b": str(sym_b),
                                    "ticker_a": str(sym_a.value),
                                    "ticker_b": str(sym_b.value),
                                    "group": group_name,
                                    "sample_size": len(X),
                                    "correlation": round(corr_value, 4),
                                    "r_squared": round(r_squared, 4),
                                    "coefficient": round(coefficient, 4),
                                    "intercept": round(intercept, 6),
                                    "lookback_days": self._pair_lookback_days
                                }
                                pairs_results.append(pair_result)
                                
                                # Plot top pairs
                                if r_squared > 0.6:  # Strong pairs
                                    self.plot("Pair Discovery (R2)", 
                                             f"{str(sym_a)}x{str(sym_b)}", 
                                             float(r_squared))
                        
                        except Exception:
                            continue
            
            # Store discovered pairs
            self._discovered_pairs = sorted(pairs_results, key=lambda x: x['r_squared'], reverse=True)
            
            if self._discovered_pairs:
                self.set_runtime_statistic("Valid Pairs", len(self._discovered_pairs))
                self.plot("Pair Discovery Count", "Total Valid Pairs", len(self._discovered_pairs))
        
        except Exception:
            pass
    
    def _detect_group_events(self, metrics_df):
        """
        Phase 3: Detect within-group divergence events.
        
        For each symbol in each cluster:
        1. Track momentum spike (>2σ from rolling mean)
        2. Track volume spike (>1.5x rolling average)
        3. Track volatility regime shift (>1.5x baseline)
        
        Aggregate to group level:
        - If N% of group members trigger event, emit group-level event
        - Store event with timestamp, group, event count, event types
        """
        if not self._correlation_groups or len(self._correlation_groups) == 0:
            return

        try:
            all_symbols = []
            group_map = {}
            
            for group_name, symbols in self._correlation_groups.items():
                for sym in symbols:
                    if isinstance(sym, Symbol):
                        all_symbols.append(sym)
                        group_map[sym] = group_name
            
            if len(all_symbols) < 2:
                return
            
            # Get historical data for event detection
            history = self.history(all_symbols, self._event_lookback_days + 5, Resolution.DAILY)
            if history.empty:
                return
            
            closes = history['close'].unstack(level=0) if 'close' in history.columns else None
            volumes = history['volume'].unstack(level=0) if 'volume' in history.columns else None
            
            if closes is None or closes.empty:
                return
            
            # Normalize columns
            ticker_to_symbol = {}
            for s in all_symbols:
                ticker_to_symbol[str(s.value)] = s
                ticker_to_symbol[str(s)] = s
            
            def normalize_columns(df):
                if df.empty: return df
                new_cols = {}
                for c in df.columns:
                    if isinstance(c, str):
                        sym = ticker_to_symbol.get(c)
                        if sym:
                            new_cols[c] = sym
                return df.rename(columns=new_cols)
            
            closes = normalize_columns(closes)
            if volumes is not None:
                volumes = normalize_columns(volumes)
            
            returns = closes.pct_change().dropna()
            if returns.empty or len(returns) < 5:
                return
            
            # Detect individual symbol events
            symbol_events = {}  # symbol -> list of event types this month
            
            for symbol in all_symbols:
                if symbol not in closes.columns:
                    continue
                
                events_for_symbol = []
                
                try:
                    # 1. Momentum spike detection
                    sym_returns = returns[symbol]
                    if len(sym_returns) >= 5:
                        rolling_mean = sym_returns.rolling(window=5).mean()
                        rolling_std = sym_returns.rolling(window=5).std()
                        latest_return = sym_returns.iloc[-1]
                        latest_mean = rolling_mean.iloc[-1]
                        latest_std = rolling_std.iloc[-1]
                        
                        if latest_std > 0 and latest_mean > 0:
                            z_score = abs((latest_return - latest_mean) / latest_std)
                            if z_score >= self._event_momentum_sigma:
                                events_for_symbol.append("momentum_spike")
                    
                    # 2. Volume spike detection
                    if volumes is not None and symbol in volumes.columns:
                        sym_volumes = volumes[symbol]
                        if len(sym_volumes) >= 5:
                            rolling_avg_volume = sym_volumes.rolling(window=20).mean()
                            latest_volume = sym_volumes.iloc[-1]
                            rolling_avg = rolling_avg_volume.iloc[-1]
                            
                            if rolling_avg > 0 and (latest_volume / rolling_avg) >= self._event_volume_threshold:
                                events_for_symbol.append("volume_spike")
                    
                    # 3. Volatility regime shift
                    if len(sym_returns) >= 20:
                        recent_vol = sym_returns.iloc[-5:].std()
                        baseline_vol = sym_returns.iloc[-20:-5].std()
                        
                        if baseline_vol > 0 and (recent_vol / baseline_vol) >= self._event_volatility_regime_shift:
                            events_for_symbol.append("volatility_spike")
                
                except Exception:
                    continue
                
                if events_for_symbol:
                    symbol_events[symbol] = events_for_symbol
            
            # Aggregate events to group level
            group_events = {}
            
            for group_name, symbols in self._correlation_groups.items():
                group_symbols = [s for s in symbols if isinstance(s, Symbol) and s in symbol_events]
                
                if len(symbols) < 2:
                    continue
                
                event_trigger_threshold = max(1, int(np.ceil(len(symbols) * self._event_group_trigger_pct)))
                
                if len(group_symbols) >= event_trigger_threshold:
                    event_types = {}
                    for sym in group_symbols:
                        for evt_type in symbol_events[sym]:
                            event_types[evt_type] = event_types.get(evt_type, 0) + 1
                    
                    event_record = {
                        "timestamp": self.utc_time.isoformat(),
                        "month": int(self._months),
                        "group": group_name,
                        "group_size": len(symbols),
                        "event_count": len(group_symbols),
                        "event_trigger_pct": round(len(group_symbols) / len(symbols), 3),
                        "event_types": event_types
                    }
                    
                    group_events[group_name] = event_record
                    
                    # Plot event occurrence
                    self.plot("Group Events", f"{group_name} Events", len(group_symbols))
            
            self._monthly_events = list(group_events.values())
            
            if self._monthly_events:
                self.set_runtime_statistic("Events", len(self._monthly_events))
        
        except Exception:
            pass
    
    def _analyze_cross_group_metrics(self, metrics_df):
        """
        Phase 4: Cross-group correlation and metrics analysis.
        
        1. Build inter-group correlation matrix using cluster-level aggregates
        2. Identify negatively correlated group pairs
        3. Use RandomForest to rank which metrics drive group correlation
        4. Store results with feature importance
        """
        if not self._correlation_groups or len(self._correlation_groups) < 2:
            return

        try:
            all_symbols = []
            group_map = {}
            
            for group_name, symbols in self._correlation_groups.items():
                for sym in symbols:
                    if isinstance(sym, Symbol):
                        all_symbols.append(sym)
                        group_map[sym] = group_name
            
            if len(all_symbols) < 3:
                return
            
            # Get historical data for inter-group correlation
            history = self.history(all_symbols, self._group_pair_lookback + 5, Resolution.DAILY)
            if history.empty:
                return
            
            closes = history['close'].unstack(level=0) if 'close' in history.columns else None
            if closes is None or closes.empty:
                return
            
            # Normalize columns
            ticker_to_symbol = {}
            for s in all_symbols:
                ticker_to_symbol[str(s.value)] = s
                ticker_to_symbol[str(s)] = s
            
            def normalize_columns(df):
                if df.empty: return df
                new_cols = {}
                for c in df.columns:
                    if isinstance(c, str):
                        sym = ticker_to_symbol.get(c)
                        if sym:
                            new_cols[c] = sym
                return df.rename(columns=new_cols)
            
            closes = normalize_columns(closes)
            returns = closes.pct_change().dropna()
            
            if returns.empty or len(returns) < 10:
                return
            
            # Build group-level aggregates
            group_returns = {}
            for group_name, symbols in self._correlation_groups.items():
                group_symbols = [s for s in symbols if isinstance(s, Symbol) and s in returns.columns]
                if len(group_symbols) >= 2:
                    group_returns[group_name] = returns[group_symbols].mean(axis=1)
            
            if len(group_returns) < 2:
                return
            
            # Build inter-group correlation matrix
            group_return_df = pd.DataFrame(group_returns)
            inter_group_corr = group_return_df.corr()
            
            pair_results = []
            
            # Find negatively correlated group pairs
            group_names = list(group_returns.keys())
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    group_a, group_b = group_names[i], group_names[j]
                    corr_value = float(inter_group_corr.at[group_a, group_b])
                    
                    if corr_value >= self._group_corr_threshold:
                        continue  # Not negative enough
                    
                    try:
                        # Get symbols from both groups
                        symbols_a = [s for s in self._correlation_groups[group_a] if isinstance(s, Symbol) and s in metrics_df.index]
                        symbols_b = [s for s in self._correlation_groups[group_b] if isinstance(s, Symbol) and s in metrics_df.index]
                        
                        if len(symbols_a) < 2 or len(symbols_b) < 2:
                            continue
                        
                        # Extract metrics for both groups
                        metrics_cols = ['momentum', 'volatility', 'price_change']
                        available_metrics = [c for c in metrics_cols if c in metrics_df.columns]
                        
                        if len(available_metrics) < 2:
                            continue
                        
                        # Build feature matrix: concat both group metrics
                        group_a_metrics = metrics_df.loc[symbols_a, available_metrics].mean(axis=0).values.reshape(1, -1)
                        group_b_metrics = metrics_df.loc[symbols_b, available_metrics].mean(axis=0).values.reshape(1, -1)
                        
                        # For regression, use group returns as X and relationship as y
                        # We want to predict group_b from group_a using RandomForest
                        X_train = group_return_df[group_a].values.reshape(-1, 1)
                        y_train = group_return_df[group_b].values
                        
                        # Fit RandomForest for trend prediction
                        rf_model = RandomForestRegressor(
                            n_estimators=self._rf_n_estimators,
                            min_samples_leaf=self._rf_min_samples_leaf,
                            random_state=self._cluster_random_state
                        )
                        rf_model.fit(X_train, y_train)
                        rf_r2 = float(rf_model.score(X_train, y_train))
                        
                        # Now use metrics to explain the pair relationship
                        # Build feature matrix from metrics of both groups
                        scaler = StandardScaler()
                        group_a_avg_metrics = metrics_df.loc[symbols_a, available_metrics].mean(axis=0).values
                        group_b_avg_metrics = metrics_df.loc[symbols_b, available_metrics].mean(axis=0).values
                        
                        # Difference in metrics explains the divergence
                        metric_diff = group_b_avg_metrics - group_a_avg_metrics
                        
                        # Create importance mapping
                        metric_importances = {}
                        for idx, metric in enumerate(available_metrics):
                            metric_importances[metric] = float(abs(metric_diff[idx]))
                        
                        # Normalize importances to sum to 1
                        total_importance = sum(metric_importances.values())
                        if total_importance > 0:
                            metric_importances = {k: v / total_importance for k, v in metric_importances.items()}
                        
                        pair_result = {
                            "pair_id": f"{group_a}-{group_b}",
                            "group_a": group_a,
                            "group_b": group_b,
                            "group_a_size": len(symbols_a),
                            "group_b_size": len(symbols_b),
                            "correlation": round(corr_value, 4),
                            "rf_r_squared": round(rf_r2, 4),
                            "sample_size": len(X_train),
                            "lookback_days": self._group_pair_lookback,
                            "metric_importances": metric_importances
                        }
                        
                        pair_results.append(pair_result)
                        
                        # Plot strong group pairs
                        if corr_value < -0.6:
                            self.plot("Group Pair Correlations", f"{group_a} vs {group_b}", float(corr_value))
                    
                    except Exception:
                        continue
            
            # Store and track results
            self._group_pair_results = sorted(pair_results, key=lambda x: abs(x['correlation']), reverse=True)
            
            if self._group_pair_results:
                self.set_runtime_statistic("Group Pairs", len(self._group_pair_results))
                self.plot("Cross-Group Pairs Count", "Total Pairs", len(self._group_pair_results))
        
        except Exception:
            pass
    
    def _analyze_group_correlations(self):
        """
        Calculate and visualize correlations between groups based on daily returns.
        Generates a Heatmap (Grid) saved to ObjectStore and plots summary stats.
        """
        if not self._correlation_groups:
            return

        # 1. Get historical data for all symbols in groups
        all_symbols = []
        for symbols in self._correlation_groups.values():
            all_symbols.extend(symbols)
        
        # Ensure we only have valid Symbol objects
        all_symbols = [s for s in all_symbols if isinstance(s, Symbol)]
        
        # Add Baseline (SPY) to history request
        if self.spy.symbol not in all_symbols:
            all_symbols.append(self.spy.symbol)
            
        # Get data: 60 days for correlation + 21 days buffer for momentum calc
        history = self.history(all_symbols, 81, Resolution.DAILY)
        if history.empty:
            return

        # Prepare DataFrames
        # unstack(level=0) moves Symbol from index to columns
        closes = history['close'].unstack(level=0)
        volumes = history['volume'].unstack(level=0)
        
        # Normalize columns to Symbol objects to ensure lookups work
        ticker_to_symbol = {}
        for s in all_symbols:
            ticker_to_symbol[str(s.value)] = s
            ticker_to_symbol[str(s)] = s
            
        def normalize_columns(df):
            if df.empty: return df
            new_cols = {}
            for c in df.columns:
                if isinstance(c, str):
                    sym = ticker_to_symbol.get(c)
                    if sym:
                        new_cols[c] = sym
            return df.rename(columns=new_cols)

        closes = normalize_columns(closes)
        volumes = normalize_columns(volumes)
        
        # Define metrics to analyze
        metrics_to_analyze = {
            "Price Returns": closes.pct_change().dropna(),
            "Volume Changes": volumes.pct_change().dropna(),
            "Momentum Trends": closes.pct_change(21).dropna(),
            "Price Levels": closes
        }
        
        for metric_name, data_df in metrics_to_analyze.items():
            # 2. Calculate Group Indices (Average of members)
            group_series = {}
            
            # Add Baseline
            for group_name, symbols in self._correlation_groups.items():
                # Filter for symbols present in this group and in the data
                group_syms = [s for s in symbols if s in data_df.columns]
                if not group_syms:
                    continue
                # Average the metric across stocks in the group for each day
                group_series[group_name] = data_df[group_syms].mean(axis=1)

            if not group_series:
                continue

            group_df = pd.DataFrame(group_series)
            
            # 3. Calculate Correlation Matrix
            corr_matrix = group_df.corr()
            
            # 4. Print Matrix to Console (The "Grid") with Labels
            #self.debug(f"\nINTER-GROUP {metric_name.upper()} CORRELATION (Month {self._months}):\n" + corr_matrix.to_string(float_format=lambda x: "{:.2f}".format(x)))
            
            # 5. Plotting
            # A. Inter-Group Structure (Price Returns Only)
            if metric_name == "Price Returns":
                # Mask diagonal to find true min/max
                mask = np.ones(corr_matrix.shape, dtype=bool)
                np.fill_diagonal(mask, 0)
                
                if len(corr_matrix) > 1:
                    # Calculate Inter-Group stats (excluding Baseline for "Inter-Group" chart consistency)
                    group_mask = [c in self._correlation_groups for c in corr_matrix.columns]
                    if sum(group_mask) > 1:
                        # Use integer indexing (iloc) with boolean mask to avoid QuantConnect PandasMapper issues
                        group_corr = corr_matrix.iloc[group_mask, group_mask]
                        
                        mask_g = np.ones(group_corr.shape, dtype=bool)
                        np.fill_diagonal(mask_g, 0)
                        
                        self.plot("Inter-Group Correlations", "Avg Correlation", float(group_corr.values[mask_g].mean()))
                        self.plot("Inter-Group Correlations", "Min Correlation", float(group_corr.values[mask_g].min()))
                
                # Plot individual group divergence
                for i, col in enumerate(corr_matrix.columns):
                    if col not in self._correlation_groups: continue
                    
                    # Divergence vs Rest (other groups)
                    other_mask = [(c in self._correlation_groups and c != col) for c in corr_matrix.columns]
                    if any(other_mask):
                        # Use .values to bypass QuantConnect Pandas wrapper
                        avg_val = corr_matrix.values[i, other_mask].mean()
                        if pd.notna(avg_val):
                            self.plot("Group Divergence", f"{col} vs Rest", float(avg_val))

            # B. Baseline Correlations (For Price Returns, Momentum, and Price Levels)
            chart_name = None
            if metric_name == "Price Returns":
                chart_name = "Baseline Correlations"
            elif metric_name == "Momentum Trends":
                chart_name = "Baseline Momentum Correlations"
            elif metric_name == "Price Levels":
                chart_name = "Baseline Price Correlations"
            
            if chart_name:
                for i, col in enumerate(corr_matrix.columns):
                    if col not in self._correlation_groups: continue

                    # Correlation vs Baseline
                    baseline_indices = [idx for idx, c in enumerate(corr_matrix.columns) if str(c) == "Baseline"]
                    if baseline_indices:
                        # Use .values to bypass QuantConnect Pandas wrapper
                        val = corr_matrix.values[i, baseline_indices[0]]
                        if pd.notna(val):
                            self.plot(chart_name, f"{col} vs SPY", float(val))

    def _save_monthly_outputs(self, metrics_df):
        """Save CSV data, clustering metadata, pair discovery results, event logs, and charts to ObjectStore."""
        try:
            # 1. Save Metrics to CSV
            csv_key = f"metrics_month_{self._months}.csv"
            self.object_store.save(csv_key, metrics_df.to_csv())

            # 2. Save clustering metadata for diagnostics/optimization
            metadata_key = f"cluster_metadata_month_{self._months}.json"
            self.object_store.save(metadata_key, json.dumps(self._last_clustering_metadata, default=str))

            # 3. Save discovered pairs to JSON and CSV (Phase 2)
            if self._discovered_pairs and len(self._discovered_pairs) > 0:
                pairs_json_key = f"pairs_month_{self._months}.json"
                self.object_store.save(pairs_json_key, json.dumps(self._discovered_pairs, default=str))
                
                # Also save as CSV for easy analysis
                pairs_df = pd.DataFrame(self._discovered_pairs)
                pairs_csv_key = f"pairs_month_{self._months}.csv"
                self.object_store.save(pairs_csv_key, pairs_df.to_csv(index=False))

            # 4. Save detected events to JSON and CSV (Phase 3)
            if self._monthly_events and len(self._monthly_events) > 0:
                events_json_key = f"events_month_{self._months}.json"
                self.object_store.save(events_json_key, json.dumps(self._monthly_events, default=str))
                
                # Also save as CSV for easy analysis
                events_df = pd.DataFrame(self._monthly_events)
                events_csv_key = f"events_month_{self._months}.csv"
                self.object_store.save(events_csv_key, events_df.to_csv(index=False))

            # 5. Save cross-group analysis results to JSON and CSV (Phase 4)
            if self._group_pair_results and len(self._group_pair_results) > 0:
                group_pairs_json_key = f"group_pairs_month_{self._months}.json"
                self.object_store.save(group_pairs_json_key, json.dumps(self._group_pair_results, default=str))
                
                # Save CSV with key metrics (flatten importances for CSV)
                group_pairs_for_csv = []
                for pair in self._group_pair_results:
                    pair_copy = pair.copy()
                    # Create separate columns for each metric importance
                    for metric, importance in pair['metric_importances'].items():
                        pair_copy[f"importance_{metric}"] = importance
                    del pair_copy['metric_importances']
                    group_pairs_for_csv.append(pair_copy)
                
                group_pairs_df = pd.DataFrame(group_pairs_for_csv)
                group_pairs_csv_key = f"group_pairs_month_{self._months}.csv"
                self.object_store.save(group_pairs_csv_key, group_pairs_df.to_csv(index=False))

            # 6. Generate and Save Cluster Plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=metrics_df, x='volatility', y='momentum', 
                            hue=self._get_labels_for_plot(metrics_df), palette='viridis')
            plt.title(f'Clustering: Momentum vs Volatility (Month {self._months})')
            plt.xlabel('Volatility')
            plt.ylabel('Momentum')

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            self.object_store.save_bytes(f"cluster_plot_month_{self._months}.png", list(img_buf.getvalue()))
            plt.close()

        except Exception:
            pass

    def _get_labels_for_plot(self, df):
        """Helper to reconstruct labels list for plotting."""
        labels = []
        # Create a mapping from symbol to group
        symbol_to_group = {}
        for group_id, symbols in self._correlation_groups.items():
            for sym in symbols:
                symbol_to_group[sym] = group_id
        
        for sym in df.index:
            labels.append(symbol_to_group.get(sym, "Unknown"))
        return labels

    def _resolve_symbol(self, ticker):
        """Resolve ticker text to an active Symbol in the current algorithm."""
        ticker = str(ticker).upper()
        for symbol in self._current_universe:
            if isinstance(symbol, Symbol) and str(symbol.value).upper() == ticker:
                return symbol
        return None

    def _refresh_pair_candidates(self, metrics_df):
        """Build monthly trade candidates from discovered pairs and group events."""
        self._pending_pair_candidates = []
        if not self._discovered_pairs:
            return

        event_groups = {evt.get("group") for evt in self._monthly_events} if self._monthly_events else set()
        candidate_rows = []

        for pair in self._discovered_pairs:
            ticker_a = pair.get("ticker_a") or pair.get("symbol_a")
            ticker_b = pair.get("ticker_b") or pair.get("symbol_b")

            if not ticker_a or not ticker_b:
                continue

            symbol_a = self._resolve_symbol(ticker_a)
            symbol_b = self._resolve_symbol(ticker_b)
            if symbol_a is None or symbol_b is None:
                continue

            if symbol_a not in metrics_df.index or symbol_b not in metrics_df.index:
                continue

            # Require either group event trigger or very strong statistical relationship
            has_event_trigger = pair.get("group") in event_groups
            is_strong_pair = float(pair.get("r_squared", 0.0)) >= 0.65
            if not (has_event_trigger or is_strong_pair):
                continue

            mom_a = float(metrics_df.at[symbol_a, "momentum"]) if "momentum" in metrics_df.columns else 0.0
            mom_b = float(metrics_df.at[symbol_b, "momentum"]) if "momentum" in metrics_df.columns else 0.0

            # Mean-reversion direction: long laggard, short leader
            if mom_a <= mom_b:
                long_symbol, short_symbol = symbol_a, symbol_b
            else:
                long_symbol, short_symbol = symbol_b, symbol_a

            candidate_rows.append({
                "pair_id": pair.get("pair_id", f"{ticker_a}-{ticker_b}"),
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "score": float(pair.get("r_squared", 0.0)) + abs(float(pair.get("correlation", 0.0))),
                "correlation": float(pair.get("correlation", 0.0)),
                "r_squared": float(pair.get("r_squared", 0.0))
            })

        candidate_rows = sorted(candidate_rows, key=lambda x: x["score"], reverse=True)

        # Fallback: if no statistical pairs are available, create event-driven momentum divergence pairs
        if not candidate_rows and self._correlation_groups:
            source_groups = list(event_groups) if event_groups else list(self._correlation_groups.keys())
            for group_name in source_groups:
                members = self._correlation_groups.get(group_name, [])
                valid_members = [s for s in members if isinstance(s, Symbol) and s in metrics_df.index]
                if len(valid_members) < 2 or "momentum" not in metrics_df.columns:
                    continue

                ranked = sorted(
                    valid_members,
                    key=lambda s: float(metrics_df.at[s, "momentum"])
                )
                long_symbol = ranked[0]
                short_symbol = ranked[-1]
                if long_symbol == short_symbol:
                    continue

                divergence = abs(float(metrics_df.at[short_symbol, "momentum"]) - float(metrics_df.at[long_symbol, "momentum"]))
                candidate_rows.append({
                    "pair_id": f"fallback-{group_name}-{long_symbol.value}-{short_symbol.value}",
                    "long_symbol": long_symbol,
                    "short_symbol": short_symbol,
                    "score": float(divergence),
                    "correlation": 0.0,
                    "r_squared": 0.0
                })

        candidate_rows = sorted(candidate_rows, key=lambda x: x["score"], reverse=True)
        self._pending_pair_candidates = candidate_rows[: max(0, self._max_pair_positions * 2)]
        self.set_runtime_statistic("Pair Candidates", len(self._pending_pair_candidates))

    def on_data(self, data):
        """Process daily pair trade entries/exits."""
        if self.is_warming_up:
            return

        # Exit logic: hold until no longer profitable (after minimum hold), or max hold reached
        active_keys = list(self._active_pair_positions.keys())
        for pair_key in active_keys:
            position = self._active_pair_positions.get(pair_key)
            if position is None:
                continue

            long_symbol = position["long_symbol"]
            short_symbol = position["short_symbol"]
            entry_time = position["entry_time"]

            if long_symbol not in self.portfolio or short_symbol not in self.portfolio:
                self._active_pair_positions.pop(pair_key, None)
                continue

            days_held = max(0, (self.time.date() - entry_time.date()).days)
            pnl = float(self.portfolio[long_symbol].unrealized_profit + self.portfolio[short_symbol].unrealized_profit)

            should_exit_for_profit_decay = days_held >= self._min_hold_days and pnl <= 0
            should_exit_for_time = days_held >= self._max_hold_days

            if should_exit_for_profit_decay or should_exit_for_time:
                self.liquidate(long_symbol)
                self.liquidate(short_symbol)
                self._active_pair_positions.pop(pair_key, None)

        # Entry logic: open top candidates until max concurrent pairs is reached
        if len(self._active_pair_positions) >= self._max_pair_positions:
            return

        # Local test fallback: inject one basic pair candidate if analytics produced none
        if (
            self._enable_test_pair_fallback
            and self._local_data_mode
            and not self._pending_pair_candidates
            and not self._active_pair_positions
        ):
            tradable_symbols = [
                s for s in self._current_universe
                if isinstance(s, Symbol) and str(s.value).upper() not in {"SPY", "AGG"}
            ]
            if len(tradable_symbols) >= 2:
                a, b = tradable_symbols[0], tradable_symbols[1]
                self._pending_pair_candidates.append({
                    "pair_id": f"test-{a.value}-{b.value}",
                    "long_symbol": a,
                    "short_symbol": b,
                    "score": 0.01,
                    "correlation": 0.0,
                    "r_squared": 0.0
                })

        for candidate in self._pending_pair_candidates:
            if len(self._active_pair_positions) >= self._max_pair_positions:
                break

            pair_id = candidate["pair_id"]
            if pair_id in self._active_pair_positions:
                continue

            long_symbol = candidate["long_symbol"]
            short_symbol = candidate["short_symbol"]

            if (long_symbol not in data.bars) or (short_symbol not in data.bars):
                continue

            if not self.securities[long_symbol].is_tradable or not self.securities[short_symbol].is_tradable:
                continue

            if self.portfolio[long_symbol].invested or self.portfolio[short_symbol].invested:
                continue

            leg_weight = max(0.01, self._pair_position_size / 2.0)
            self.set_holdings(long_symbol, leg_weight)
            self.set_holdings(short_symbol, -leg_weight)

            self._active_pair_positions[pair_id] = {
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "entry_time": self.time
            }

            self.plot("Pair Execution", "Active Pair Positions", len(self._active_pair_positions))

    def on_end_of_algorithm(self):
        """Save logs to Object Store at the end."""
        self.object_store.save("algorithm_log.txt", self._log_content)
        # self.debug("Log file saved to Object Store: algorithm_log.txt")
