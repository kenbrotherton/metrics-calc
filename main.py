#region imports
from AlgorithmImports import *

# Data manipulation and numerical computing
import pandas as pd
import numpy as np

# Scikit-learn for clustering and regression
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#endregion


class PairedSwitching(QCAlgorithm):
    
    def initialize(self):
        """Initialize the algorithm with settings and universe."""
        self.set_start_date(2025, 3, 15)
        self.set_cash(100000)
        
        # Primary assets
        self.spy = self.add_equity("SPY", Resolution.DAILY)
        self.agg = self.add_equity("AGG", Resolution.DAILY)
        
        # Warmup period (3 years of historical data)
        self.set_warmup(timedelta(days=756))
        
        # Data storage
        self._price_data = {}
        self._market_cap_data = {}
        self._current_universe = set()
        self._correlation_groups = {}
        self._months = 0
        
        # Add SPY as baseline to universe
        self._current_universe.add("SPY")
        
        # Use coarse/fine universe selection for S&P 500 stocks
        self.universe_settings.resolution = Resolution.DAILY
        self.add_universe(self._coarse_filter, self._fine_filter)
        
        # Schedule monthly retraining
        self.schedule.on(
            self.date_rules.month_start("SPY"),
            self.time_rules.after_market_open("SPY", 1),
            self._monthly_rebalance
        )
    
    def _coarse_filter(self, coarse):
        """Filter stocks by price and volume."""
        selected = []
        for cf in coarse:
            if cf.price > 1 and cf.volume > 100000:
                selected.append(cf.symbol)
                self._current_universe.add(cf.symbol)
        
        self.debug(f"Coarse filter: {len(selected)} symbols selected")
        return selected
    
    def _fine_filter(self, fine):
        """Fine filter for stocks with valid market cap."""
        # Filter for stocks with market cap data
        sp500_stocks = [f for f in fine if f.market_cap > 0]
        
        selected = [f.symbol for f in sp500_stocks]
        for symbol in selected:
            self._current_universe.add(symbol)
        
        self.debug(f"Fine filter: {len(selected)} stocks with market cap data")
        return selected
    
    def on_securities_changed(self, changes):
        """Track universe additions and removals."""
        if len(changes.added_securities) > 0:
            self.debug(f"Adding {len(changes.added_securities)} securities to universe")
            for security in changes.added_securities:
                self._current_universe.add(str(security.symbol))
        
        if len(changes.removed_securities) > 0:
            self.debug(f"Removing {len(changes.removed_securities)} securities from universe")
            for security in changes.removed_securities:
                self._current_universe.discard(str(security.symbol))
    
    def _monthly_rebalance(self):
        """Monthly event handler to retrain clustering model."""
        self._months += 1
        
        # Skip during warmup
        if self.is_warming_up:
            return
        
        # First month after warmup
        if self._months == 7:
            self.debug("=" * 100)
            self.debug("WARMUP COMPLETE - Starting cluster and regression analysis")
            self.debug("=" * 100)
        
        # Perform analysis after warmup
        if self._months >= 7:
            self.debug(f"\n{'=' * 100}")
            self.debug(f"MONTHLY RETRAINING - Month {self._months}")
            self.debug(f"{'=' * 100}")
            
            # Collect stock metrics
            metrics_data = self._collect_stock_metrics()
            
            if metrics_data is not None and len(metrics_data) > 0:
                # Perform clustering
                self._perform_clustering(metrics_data)
                
                # Generate group report
                self._generate_group_report(metrics_data)
                
                # Perform regression analysis
                self._perform_regression_analysis(metrics_data)
    
    def _collect_stock_metrics(self):
        """Collect price, momentum, and other metrics for all stocks in universe."""
        symbols = list(self._current_universe)
        
        if not symbols:
            self.debug("No symbols in universe")
            return None
        
        self.debug(f"Universe contains {len(symbols)} symbols")
        self.debug(f"Collecting metrics for {len(symbols)} stocks...")
        
        try:
            # Fetch historical data (1 year for momentum calculation)
            history = self.history(symbols, 252, Resolution.DAILY)
            
            if history.empty:
                self.debug("No historical data retrieved")
                return None
            
            # Extract available symbols
            if hasattr(history.index, 'levels'):
                available_symbols = history.index.get_level_values(0).unique()
            else:
                available_symbols = []
            
            self.debug(f"Retrieved data for {len(available_symbols)} symbols from history")
            
            metrics = {}
            
            for symbol in symbols:
                try:
                    sym_str = str(symbol)
                    
                    # Check if symbol exists in history
                    if sym_str not in [str(s) for s in available_symbols]:
                        continue
                    
                    sym_history = history.loc[sym_str]
                    
                    # Need sufficient data
                    if len(sym_history) < 252 * 0.5:
                        continue
                    
                    close_prices = sym_history['close'].values
                    volumes = sym_history['volume'].values
                    
                    # Calculate metrics
                    current_price = close_prices[-1]
                    price_change = (close_prices[-1] - close_prices[0]) / close_prices[0] if close_prices[0] != 0 else 0
                    momentum = (close_prices[-1] - close_prices[-21]) / close_prices[-21] if len(close_prices) > 20 and close_prices[-21] != 0 else 0
                    volatility = np.std(np.diff(close_prices) / close_prices[:-1])
                    avg_volume = np.mean(volumes)
                    direction = 1 if price_change > 0 else -1
                    
                    metrics[sym_str] = {
                        'price': current_price,
                        'price_change': price_change,
                        'momentum': momentum,
                        'volatility': volatility,
                        'volume': avg_volume,
                        'direction': direction
                    }
                    
                except (KeyError, Exception):
                    pass
            
            self.debug(f"Collected metrics for {len(metrics)} stocks")
            
            if len(metrics) > 0:
                return pd.DataFrame(metrics).T
            else:
                return None
            
        except Exception as e:
            self.debug(f"Error collecting metrics: {str(e)}")
            return None
    
    def _perform_clustering(self, metrics_df):
        """Perform K-Means clustering on stock metrics."""
        if metrics_df is None or len(metrics_df) < 2:
            self.debug(f"Insufficient data for clustering (need 2+, have {len(metrics_df) if metrics_df is not None else 0})")
            return
        
        try:
            # Normalize the metrics
            scaler = preprocessing.StandardScaler()
            metrics_scaled = scaler.fit_transform(metrics_df)
            
            # Determine optimal number of clusters
            n_stocks = len(metrics_df)
            n_clusters = max(2, min(15, max(2, n_stocks // 30)))  # At least 2 clusters
            
            # Prevent more clusters than stocks
            if n_clusters > n_stocks:
                n_clusters = max(2, n_stocks // 2)
            
            self.debug(f"Performing K-Means clustering with {n_clusters} clusters on {n_stocks} stocks...")
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(metrics_scaled)
            
            # Store results
            self._correlation_groups = {}
            for i in range(n_clusters):
                self._correlation_groups[i] = []
            
            for symbol, label in zip(metrics_df.index, labels):
                self._correlation_groups[label].append(symbol)
            
            self.debug(f"Clustering complete: {n_clusters} groups created")
            
        except Exception as e:
            self.debug(f"Error in clustering: {str(e)}")
    
    def _generate_group_report(self, metrics_df):
        """Generate and print group report."""
        if not self._correlation_groups:
            self.debug("No groups to report")
            return
        
        self.debug("\n" + "=" * 100)
        self.debug("GROUP REPORT")
        self.debug("=" * 100)
        
        for group_id, symbols in sorted(self._correlation_groups.items()):
            if not symbols:
                continue
            
            self.debug(f"\n{'─' * 100}")
            self.debug(f"GROUP {group_id} - {len(symbols)} MEMBERS")
            self.debug(f"{'─' * 100}")
            
            # Group statistics
            group_metrics = metrics_df.loc[symbols]
            
            avg_price = group_metrics['price'].mean()
            avg_price_change = group_metrics['price_change'].mean()
            avg_momentum = group_metrics['momentum'].mean()
            avg_volatility = group_metrics['volatility'].mean()
            avg_volume = group_metrics['volume'].mean()
            
            self.debug(f"GROUP METRICS:")
            self.debug(f"  Average Price: ${avg_price:.2f}")
            self.debug(f"  Average Price Change: {avg_price_change*100:.2f}%")
            self.debug(f"  Average Momentum (21d): {avg_momentum*100:.2f}%")
            self.debug(f"  Average Volatility: {avg_volatility*100:.2f}%")
            self.debug(f"  Average Volume: {avg_volume:,.0f}")
            
            # Characterize the group
            if avg_momentum > 0.05:
                self.debug(f"  Characteristic: STRONG UPTREND")
            elif avg_momentum < -0.05:
                self.debug(f"  Characteristic: STRONG DOWNTREND")
            else:
                self.debug(f"  Characteristic: NEUTRAL/SIDEWAYS")
            
            if avg_volatility > 0.025:
                self.debug(f"  Volatility Profile: HIGH")
            elif avg_volatility > 0.015:
                self.debug(f"  Volatility Profile: MODERATE")
            else:
                self.debug(f"  Volatility Profile: LOW")
            
            # Member list
            self.debug(f"\nMEMBERS ({len(symbols)} stocks):")
            sorted_symbols = sorted(symbols)
            
            for i in range(0, len(sorted_symbols), 5):
                row = sorted_symbols[i:i+5]
                self.debug(f"  {', '.join(row)}")
        
        self.debug("\n" + "=" * 100)
    
    def _perform_regression_analysis(self, metrics_df):
        """Perform regression analysis between groups and metrics."""
        if not self._correlation_groups:
            self.debug("No groups for regression analysis")
            return
        
        try:
            self.debug(f"\nPerforming regression analysis between groups...")
            
            # Create group assignment series
            group_assignments = {}
            for group_id, symbols in self._correlation_groups.items():
                for symbol in symbols:
                    group_assignments[symbol] = group_id
            
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
                    
                except Exception as e:
                    results[metric_col] = 0.0
            
            # Generate regression report
            self._generate_regression_report(results)
            
        except Exception as e:
            self.debug(f"Error in regression analysis: {str(e)}")
    
    def _generate_regression_report(self, results):
        """Generate and print regression report."""
        self.debug("\n" + "=" * 100)
        self.debug("REGRESSION ANALYSIS REPORT - Group Metrics Correlation")
        self.debug("=" * 100)
        
        self.debug(f"\nR-squared values indicate how well each metric correlates with group membership:")
        self.debug(f"(Higher = stronger correlation between metric and group)\n")
        
        # Sort by R-squared descending
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        for metric, r_squared in sorted_results:
            strength = "STRONG" if r_squared > 0.5 else "MODERATE" if r_squared > 0.25 else "WEAK"
            self.debug(f"  {metric:<20} R² = {r_squared:.4f}  [{strength}]")
        
        # Summary
        self.debug("\nINTERPRETATION:")
        high_correlation = [m for m, r2 in results.items() if r2 > 0.5]
        
        if high_correlation:
            self.debug(f"  Strong group distinction factors: {', '.join(high_correlation)}")
        else:
            self.debug(f"  Groups show weak metric-based distinction - may indicate alternative grouping factors")
        
        self.debug("\n" + "=" * 100)
    
    def on_data(self, data):
        """Process incoming data (placeholder)."""
        pass
