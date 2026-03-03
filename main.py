#region imports
from AlgorithmImports import *

# Data manipulation and numerical computing
import pandas as pd
import numpy as np

# Scikit-learn for clustering and regression
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import io
#endregion


class PairedSwitching(QCAlgorithm):
    
    def initialize(self):
        """Initialize the algorithm with settings and universe."""
        # Initialize log buffer
        self._log_content = "Algorithm Initialized\n"
        
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
        self._group_labels = {}
        self._months = 0
        
        # Add SPY as baseline to universe
        self._current_universe.add(self.spy.Symbol)

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
        # Select stocks with fundamental data and price > 1
        filtered = [x for x in coarse if x.has_fundamental_data and x.price > 1]
        
        # Sort by Dollar Volume to get the most liquid stocks
        # We take top 1000 to ensure we have enough candidates for each sector in fine selection
        sorted_by_dollar_vol = sorted(filtered, key=lambda x: x.dollar_volume, reverse=True)
        selected = [x.symbol for x in sorted_by_dollar_vol[:1000]]
        
        # self.debug(f"Coarse filter: {len(selected)} symbols selected")
        return selected
    
    def _fine_filter(self, fine):
        """Select top 10 stocks by market cap in each sector."""
        # Group stocks by Morningstar Sector Code
        sector_dict = {}
        for f in fine:
            if f.market_cap == 0: continue
            sector = f.asset_classification.morningstar_sector_code
            if sector not in sector_dict:
                sector_dict[sector] = []
            sector_dict[sector].append(f)
        
        selected = []
        # Select top 10 by market cap for each sector
        for sector in sector_dict:
            sorted_sector = sorted(sector_dict[sector], key=lambda x: x.market_cap, reverse=True)
            selected.extend([x.symbol for x in sorted_sector[:10]])
        
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
    
    def _monthly_rebalance(self):
        """Monthly event handler to retrain clustering model."""
        self._months += 1
        
        # Skip during warmup
        if self.is_warming_up:
            return
        
        # Perform analysis after warmup
        if self._months >= 7:
            # self.debug(f"\n{'=' * 100}")
            # self.debug(f"MONTHLY RETRAINING - Month {self._months}")
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
                
                # Analyze correlations BETWEEN groups (The Grid)
                self._analyze_group_correlations()
                
                # Save visual and data outputs
                self._save_monthly_outputs(metrics_data)
    
    def _collect_stock_metrics(self):
        """Collect price, momentum, and other metrics for all stocks in universe."""
        symbols = list(self._current_universe)
        
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
            
            # Extract available symbols
            if hasattr(history.index, 'levels'):
                available_symbols = history.index.get_level_values(0).unique()
            else:
                available_symbols = []
            
            # self.debug(f"Retrieved data for {len(available_symbols)} symbols from history")
            
            metrics = {}
            
            for symbol in symbols:
                try:
                    # Check if symbol exists in history
                    if symbol not in available_symbols:
                        continue
                    
                    sym_history = history.loc[symbol]
                    
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
                    
                    metrics[symbol] = {
                        'price': current_price,
                        'price_change': price_change,
                        'momentum': momentum,
                        'volatility': volatility,
                        'volume': avg_volume,
                        'direction': direction
                    }
                    
                except (KeyError, Exception):
                    pass
            
            # self.debug(f"Collected metrics for {len(metrics)} stocks")
            
            if len(metrics) > 0:
                return pd.DataFrame(metrics).T
            else:
                return None
            
        except Exception as e:
            # self.debug(f"Error collecting metrics: {str(e)}")
            return None
    
    def _perform_clustering(self, metrics_df):
        """Perform K-Means clustering on stock metrics."""
        if metrics_df is None or len(metrics_df) < 2:
            # self.debug(f"Insufficient data for clustering (need 2+, have {len(metrics_df) if metrics_df is not None else 0})")
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
            
            # self.debug(f"Performing K-Means clustering with {n_clusters} clusters on {n_stocks} stocks...")
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(metrics_scaled)
            
            # Store results
            self._correlation_groups = {}
            for i in range(n_clusters):
                self._correlation_groups[i] = []
            
            for symbol, label in zip(metrics_df.index, labels):
                self._correlation_groups[label].append(symbol)
            
            # Update runtime statistic in the banner
            self.set_runtime_statistic("Clusters", n_clusters)
            
            # Calculate and plot Silhouette Score (measure of separation/cohesion)
            if len(set(labels)) > 1:
                sil_score = silhouette_score(metrics_scaled, labels)
                self.plot("Clustering Performance", "Silhouette Score", sil_score)
            self.plot("Clustering Performance", "Cluster Count", n_clusters)
            
            # self.debug(f"Clustering complete: {n_clusters} groups created")
            
        except Exception as e:
            # self.debug(f"Error in clustering: {str(e)}")
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
        
        for group_id, symbols in sorted(self._correlation_groups.items()):
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
            self.plot("Group Momentum", f"Group {group_id}", avg_momentum)
            
            # Plot group size
            self.plot("Group Sizes", f"Group {group_id}", len(symbols))
            
            # self.debug(f"GROUP METRICS:")
            # self.debug(f"  Average Price: ${avg_price:.2f}")
            # self.debug(f"  Average Price Change: {avg_price_change*100:.2f}%")
            # self.debug(f"  Average Momentum (21d): {avg_momentum*100:.2f}%")
            # self.debug(f"  Average Volatility: {avg_volatility*100:.2f}%")
            # self.debug(f"  Average Volume: {avg_volume:,.0f}")
            
            # Characterize the group
            # Assign label based on momentum
            if avg_momentum > 0.02:
                label = "Bullish"
            elif avg_momentum < -0.02:
                label = "Bearish"
            else:
                label = "Sideways"
            self._group_labels[group_id] = label
            
            if avg_volatility > 0.025:
                # self.debug(f"  Volatility Profile: HIGH")
                pass
            elif avg_volatility > 0.015:
                # self.debug(f"  Volatility Profile: MODERATE")
                pass
            else:
                # self.debug(f"  Volatility Profile: LOW")
                pass
            
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
                    
                    # Plot the R-squared value to visualize correlation strength over time
                    self.plot("Cluster Feature Importance (R2)", metric_col, r_squared)
                    
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
        
        # Add Baseline (SPY) to history request
        if self.spy.Symbol not in all_symbols:
            all_symbols.append(self.spy.Symbol)
        
        # Get data: 60 days for correlation + 21 days buffer for momentum calc
        history = self.history(all_symbols, 81, Resolution.DAILY)
        if history.empty:
            return

        # Prepare DataFrames
        # unstack(level=0) moves Symbol from index to columns
        closes = history['close'].unstack(level=0)
        volumes = history['volume'].unstack(level=0)
        
        # Define metrics to analyze
        metrics_to_analyze = {
            "Price Returns": closes.pct_change().dropna(),
            "Volume Changes": volumes.pct_change().dropna(),
            "Momentum Trends": closes.pct_change(21).dropna()
        }
        
        for metric_name, data_df in metrics_to_analyze.items():
            # 2. Calculate Group Indices (Average of members)
            group_series = {}
            
            # Add Baseline
            if self.spy.Symbol in data_df.columns:
                group_series["Baseline"] = data_df[self.spy.Symbol]
            
            for group_id, symbols in self._correlation_groups.items():
                # Filter for symbols present in this group and in the data
                group_syms = [s for s in symbols if s in data_df.columns]
                if not group_syms:
                    continue
                # Average the metric across stocks in the group for each day
                group_series[f"Group {group_id}"] = data_df[group_syms].mean(axis=1)

            if not group_series:
                continue

            group_df = pd.DataFrame(group_series)
            
            # 3. Calculate Correlation Matrix
            corr_matrix = group_df.corr()
            
            # 4. Print Matrix to Console (The "Grid") with Labels
            display_matrix = corr_matrix.copy()
            rename_map = {}
            for col in display_matrix.columns:
                if col.startswith("Group"):
                    try:
                        g_id = int(col.split(" ")[1])
                        lbl = self._group_labels.get(g_id, "")
                        rename_map[col] = f"G{g_id} ({lbl})"
                    except:
                        pass
            display_matrix.rename(columns=rename_map, index=rename_map, inplace=True)
            
            #self.debug(f"\nINTER-GROUP {metric_name.upper()} CORRELATION (Month {self._months}):\n" + display_matrix.to_string(float_format=lambda x: "{:.2f}".format(x)))
            
            # 5. Plotting (Only for Price Returns to avoid chart clutter)
            if metric_name == "Price Returns":
                # Mask diagonal to find true min/max
                mask = np.ones(corr_matrix.shape, dtype=bool)
                np.fill_diagonal(mask, 0)
                
                if len(corr_matrix) > 1:
                    # Calculate Inter-Group stats (excluding Baseline for "Inter-Group" chart consistency)
                    group_cols = [c for c in corr_matrix.columns if c.startswith("Group")]
                    if len(group_cols) > 1:
                        group_corr = corr_matrix.loc[group_cols, group_cols]
                        mask_g = np.ones(group_corr.shape, dtype=bool)
                        np.fill_diagonal(mask_g, 0)
                        
                        self.plot("Inter-Group Correlations", "Avg Correlation", group_corr.values[mask_g].mean())
                        self.plot("Inter-Group Correlations", "Min Correlation", group_corr.values[mask_g].min())
                
                # Plot individual group divergence and Baseline Correlation
                for col in corr_matrix.columns:
                    if not col.startswith("Group"): continue
                    
                    # Divergence vs Rest (other groups)
                    other_groups = [c for c in corr_matrix.columns if c.startswith("Group") and c != col]
                    if other_groups:
                        avg_val = corr_matrix.loc[col, other_groups].mean()
                        self.plot("Group Divergence", f"{col} vs Rest", avg_val)
                    
                    # Correlation vs Baseline
                    if "Baseline" in corr_matrix.columns:
                        base_corr = corr_matrix.loc[col, "Baseline"]
                        self.plot("Baseline Correlations", f"{col} vs SPY", base_corr)

    def _save_monthly_outputs(self, metrics_df):
        """Save CSV data and Matplotlib charts to ObjectStore."""
        try:
            # 1. Save Metrics to CSV
            # This allows you to download the raw data later for local analysis
            csv_key = f"metrics_month_{self._months}.csv"
            self.object_store.save(csv_key, metrics_df.to_csv())
            
            # 2. Generate and Save Cluster Plot
            # Create a scatter plot of Momentum vs Volatility colored by Cluster
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=metrics_df, x='volatility', y='momentum', 
                            hue=self._get_labels_for_plot(metrics_df), palette='viridis')
            plt.title(f'Clustering: Momentum vs Volatility (Month {self._months})')
            plt.xlabel('Volatility')
            plt.ylabel('Momentum')
            
            # Save plot to a bytes buffer, then to ObjectStore
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            self.object_store.save_bytes(f"cluster_plot_month_{self._months}.png", list(img_buf.getvalue()))
            plt.close()
            
        except Exception as e:
            # self.debug(f"Error saving outputs: {str(e)}")
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
            labels.append(symbol_to_group.get(sym, -1))
        return labels

    def on_data(self, data):
        """Process incoming data (placeholder)."""
        pass

    def on_end_of_algorithm(self):
        """Save logs to Object Store at the end."""
        self.object_store.save("algorithm_log.txt", self._log_content)
        # self.debug("Log file saved to Object Store: algorithm_log.txt")
