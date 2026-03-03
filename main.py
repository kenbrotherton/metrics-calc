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
        self._current_universe.add(self.spy.symbol)
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
        """Classify stocks into 6 Market Regimes based on Momentum and Volatility."""
        if metrics_df is None or len(metrics_df) < 2:
            # self.debug(f"Insufficient data for clustering (need 2+, have {len(metrics_df) if metrics_df is not None else 0})")
            return
        
        try:
            # Define 6 Fixed Regimes
            # 0: Calm Bull      (Mom > 2%, Vol < 1.5%)
            # 1: Volatile Bull  (Mom > 2%, Vol > 1.5%)
            # 2: Calm Bear      (Mom < -2%, Vol < 1.5%)
            # 3: Volatile Bear  (Mom < -2%, Vol > 1.5%)
            # 4: Calm Sideways  (Mom between -2% and 2%, Vol < 1.5%)
            # 5: Volatile Sideways (Mom between -2% and 2%, Vol > 1.5%)
            
            regime_names = [
                "Calm Bull", "Volatile Bull",
                "Calm Bear", "Volatile Bear",
                "Calm Sideways", "Volatile Sideways"
            ]
            groups = {name: [] for name in regime_names}
            
            for symbol, row in metrics_df.iterrows():
                mom = row['momentum']
                vol = row['volatility']
                
                # Determine Trend
                if mom > 0.02: trend = "Bull"
                elif mom < -0.02: trend = "Bear"
                else: trend = "Sideways"
                
                # Determine Volatility (1.5% daily threshold)
                if vol > 0.015: vol_type = "Volatile"
                else: vol_type = "Calm"
                
                # Assign Group ID
                if trend == "Bull" and vol_type == "Calm": grp = "Calm Bull"
                elif trend == "Bull" and vol_type == "Volatile": grp = "Volatile Bull"
                elif trend == "Bear" and vol_type == "Calm": grp = "Calm Bear"
                elif trend == "Bear" and vol_type == "Volatile": grp = "Volatile Bear"
                elif trend == "Sideways" and vol_type == "Calm": grp = "Calm Sideways"
                elif trend == "Sideways" and vol_type == "Volatile": grp = "Volatile Sideways"
                
                groups[grp].append(symbol)
            
            self._correlation_groups = groups
            
            # Update runtime statistic in the banner
            self.set_runtime_statistic("Clusters", 6)
            self.plot("Clustering Performance", "Cluster Count", 6)
            
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
            if l]
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
            labels.append(symbol_to_group.get(sym, "Unknown"))
        return labels

    def on_data(self, data):
        """Process incoming data (placeholder)."""
        pass

    def on_end_of_algorithm(self):
        """Save logs to Object Store at the end."""
        self.object_store.save("algorithm_log.txt", self._log_content)
        # self.debug("Log file saved to Object Store: algorithm_log.txt")
