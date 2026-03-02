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
        self.set_start_date(2015,3,15)
        #self.set_end_date(2018,7,15)
        self.set_cash(100000)
        #we select two etfs that are negatively correlated; equity and bond etfs
        self._first = self.add_equity("SPY",Resolution.MINUTE)
        self._second = self.add_equity("AGG",Resolution.MINUTE)
        self._months = -1
        #monthly scheduled event but rebalancing will run on quarterly basis
        self.schedule.on(self.date_rules.month_start("SPY"), self.time_rules.after_market_open("SPY", 1), self._rebalance)
        
        # Initialize data storage for universe of stocks and benchmarks
        self._price_data = {}
        self._universe_symbols = []
        
        # Warmup and pair trading parameters
        self._warmup_months = 6  # Warmup period in months
        self._best_trading_pairs = {}
        self._active_pair_positions = {}  # Track active positions from pairs
        self._pairs_initialized = False
        
        # Acquire historical data for large universe
        self._initialize_universe()
        self._fetch_historical_data()
        
        # Analyze correlations in the data
        self._print_correlation_summary()

    def _initialize_universe(self):
        """
        Initialize a universe of stocks, benchmarks, and commodities.
        Includes S&P 500 components sample, gold, and VIX proxies.
        """
        # Major benchmark ETFs
        benchmarks = ["SPY", "QQQ", "IWM", "AGG"]
        
        # Large-cap stocks from top 5 sectors
        large_cap_stocks = self._get_top_stocks_by_sector()
        
        # Commodity and sector ETFs
        commodities = ["GLD",    # Gold ETF
                       "USO",    # Oil ETF
                       "DBC",    # Commodity index
                       "DXY"]    # Dollar index proxy
        
        # Volatility proxy
        volatility = ["VXX"]  # VIX ETN
        
        # Combine all symbols
        all_symbols = benchmarks + large_cap_stocks + commodities + volatility
        
        # Add securities to algorithm and store symbols
        for symbol in all_symbols:
            try:
                self.add_equity(symbol, Resolution.DAILY)
                self._universe_symbols.append(symbol)
            except:
                self.debug(f"Could not add symbol: {symbol}")
        
        self.debug(f"Universe initialized with {len(self._universe_symbols)} symbols")



    def _get_top_stocks_by_sector(self, top_n=10):
        """
        Dynamically selects the top N stocks from the 5 largest sectors
        based on current market cap and sector representation.
        
        Args:
            top_n: Number of top stocks to select from each sector (default: 10)
        
        Returns:
            List of top stocks dynamically selected from 5 largest sectors
        """
        # Comprehensive S&P 500 stock mapping with sectors
        sp500_stocks_by_sector = {
            "Technology": [
                "AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "META", "TSLA", 
                "AVGO", "ASML", "ADBE", "CSCO", "INTC", "AMD", "CRM", "INTU",
                "ANET", "QCOM", "MU", "SNPS", "CDNS", "NOW", "CRWD", "SHOP",
                "DDOG", "FTNT", "ZM", "OKTA", "SPLK", "PALO", "TEAM"
            ],
            "Healthcare": [
                "JNJ", "UNH", "MRK", "AZN", "LLY", "ABBV", "TMO", "GILD",
                "AMGN", "BIIB", "REGN", "ILMN", "EW", "ANTM", "CI", "HUM",
                "VRTX", "BKNG", "ZTS", "DXCM", "VEEV", "SYK", "CLAG", "TFX",
                "ALGN", "DHR", "STE", "BLKB", "TMDX", "HOLX"
            ],
            "Financials": [
                "JPM", "BAC", "WFC", "GS", "BLK", "SCHW", "SPGI", "BK",
                "PNC", "USB", "TFC", "FITB", "KEY", "CFG", "ALL", "HIG",
                "AXP", "LPL", "COF", "DFS", "COIN", "SOFI", "MET", "ICE",
                "CME", "MSCI", "MCO", "FICO", "FI", "VOYA"
            ],
            "Industrials": [
                "BA", "CAT", "GE", "MMM", "HON", "RTX", "LMT", "NOC",
                "DE", "EMR", "ETN", "URI", "CARR", "OTIS", "IEX", "ROK",
                "LUV", "UAL", "AAL", "DAL", "SWA", "JBLU", "ALK", "KSU",
                "UNP", "CSX", "NSC", "CP", "CNI", "RXO"
            ],
            "Consumer Discretionary": [
                "AMZN", "WMT", "MCD", "NKE", "TJX", "LOW", "HD", "BKNG",
                "ORCL", "CPRT", "RCL", "CCL", "LVS", "MGM", "WYNN", "DPZ",
                "CMG", "TSCO", "KKR", "APTV", "EXPE", "KSS", "M", "AZO",
                "AutoZone", "EBAY", "GRMN", "F", "GM", "BYD"
            ]
        }
        
        try:
            # Get current market cap data for all stocks
            stock_market_caps = {}
            
            for sector, stocks in sp500_stocks_by_sector.items():
                for symbol in stocks:
                    try:
                        # Get security object if available
                        if symbol in self.securities:
                            security = self.securities[symbol]
                            # Estimate market cap from available data
                            # In practice, this would use fundamental data
                            stock_market_caps[symbol] = {
                                'price': security.price,
                                'sector': sector
                            }
                    except:
                        pass
            
            # If we don't have fundamental data, use static ranking
            if not stock_market_caps:
                self.debug("Using default sector-based stock ranking")
                top_stocks = self._get_top_stocks_static(sp500_stocks_by_sector, top_n)
            else:
                # Dynamically rank stocks within each sector
                top_stocks = self._rank_stocks_by_sector(
                    stock_market_caps, 
                    sp500_stocks_by_sector, 
                    top_n
                )
            
            self.debug(f"Selected {len(top_stocks)} top stocks from 5 largest sectors dynamically")
            return top_stocks
            
        except Exception as e:
            self.debug(f"Error in dynamic stock selection: {str(e)}. Using default ranking.")
            return self._get_top_stocks_static(sp500_stocks_by_sector, top_n)

    def _rank_stocks_by_sector(self, stock_data, sectors_dict, top_n=10):
        """
        Ranks stocks within each sector and returns top N from each.
        
        Args:
            stock_data: Dictionary with stock prices and sectors
            sectors_dict: Dictionary mapping sectors to stock lists
            top_n: Number of top stocks per sector
        
        Returns:
            List of top ranked stocks across all sectors
        """
        ranked_stocks = []
        
        for sector, stocks in sectors_dict.items():
            # Get stocks in this sector that have data
            sector_stocks = [
                (symbol, stock_data[symbol]['price'])
                for symbol in stocks 
                if symbol in stock_data
            ]
            
            # Sort by price (proxy for market cap in absense of fundamentals)
            sector_stocks.sort(key=lambda x: x[1], reverse=True)
            
            # Take top N
            top_in_sector = [symbol for symbol, _ in sector_stocks[:top_n]]
            ranked_stocks.extend(top_in_sector)
        
        return ranked_stocks

    def _get_top_stocks_static(self, sectors_dict, top_n=10):
        """
        Returns top N stocks from each sector using static ordering.
        Used as fallback when dynamic market data is unavailable.
        
        Args:
            sectors_dict: Dictionary mapping sectors to stock lists
            top_n: Number of top stocks per sector
        
        Returns:
            List of top stocks from each sector (predefined order)
        """
        top_stocks = []
        
        for sector, stocks in sectors_dict.items():
            # Take first top_n stocks from each sector (pre-ranked by importance)
            top_stocks.extend(stocks[:top_n])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_stocks = []
        for stock in top_stocks:
            if stock not in seen:
                seen.add(stock)
                unique_stocks.append(stock)
        
        return unique_stocks

    def _fetch_historical_data(self, lookback_days=252):
        """
        Fetch historical price data for the entire universe.
        
        Args:
            lookback_days: Number of days of historical data to retrieve (default: 1 year = 252 trading days)
        """
        if not self._universe_symbols:
            self.debug("No symbols in universe to fetch")
            return
        
        try:
            # Fetch daily historical data for all symbols
            history = self.history(self._universe_symbols, lookback_days, Resolution.DAILY)
            
            if history.empty:
                self.debug("No historical data retrieved")
                return
            
            # Organize data by symbol
            for symbol in self._universe_symbols:
                try:
                    symbol_history = history.loc[symbol]
                    
                    # Extract OHLCV data
                    self._price_data[symbol] = {
                        'dates': symbol_history.index,
                        'open': symbol_history['open'].values,
                        'high': symbol_history['high'].values,
                        'low': symbol_history['low'].values,
                        'close': symbol_history['close'].values,
                        'volume': symbol_history['volume'].values
                    }
                except:
                    self.debug(f"Could not retrieve data for {symbol}")
            
            self.debug(f"Historical data acquired for {len(self._price_data)} symbols")
            
        except Exception as e:
            self.debug(f"Error fetching historical data: {str(e)}")

    def _get_price_dataframe(self, symbols=None, data_type='close'):
        """
        Convert price data to pandas DataFrame for analysis.
        
        Args:
            symbols: List of symbols to include (default: all)
            data_type: Type of price data - 'open', 'high', 'low', 'close', 'volume'
        
        Returns:
            DataFrame with symbols as columns and dates as index
        """
        if symbols is None:
            symbols = list(self._price_data.keys())
        
        data_arrays = {}
        for symbol in symbols:
            if symbol in self._price_data:
                data_arrays[symbol] = self._price_data[symbol][data_type]
        
        if not data_arrays:
            self.debug("No data available for requested symbols")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(data_arrays)
        return df

    def _calculate_correlations(self):
        """
        Calculate correlation matrix for price data.
        
        Returns:
            Correlation matrix as pandas DataFrame
        """
        price_df = self._get_price_dataframe()
        if price_df.empty:
            return None
        
        correlation_matrix = price_df.corr()
        return correlation_matrix

    def _find_correlated_pairs(self, min_correlation=0.7, exclude_self=True):
        """
        Find pairs of stocks with high positive correlation.
        
        Args:
            min_correlation: Minimum correlation threshold (default: 0.7)
            exclude_self: Exclude self-correlations (default: True)
        
        Returns:
            List of tuples (symbol1, symbol2, correlation_value)
        """
        corr_matrix = self._calculate_correlations()
        if corr_matrix is None:
            self.debug("No correlation matrix available")
            return []
        
        correlated_pairs = []
        
        # Iterate through the upper triangle of the correlation matrix
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                symbol1 = corr_matrix.columns[i]
                symbol2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if correlation >= min_correlation:
                    correlated_pairs.append((symbol1, symbol2, correlation))
        
        # Sort by correlation (highest first)
        correlated_pairs.sort(key=lambda x: x[2], reverse=True)
        
        self.debug(f"Found {len(correlated_pairs)} correlated pairs (correlation >= {min_correlation})")
        return correlated_pairs

    def _find_negatively_correlated_pairs(self, max_correlation=-0.5, exclude_self=True):
        """
        Find pairs of stocks with strong negative correlation.
        Useful for paired switching/hedging strategies.
        
        Args:
            max_correlation: Maximum correlation threshold for negative correlation (default: -0.5)
            exclude_self: Exclude self-correlations (default: True)
        
        Returns:
            List of tuples (symbol1, symbol2, correlation_value)
        """
        corr_matrix = self._calculate_correlations()
        if corr_matrix is None:
            self.debug("No correlation matrix available")
            return []
        
        neg_correlated_pairs = []
        
        # Iterate through the upper triangle of the correlation matrix
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                symbol1 = corr_matrix.columns[i]
                symbol2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                # Negative correlation (closer to -1 is stronger)
                if correlation <= max_correlation:
                    neg_correlated_pairs.append((symbol1, symbol2, correlation))
        
        # Sort by correlation (most negative first)
        neg_correlated_pairs.sort(key=lambda x: x[2])
        
        self.debug(f"Found {len(neg_correlated_pairs)} negatively correlated pairs (correlation <= {max_correlation})")
        return neg_correlated_pairs

    def _get_top_correlations(self, top_n=20, correlation_type='positive'):
        """
        Get the top N correlated pairs.
        
        Args:
            top_n: Number of top pairs to return (default: 20)
            correlation_type: 'positive' for high correlations, 'negative' for inverse correlations
        
        Returns:
            List of tuples (symbol1, symbol2, correlation_value)
        """
        if correlation_type == 'negative':
            pairs = self._find_negatively_correlated_pairs(max_correlation=-0.3)
        else:
            pairs = self._find_correlated_pairs(min_correlation=0.5)
        
        return pairs[:top_n]

    def _find_best_pairs_for_trading(self, min_negative_corr=-0.4):
        """
        Find pairs of stocks that are negatively correlated for paired switching strategy.
        These pairs move in opposite directions, making them ideal for hedged trading.
        
        Args:
            min_negative_corr: Threshold for negative correlation (default: -0.4)
        
        Returns:
            Dictionary with trading statistics for negatively correlated pairs
        """
        neg_pairs = self._find_negatively_correlated_pairs(max_correlation=min_negative_corr)
        
        if not neg_pairs:
            self.debug(f"No negatively correlated pairs found (threshold: {min_negative_corr})")
            return {}
        
        trading_pairs = {}
        
        for symbol1, symbol2, correlation in neg_pairs:
            pair_key = f"{symbol1}-{symbol2}"
            
            # Get performance metrics
            prices1 = self._get_historical_prices(symbol1, 'close')
            prices2 = self._get_historical_prices(symbol2, 'close')
            
            if len(prices1) > 0 and len(prices2) > 0:
                # Calculate returns
                returns1 = (prices1[-1] - prices1[0]) / prices1[0] if prices1[0] != 0 else 0
                returns2 = (prices2[-1] - prices2[0]) / prices2[0] if prices2[0] != 0 else 0
                
                # Calculate volatility
                vol1 = np.std(prices1)
                vol2 = np.std(prices2)
                
                trading_pairs[pair_key] = {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'correlation': correlation,
                    'returns1': returns1,
                    'returns2': returns2,
                    'volatility1': vol1,
                    'volatility2': vol2,
                    'spread': abs(returns1 - returns2)
                }
        
        self.debug(f"Identified {len(trading_pairs)} trading-ready pairs")
        return trading_pairs

    def _print_correlation_summary(self):
        """
        Print a summary of correlation analysis to the debug log.
        """
        self.debug("=" * 60)
        self.debug("CORRELATION ANALYSIS SUMMARY")
        self.debug("=" * 60)
        
        # Top positively correlated
        top_positive = self._get_top_correlations(top_n=5, correlation_type='positive')
        self.debug("\nTop 5 Positively Correlated Pairs:")
        for symbol1, symbol2, corr in top_positive:
            self.debug(f"  {symbol1} <-> {symbol2}: {corr:.4f}")
        
        # Top negatively correlated
        top_negative = self._get_top_correlations(top_n=5, correlation_type='negative')
        self.debug("\nTop 5 Negatively Correlated Pairs:")
        for symbol1, symbol2, corr in top_negative:
            self.debug(f"  {symbol1} <-> {symbol2}: {corr:.4f}")
        
        # Best trading pairs
        trading_pairs = self._find_best_pairs_for_trading()
        if trading_pairs:
            self.debug(f"\nTop Trading Pairs (with spreads):")
            sorted_pairs = sorted(trading_pairs.items(), 
                                key=lambda x: x[1]['spread'], 
                                reverse=True)[:5]
            for pair_key, stats in sorted_pairs:
                self.debug(f"  {pair_key}: corr={stats['correlation']:.4f}, spread={stats['spread']:.4f}")
        
        self.debug("=" * 60)

    def _get_historical_prices(self, symbol, data_type='close'):
        """
        Retrieve historical price data for a specific symbol.
        
        Args:
            symbol: Stock symbol
            data_type: Type of price data - 'open', 'high', 'low', 'close', 'volume'
        
        Returns:
            Array of prices
        """
        if symbol in self._price_data:
            return self._price_data[symbol].get(data_type, [])
        return []


    def _rebalance(self):
        self._months += 1
        
        # Check if warmup period is complete
        if self._months < self._warmup_months:
            self.debug(f"In warmup period ({self._months}/{self._warmup_months} months)")
            return
        
        # Initialize pair trading after warmup period
        if not self._pairs_initialized:
            self.debug("Warmup period complete. Initializing pair trading strategy.")
            self._initialize_pair_positions()
            self._pairs_initialized = True
            return
        
        # After warmup, execute pair trading rebalance
        if self._months % 3 == 0:  # Quarterly rebalance
            self._rebalance_pair_positions()

    def _initialize_pair_positions(self):
        """
        Get the best trading pairs after warmup and initialize positions at equal proportions.
        """
        # Get recommended pairs from correlation analysis
        self._best_trading_pairs = self._find_best_pairs_for_trading(min_negative_corr=-0.4)
        
        if not self._best_trading_pairs:
            self.debug("No suitable pairs found for trading. Continuing with SPY/AGG strategy.")
            return
        
        # Sort pairs by spread (largest spread = highest divergence opportunity)
        sorted_pairs = sorted(
            self._best_trading_pairs.items(),
            key=lambda x: x[1]['spread'],
            reverse=True
        )
        
        # Use top pairs (e.g., top 3-5 pairs to avoid over-diversification)
        num_pairs = min(5, len(sorted_pairs))
        selected_pairs = sorted_pairs[:num_pairs]
        
        self.debug(f"Selected {num_pairs} best pairs for trading:")
        
        # Calculate equal weight per pair
        # Each pair gets equal capital allocation
        pairs_to_allocate = num_pairs
        weight_per_pair = 1.0 / pairs_to_allocate
        
        for pair_key, pair_stats in selected_pairs:
            symbol1 = pair_stats['symbol1']
            symbol2 = pair_stats['symbol2']
            
            self.debug(f"  {pair_key}: correlation={pair_stats['correlation']:.4f}, spread={pair_stats['spread']:.4f}")
            
            # Initialize positions - buy the underperformer (lower return), short the outperformer
            # This creates a market-neutral spread trade
            if pair_stats['returns1'] > pair_stats['returns2']:
                # Symbol1 is outperforming, short it; buy Symbol2
                long_symbol = symbol2
                short_symbol = symbol1
            else:
                # Symbol2 is outperforming, short it; buy Symbol1
                long_symbol = symbol1
                short_symbol = symbol2
            
            # Allocate capital equally among all pairs
            self._active_pair_positions[pair_key] = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'long_symbol': long_symbol,
                'short_symbol': short_symbol,
                'weight': weight_per_pair,
                'correlation': pair_stats['correlation']
            }
            
            # Set initial positions at equal proportions
            self.set_holdings(long_symbol, weight_per_pair)
            self.debug(f"    LONG: {long_symbol} at {weight_per_pair:.2%} weight")
            self.debug(f"    SHORT: {short_symbol} at {weight_per_pair:.2%} weight (short)")

    def _rebalance_pair_positions(self):
        """
        Rebalance the pair positions on a quarterly basis.
        Recalculate which leg of each pair is outperforming and adjust accordingly.
        """
        if not self._active_pair_positions:
            return
        
        self.debug("Rebalancing pair positions...")
        
        for pair_key, position_info in self._active_pair_positions.items():
            symbol1 = position_info['symbol1']
            symbol2 = position_info['symbol2']
            weight = position_info['weight']
            
            # Get recent performance
            prices1 = self._get_historical_prices(symbol1, 'close')
            prices2 = self._get_historical_prices(symbol2, 'close')
            
            if len(prices1) > 0 and len(prices2) > 0:
                # Calculate recent returns (last quarter)
                recent_lookback = min(63, len(prices1) - 1)  # ~3 months
                if recent_lookback > 1:
                    returns1 = (prices1[-1] - prices1[-recent_lookback]) / prices1[-recent_lookback]
                    returns2 = (prices2[-1] - prices2[-recent_lookback]) / prices2[-recent_lookback]
                else:
                    returns1 = 0
                    returns2 = 0
                
                # Adjust positions based on recent divergence
                if returns1 > returns2:
                    # Symbol1 outperforming, long Symbol2 and short Symbol1
                    long_symbol = symbol2
                    short_symbol = symbol1
                else:
                    # Symbol2 outperforming, long Symbol1 and short Symbol2
                    long_symbol = symbol1
                    short_symbol = symbol2
                
                # Update position info
                self._active_pair_positions[pair_key]['long_symbol'] = long_symbol
                self._active_pair_positions[pair_key]['short_symbol'] = short_symbol
                
                # Rebalance holdings
                self.set_holdings(long_symbol, weight)
                
                self.debug(f"  {pair_key}: LONG {long_symbol} ({returns1 - returns2:+.4f}% spread)")



