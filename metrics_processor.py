"""
Metrics Processor for Paired Switching Research

This script processes historical data and calculates rolling time-series metrics
that are stored per-date per-symbol, enabling queries like:
"What was AAPL's 20-day VWAP on 2025-01-01?"

Usage:
    python metrics_processor.py [--output-dir processed_metrics] [--symbols-limit 560]
"""

import pandas as pd
import numpy as np
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import json
import argparse
import sys
from metrics_calculator import MetricsEngine


class MetricsProcessor:
    def __init__(self, data_root="/Lean/Data", output_dir="processed_metrics"):
        self.data_root = Path(data_root)
        self.equity_daily = self.data_root / "equity" / "usa" / "daily"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize metrics engine with default metrics
        self.metrics_engine = MetricsEngine()
        self.metrics_engine.register_default_metrics()
        
        # Validate data directory
        if not self.equity_daily.exists():
            raise FileNotFoundError(f"Data directory not found: {self.equity_daily}")
        
        print(f"[OK] Data root: {self.data_root}")
        print(f"[OK] Output directory: {self.output_dir.absolute()}")
        print(f"[OK] Registered metrics: {', '.join(self.metrics_engine.list_metrics()[:5])}... ({len(self.metrics_engine.list_metrics())} total)")
    
    def get_available_symbols(self, limit=None):
        """Get list of available symbols from local data directory."""
        zip_files = sorted(self.equity_daily.glob("*.zip"))
        symbols = [f.stem.upper() for f in zip_files]
        
        if limit:
            symbols = symbols[:limit]
        
        print(f"[OK] Found {len(symbols)} available symbols")
        return symbols
    
    def load_symbol_data(self, symbol, start_date, end_date):
        """Load historical price data for a single symbol."""
        zip_path = self.equity_daily / f"{symbol.lower()}.zip"
        
        if not zip_path.exists():
            return None
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                csv_name = f"{symbol.lower()}.csv"
                file_list = zf.namelist()
                if csv_name not in file_list:
                    csv_name = file_list[0]
                
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)

                    # LEAN local daily files are commonly headerless:
                    # time,open,high,low,close,volume[,openinterest]
                    if 'close' not in [str(c).lower() for c in df.columns]:
                        f.seek(0)
                        raw_df = pd.read_csv(f, header=None)
                        if raw_df.shape[1] >= 6:
                            column_names = ['time', 'open', 'high', 'low', 'close', 'volume']
                            if raw_df.shape[1] >= 7:
                                column_names.append('openinterest')
                            raw_df.columns = column_names[:raw_df.shape[1]]
                            df = raw_df
                        else:
                            return None

                    # Parse date column
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    elif 'time' in df.columns:
                        df['date'] = pd.to_datetime(df['time'])
                    else:
                        df['date'] = pd.to_datetime(df.iloc[:, 0])
                    
                    df.set_index('date', inplace=True)
                    df.columns = [str(c).lower() for c in df.columns]
                    
                    # Filter by date range
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]
                    
                    return df
        except Exception as e:
            return None
    
    def collect_metrics(self, symbols, start_date, end_date, min_days=60):
        """
        Calculate rolling time-series metrics for all symbols.
        
        Returns a dictionary mapping symbol -> DataFrame with date-indexed metrics.
        """
        metrics_data = {}
        loaded_count = 0
        
        print(f"\nProcessing {len(symbols)} symbols with rolling metrics...")
        
        for i, symbol in enumerate(symbols):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(symbols)}")
            
            df = self.load_symbol_data(symbol, start_date, end_date)
            if df is None or len(df) < min_days * 0.5:
                continue
            
            if 'close' not in df.columns:
                continue
            
            try:
                # Calculate all rolling metrics using the metrics engine
                metrics_df = self.metrics_engine.calculate_all(df)
                
                if metrics_df is not None and len(metrics_df) > 0:
                    metrics_data[symbol] = metrics_df
                    loaded_count += 1
                    
            except Exception as e:
                print(f"[!] Error processing {symbol}: {e}")
                continue
        
        print(f"\n[OK] Collected rolling metrics for {loaded_count} stocks")
        
        return metrics_data if len(metrics_data) > 0 else None
    
    def save_metrics(self, metrics_data, end_date):
        """
        Save time-series metrics to per-symbol CSV files.
        
        Parameters:
        -----------
        metrics_data : dict
            Dictionary mapping symbol -> DataFrame with date-indexed metrics
        end_date : datetime
            Analysis end date for folder organization
        """
        if metrics_data is None or len(metrics_data) == 0:
            print("[X] No metrics to save")
            return None
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_date = end_date.strftime("%Y%m%d")
        run_dir = self.output_dir / f"run_{analysis_date}_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        print(f"\n[OK] Saving metrics to: {run_dir.absolute()}")
        
        # Save each symbol's time-series metrics
        saved_files = []
        for symbol, metrics_df in metrics_data.items():
            symbol_file = run_dir / f"{symbol.lower()}_metrics.csv"
            metrics_df.to_csv(symbol_file)
            saved_files.append(symbol_file.name)
        
        print(f"[OK] Saved {len(saved_files)} symbol metric files")
        
        # Save metadata
        metadata = {
            "created": datetime.now().isoformat(),
            "analysis_date": end_date.isoformat(),
            "symbol_count": len(metrics_data),
            "run_directory": run_dir.name,
            "metrics_calculated": self.metrics_engine.list_metrics(),
            "symbols": list(metrics_data.keys())
        }
        
        metadata_file = run_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Metadata saved")
        
        # Create consolidated index file
        self._create_index(run_dir, metrics_data)
        
        return run_dir
    
    def _create_index(self, run_dir, metrics_data):
        """Create an index file listing all symbols and date ranges."""
        index_data = []
        
        for symbol, metrics_df in metrics_data.items():
            index_data.append({
                'symbol': symbol,
                'start_date': metrics_df.index.min().isoformat(),
                'end_date': metrics_df.index.max().isoformat(),
                'row_count': len(metrics_df),
                'file': f"{symbol.lower()}_metrics.csv"
            })
        
        index_df = pd.DataFrame(index_data)
        index_file = run_dir / "index.csv"
        index_df.to_csv(index_file, index=False)
        print(f"[OK] Index created with {len(index_data)} symbols")
    
    def load_symbol_metrics(self, symbol, run_dir=None):
        """
        Load time-series metrics for a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., 'AAPL')
        run_dir : str or Path, optional
            Specific run directory to load from. If None, loads from latest run.
        
        Returns:
        --------
        pd.DataFrame
            Date-indexed DataFrame with all metrics
        """
        if run_dir is None:
            # Find latest run directory
            run_dirs = sorted(self.output_dir.glob("run_*"))
            if not run_dirs:
                print("[X] No run directories found")
                return None
            run_dir = run_dirs[-1]
        else:
            run_dir = Path(run_dir)
        
        metrics_file = run_dir / f"{symbol.lower()}_metrics.csv"
        
        if not metrics_file.exists():
            print(f"[X] Metrics file not found for {symbol}")
            return None
        
        df = pd.read_csv(metrics_file, index_col=0, parse_dates=True)
        return df
    
    def query_metric(self, symbol, date, metric_name, run_dir=None):
        """
        Query a specific metric value for a symbol on a specific date.
        
        Example:
        --------
        >>> processor.query_metric('AAPL', '2025-01-01', 'vwap_20d')
        145.23
        """
        metrics_df = self.load_symbol_metrics(symbol, run_dir)
        
        if metrics_df is None:
            return None
        
        try:
            date = pd.to_datetime(date)
            if date in metrics_df.index and metric_name in metrics_df.columns:
                return metrics_df.loc[date, metric_name]
        except Exception as e:
            print(f"[X] Error querying metric: {e}")
        
        return None


def main():
    """Main entry point for script execution."""
    parser = argparse.ArgumentParser(description="Process historical data and calculate rolling metrics")
    parser.add_argument("--output-dir", default="processed_metrics", help="Output directory for metrics")
    parser.add_argument("--symbols-limit", type=int, default=None, help="Limit number of symbols to process")
    parser.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = MetricsProcessor(output_dir=args.output_dir)
        
        # Get available symbols
        symbols = processor.get_available_symbols(limit=args.symbols_limit)
        
        # Set date range
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else (end_date - timedelta(days=365*3))
        
        print(f"\n[OK] Processing date range: {start_date.date()} to {end_date.date()}")
        
        # Collect rolling metrics
        metrics_data = processor.collect_metrics(symbols, start_date, end_date)
        
        if metrics_data is not None:
            # Save to files
            run_dir = processor.save_metrics(metrics_data, end_date)
            print(f"\n[OK] Metrics processing complete!")
            print(f"[OK] Results saved to: {run_dir.absolute()}")
        else:
            print("\n[X] Failed to collect metrics")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n[X] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
