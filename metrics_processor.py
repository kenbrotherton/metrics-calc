"""
Metrics Processor for Paired Switching Research

This script processes historical data independently and saves computed metrics
to a file for later use by the research notebook.

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


class MetricsProcessor:
    def __init__(self, data_root="/Lean/Data", output_dir="processed_metrics"):
        self.data_root = Path(data_root)
        self.equity_daily = self.data_root / "equity" / "usa" / "daily"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Validate data directory
        if not self.equity_daily.exists():
            raise FileNotFoundError(f"Data directory not found: {self.equity_daily}")
        
        print(f"[OK] Data root: {self.data_root}")
        print(f"[OK] Output directory: {self.output_dir.absolute()}")
    
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
            print(f"[!] Error loading {symbol}: {e}")
            return None
    
    def collect_metrics(self, symbols, start_date, end_date, min_days=252):
        """Collect metrics for all symbols."""
        metrics = {}
        loaded_count = 0
        
        print(f"\nProcessing {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(symbols)}")
            
            df = self.load_symbol_data(symbol, start_date, end_date)
            if df is None or len(df) < min_days * 0.5:
                continue
            
            if 'close' not in df.columns:
                continue
            
            try:
                close_prices = df['close'].values.astype(np.float64)
                
                # Volume
                if 'volume' in df.columns:
                    volumes = df['volume'].values.astype(np.float64)
                    avg_volume = np.mean(volumes)
                else:
                    avg_volume = 0
                
                # Basic metrics
                current_price = close_prices[-1]
                start_price = close_prices[0]
                price_change = (current_price - start_price) / start_price if start_price != 0 else 0
                
                # Momentum (21-day)
                momentum = 0
                if len(close_prices) > 21:
                    prev_price = close_prices[-21]
                    momentum = (current_price - prev_price) / prev_price if prev_price != 0 else 0
                
                # Volatility
                volatility = 0
                if len(close_prices) > 1:
                    returns = np.diff(close_prices) / close_prices[:-1]
                    returns = returns[np.isfinite(returns)]
                    volatility = np.std(returns) if len(returns) > 0 else 0
                
                # Direction
                direction = 1 if price_change > 0 else -1
                
                metrics[symbol] = {
                    'price': current_price,
                    'price_change': price_change,
                    'momentum': momentum,
                    'volatility': volatility,
                    'volume': avg_volume,
                    'direction': direction
                }
                
                loaded_count += 1
            
            except Exception as e:
                print(f"[!] Error processing {symbol}: {e}")
                continue
        
        print(f"\n[OK] Collected metrics for {loaded_count} stocks")
        
        if len(metrics) > 0:
            return pd.DataFrame(metrics).T
        else:
            return None
    
    def save_metrics(self, metrics_df, end_date):
        """Save metrics to CSV file with metadata."""
        if metrics_df is None or len(metrics_df) == 0:
            print("[X] No metrics to save")
            return None
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_date = end_date.strftime("%Y%m%d")
        filename = f"metrics_{analysis_date}_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        # Save metrics
        metrics_df.to_csv(filepath)
        print(f"[OK] Metrics saved to: {filepath.absolute()}")
        
        # Save metadata
        metadata = {
            "created": datetime.now().isoformat(),
            "analysis_date": end_date.isoformat(),
            "symbol_count": len(metrics_df),
            "metrics_file": filename,
            "columns": list(metrics_df.columns)
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Metadata saved to: {metadata_file.absolute()}")
        
        # Update index
        index_file = self.output_dir / "index.csv"
        index_entry = {
            "filename": filename,
            "created": datetime.now().isoformat(),
            "analysis_date": end_date.isoformat(),
            "symbol_count": len(metrics_df)
        }
        
        if index_file.exists():
            index_df = pd.read_csv(index_file)
            index_df = pd.concat([index_df, pd.DataFrame([index_entry])], ignore_index=True)
        else:
            index_df = pd.DataFrame([index_entry])
        
        index_df.to_csv(index_file, index=False)
        print(f"[OK] Index updated: {index_file.absolute()}")
        
        return filepath


def main():
    parser = argparse.ArgumentParser(description="Process metrics from historical data")
    parser.add_argument("--data-root", default="/Lean/Data", help="Root data directory")
    parser.add_argument("--output-dir", default="processed_metrics", help="Output directory for metrics")
    parser.add_argument("--symbols-limit", type=int, default=None, help="Limit number of symbols (for testing)")
    parser.add_argument("--analysis-date", default="2025-10-18", help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--history-days", type=int, default=1095, help="Days of history (default 3 years)")
    
    args = parser.parse_args()
    
    try:
        processor = MetricsProcessor(args.data_root, args.output_dir)
        
        # Parse dates
        end_date = datetime.strptime(args.analysis_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=args.history_days)
        
        print(f"\nAnalysis period: {start_date.date()} to {end_date.date()}\n")
        
        # Get symbols
        symbols = processor.get_available_symbols(limit=args.symbols_limit)
        
        # Collect metrics
        metrics_df = processor.collect_metrics(symbols, start_date, end_date)
        
        if metrics_df is not None:
            # Save to file
            processor.save_metrics(metrics_df, end_date)
            print("\n[OK] Metrics processing complete!")
        else:
            print("\n[X] Failed to collect metrics")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n[X] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
