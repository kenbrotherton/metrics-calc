# Paired Switching - Local Data Research

This project contains the **Paired Switching** algorithm for QuantConnect, with support for **local data** research and backtesting.

## Project Structure

```
PairedSwitching/
├── main.py                    # Main algorithm file (for QuantConnect/LEAN)
├── research.ipynb             # Original research notebook (requires QC API)
├── research_local.ipynb       # NEW: Local data research notebook
├── config.json                # Cloud configuration
├── local_config.json          # NEW: Local backtesting configuration
└── README_LOCAL.md            # This file
```

## Local Data Setup

Your local data is located at: **`C:\Users\kenbr\QC\data`**

The data structure follows QuantConnect's format:
```
data/
├── equity/
│   └── usa/
│       ├── daily/          # Daily price data (ZIP files)
│       ├── minute/         # Minute price data
│       ├── fundamental/    # Fundamental data
│       └── map_files/      # Symbol mapping files
├── symbol-properties/      # Symbol metadata
└── market-hours/          # Market hours database
```

## Using the Local Research Notebook

### Option 1: Jupyter Notebook
1. Open `research_local.ipynb` in Jupyter or VS Code
2. Run cells sequentially from top to bottom
3. The notebook will:
   - Load price data from local ZIP files
   - Calculate momentum and volatility metrics
   - Classify stocks into 6 market regimes
   - Generate visualizations

### Option 2: VS Code Jupyter Extension
1. Open `research_local.ipynb` in VS Code
2. Select Python kernel
3. Run cells using the play button or `Shift+Enter`

## Notebook Features

The local research notebook includes:

1. **Data Loading** (Cells 1-4)
   - Loads historical data from local ZIP files
   - Supports custom date ranges
   - Automatic symbol property loading

2. **Metric Collection** (Cells 5-7)
   - Price change calculation
   - 21-day momentum
   - Daily volatility (standard deviation of returns)
   - Volume analysis

3. **Market Regime Clustering** (Cells 5-7)
   - 6 predefined regimes based on momentum and volatility:
     - **Calm Bull**: Momentum > 2%, Volatility < 1.5%
     - **Volatile Bull**: Momentum > 2%, Volatility > 1.5%
     - **Calm Bear**: Momentum < -2%, Volatility < 1.5%
     - **Volatile Bear**: Momentum < -2%, Volatility > 1.5%
     - **Calm Sideways**: -2% < Momentum < 2%, Volatility < 1.5%
     - **Volatile Sideways**: -2% < Momentum < 2%, Volatility > 1.5%

4. **Visualizations** (Cells 8-9)
   - Scatter plot: Momentum vs Volatility with regime boundaries
   - Distribution histograms
   - Regime member counts

5. **Analysis** (Cells 10-11)
   - Detailed regime statistics
   - Top performers by regime
   - Member listings

6. **Export** (Cell 12)
   - Save metrics to CSV
   - Save regime assignments

## Running Local Backtests with LEAN

To run the `main.py` algorithm with local data:

1. Ensure you have QuantConnect LEAN installed
2. Update the configuration to point to your local data:
   ```json
   {
       "data-folder": "C:\\Users\\kenbr\\QC\\data",
       "algorithm-location": "main.py"
   }
   ```
3. Run LEAN from the command line:
   ```bash
   lean backtest "PairedSwitching" --data-folder "C:\Users\kenbr\QC\data"
   ```

## Configuration Notes

### Data Path
- The notebook is configured to use: `C:\Users\kenbr\QC\data`
- If your data is elsewhere, update the `DATA_ROOT` variable in Cell 1

### Universe Selection
- The notebook uses a curated list of top 100 liquid stocks
- You can modify the `TOP_SYMBOLS` list in Cell 3
- Or use automatic symbol discovery from your data folder

### Date Range
- Default: 3 years of history ending Oct 18, 2025
- Modify `END_DATE` and `START_DATE` in Cell 3

## Troubleshooting

### "No data loaded" error
- Verify data path: `C:\Users\kenbr\QC\data\equity\usa\daily`
- Check that ZIP files exist for your symbols
- Ensure ZIP files contain CSV data in QuantConnect format

### "Insufficient data for clustering" error
- Some symbols may not have enough history
- The notebook requires at least 126 days (50% of 252 trading days)
- Try expanding the date range or using different symbols

### Import errors
- Ensure pandas, numpy, matplotlib, and seaborn are installed:
  ```bash
  pip install pandas numpy matplotlib seaborn
  ```

## Differences from Cloud Version

| Feature | Cloud (research.ipynb) | Local (research_local.ipynb) |
|---------|----------------------|---------------------------|
| Data Source | QuantConnect API | Local ZIP files |
| Universe | QC FundamentalUniverse | Predefined symbol list |
| QuantBook | Required | Not required |
| Fine/Coarse Filter | API-based | Manual selection |
| Speed | Slower (API calls) | Faster (local I/O) |

## Next Steps

1. ✅ Run the local research notebook
2. Analyze regime classifications
3. Backtest strategy in LEAN with local data
4. Optimize parameters based on research insights
5. Deploy to QuantConnect cloud for live trading

## Support

For issues with:
- **QuantConnect LEAN**: https://github.com/QuantConnect/Lean
- **Data format**: https://www.quantconnect.com/docs/v2/writing-algorithms/importing-data/bulk-downloads
- **Algorithm logic**: Review `main.py` and inline comments
