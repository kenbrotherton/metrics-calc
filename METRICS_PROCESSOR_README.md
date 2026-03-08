# Metrics Processor

Standalone script to process historical price data and compute momentum/volatility metrics **independently** of the research notebook.

## Usage

### Basic usage (processes all symbols):
```bash
python metrics_processor.py
```

### For testing (limit to 100 symbols):
```bash
python metrics_processor.py --symbols-limit 100
```

### Custom analysis date and history:
```bash
python metrics_processor.py --analysis-date 2025-10-18 --history-days 1095
```

### Full options:
```bash
python metrics_processor.py --help
```

## Output Structure

Metrics are saved to `processed_metrics/` directory:

```
processed_metrics/
├── metrics_20251018_120530.csv    # Main metrics file (dated + timestamped)
├── metadata.json                   # Latest analysis metadata
└── index.csv                       # Index of all processed metric files
```

### Metrics CSV Columns
- `price` - Current price
- `price_change` - 3-year price change %
- `momentum` - 21-day momentum %
- `volatility` - Standard deviation of daily returns
- `volume` - Average trading volume
- `direction` - Price direction (1 = up, -1 = down)

## Workflow

1. **Run processor** (once or regularly):
   ```bash
   python metrics_processor.py
   ```
   This takes 2-5 minutes depending on symbol count.

2. **Load metrics in notebook**:
   The research notebook automatically loads the latest metrics file from `processed_metrics/`

3. **Run clustering/analysis**:
   Fast iteration on clustering, visualization, and analysis without re-processing data

## Benefits

- **Separation of concerns**: Data processing separate from analysis
- **Performance**: Compute metrics once, analyze many times
- **Caching**: Pre-processed data persisted to disk
- **Reproducibility**: All analysis dates tracked in index.csv
- **Parallel processing**: Could be run on schedule or separate machine
