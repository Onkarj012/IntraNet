# 🌅 Morning Picks System (`morning_picks.py`)

The **Morning Picks** script is a pre-market recommendation engine powered by the **IntradayNet** architecture. It is designed to be run just before the market opens (around 8:30 AM - 9:15 AM).

It downloads the latest live minute-level stock data and macroeconomic indicators (VIX, Crude, Gold, USD/INR, NIFTY 50), computes per-bar and session features, integrates news sentiment analysis, and runs inference through your trained Deep Learning models (ResNLS, TCN, Compact CNN, LightGBM, etc.).

It outputs a concise, actionable list of the **Top LONG and SHORT picks** for the day, including exact **Entry**, **Target**, and **Stop Loss** levels based on the model's magnitude predictions.

---

## 🚀 Quick Start Guide

### 0. Install With `uv`
Use `uv` to create the environment and expose the package CLI:

```bash
uv sync
```

You can now run the unified CLI with:

```bash
uv run intradaynet --help
```

### 1. Basic PyTorch Execution
Run the script using the best compiled PyTorch model inside your `runs/intraday/` directory.

```bash
uv run intradaynet picks --model runs/intraday/resnls/best_model.pt
```

### 2. Basic LightGBM Execution
Run the script using a folder containing the LightGBM booster `.lgb` files.

```bash
uv run intradaynet picks --model runs/lgbm/
```

### 3. Save Output to CSV
To keep a record of the recommendations or pass them to another trading bot/script.

```bash
uv run intradaynet picks --model runs/intraday/resnls/best_model.pt --save-csv today_picks.csv
```

### 4. Skip Live Data Download (Dry Run / Testing)
Useful if you have already downloaded the data and just want to tweak prediction thresholds without waiting for `yfinance` to update again.

```bash
uv run intradaynet picks --model runs/intraday/resnls/best_model.pt --no-download
```

### 5. Get Picks for a Specific Past Date (Backtesting proxy)
Generates predictions specifically for the date provided but *only* trained on data up to the previous day.

```bash
uv run intradaynet picks --model runs/lgbm/ --date 2026-03-12
```

```bash
uv run intradaynet picks --model runs/lgbm/ --date 2026-04-01 --max-price 1000 --top-n 15 --horizon H375
```

---

## 🛠️ Detailed Parameter Reference

The script is highly customizable through its command-line arguments. Here is every parameter detailed:

### Core Parameters
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `str` | **Required** | The path to your compiled model. Can be a `.pt` PyTorch checkpoint file or a directory path containing your `.lgb` LightGBM models. |
| `--config` | `str` | `configs/intraday_config.yaml` | Path to the YAML configuration file defining the model architecture, feature spaces, and data paths. |

### Prediction Adjustments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--horizon` | `str` | `"H60"` | The prediction horizon timeframe to infer. Valid options depend on your `configs/intraday_config.yaml` but typically include: `H15` (15 mins), `H30` (30 mins), `H60` (1 hour), `H375` (Full day). |
| `--dir-threshold`| `float`| `0.60` | The minimum probability threshold the direction model must hit to classify a stock as a LONG (≥0.60) or SHORT (≤0.40). Increase this for higher accuracy / fewer picks. |
| `--min-confidence`| `float`| `0.55` | The minimum confidence score output by the PyTorch model (or calculated proxy for LightGBM). Filters out uncertain picks. |

### Output Formatting and Options
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--top-n` | `int` | `5` | The maximum number of recommendations to show for EACH direction (e.g., `5` means up to 5 longs and 5 shorts). |
| `--save-csv` | `str` | `""` (None) | If provided, writes the formatted picks (with entry, target, SL, confidence) to the specified CSV file path. |
| `--date` | `str` | `""` (None) | Generates picks for this specific date (`YYYY-MM-DD`). Automatically restricts data lookup to the day *prior*. If omitted, it auto-detects the next trading day from the most recent CSV data. |

### Trading Data Integration
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--max-price` | `float`| `0` (None) | Filters out any stock from the recommendation list whose closing price is HIGHER than this value. Useful for small accounts restricting expensive stocks. |
| `--stop-loss` | `float`| `0.01` | The default stop-loss percentage to apply to entry prices. `0.01` equals 1%. Used to calculate the strict Stop-Loss limit outputted in the tables. |
| `--no-download`| `flag` | `False` | Skips connecting to `yfinance` to append the latest minute-level data. The script will ONLY use the currently existing `.csv` minute files. |
| `--max-stocks` | `int` | `0` (All) | Limits the processing pipeline to the first `N` stocks found in your minute data directory. Useful for quick debugging. |

---

## 📊 How It Works under the hood

1. **Initialization:** The script reads `--config` to reconstruct the `SentimentFeatureBuilder` and `MarketFeatureBuilder` matching your model's pipeline structure.
2. **Data Syncing:** Unless `--no-download` is passed, the script iterates over the minute-level CSVs inside your data directory, determines the last recorded timestamp, and pulls new 1-minute interval data via `yfinance` to bridge the gap up to the current moment.
3. **Macro Downloads:** Connects and downloads the latest data for indices, crude oil, gold, and VIX to populate `context_t` session features.
4. **Feature Extraction:** For each stock, it converts the raw minute rows into sequence sliding windows (length `seq_length`). It parses news sentiment if configured.
5. **Inference Execution:**
   - **PyTorch (ResNLS/Compact CNN/TCN):** Computes confidence, direction, and magnitude tensors through `torch.no_grad()`. Checks backward compatibility between old 14-feature vs 24-feature sentiment layers.
   - **LightGBM:** Flattens the feature window sequence using rolling means and standard deviations (W=5, 30, 120), then predicts via the direction and magnitude booster `.lgb` binaries.
6. **Risk Management & Filtering:** It removes predictions failing `--dir-threshold`, `--min-confidence`, or `--max-price`. It generates a composite "Score" mapping `Confidence * Magnitude`.
7. **Ranking & Presentation:** Sorts recommendations by **Score**. Outputs a clean CLI Dashboard using the `Rich` Python library and saves it optionally to `--save-csv`.

---
## ✨ Interpreting the Output Table

Once the predictions complete, you will see a stylized summary:
* **Score:** The primary ranking metric. Higher score implies higher confidence coupled with a larger predicted price movement.
* **Target (₹):** The exact calculated Exit limit based on Model's specific `Magnitude` prediction.
* **Stop Loss (₹):** The calculated safety-net price based exactly on the `--stop-loss` fraction you set.
* **Last Close (₹):** The actual closing price representing the Entry trigger from `yfinance`.
