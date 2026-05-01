# Morning Picks CLI

This is the current recommendation workflow for IntradayNet. The unified CLI command is `intradaynet picks`, which maps to `scripts/recommend_intraday.py`.

Use `premarket` when you want picks before the open from the latest completed session. Use `post-open` when you want gap-aware picks after the first live bars are available.

## Common Commands

Premarket, with explicit pick counts:

```bash
uv run intradaynet picks --mode premarket --long-count 5 --short-count 5
```

Post-open, with explicit pick counts:

```bash
uv run intradaynet picks --mode post-open --post-open-cutoff 09:30 --long-count 5 --short-count 5
```

If you prefer the script directly:

```bash
python scripts/recommend_intraday.py --mode premarket --long-count 5 --short-count 5
python scripts/recommend_intraday.py --mode post-open --post-open-cutoff 09:30 --long-count 5 --short-count 5
```

Useful variants:

```bash
uv run intradaynet picks --mode premarket --target-date 2026-04-24 --long-count 3 --short-count 2
uv run intradaynet picks --mode post-open --target-date 2026-04-24 --post-open-cutoff 09:30 --long-count 3 --short-count 2
uv run intradaynet picks --mode premarket --industry "Information Technology" --long-count 3 --short-count 2
uv run intradaynet picks --mode post-open --industry "Automobile and Auto Components, Financial Services" --post-open-news-cutoff 09:20
uv run intradaynet picks --mode premarket --save-csv recommendations/picks.csv --save-json recommendations/picks.json
uv run intradaynet picks --mode post-open --allow-below-preferred --long-count 5 --short-count 5
```

## What The Main Flags Do

| Flag | Default | What it controls |
| :--- | :--- | :--- |
| `--mode` | `premarket` | Chooses premarket or post-open scoring.
| `--long-count` | `3` | Number of LONG picks to return.
| `--short-count` | `2` | Number of SHORT picks to return.
| `--per-side` | `-1` | Overrides both long and short counts with the same value.
| `--target-date` | next business day | Picks-for date in `YYYY-MM-DD` format.
| `--industry` | all industries | Filters the universe to one or more exact CSV industry values. Repeat the flag or pass a comma-separated list.
| `--post-open-cutoff` | `09:30` | Cutoff time used in post-open mode.
| `--post-open-news-cutoff` | `09:20` | Latest live-news timestamp included in post-open mode.
| `--post-open-min-alignment` | `0.05` | Minimum same-day alignment score in post-open mode.
| `--risk-profile` | `balanced` | Selects the default threshold bundle.
| `--min-confidence` | profile default | Overrides the minimum confidence gate.
| `--min-predicted-magnitude` | profile default | Overrides the minimum predicted magnitude gate.
| `--max-stocks` | `0` | Limits how many symbols are scored.
| `--allow-below-preferred` | off | Backfills requested slots with below-threshold names if not enough preferred picks exist.
| `--max-feature-staleness-bdays` | `0` | Rejects symbols whose latest feature row is too old.
| `--target-pct` | profile default | Sets the target level used for entry/target calculations.
| `--stop-loss-pct` | profile default | Sets the stop-loss level used for entry/stop calculations.
| `--refresh-yfinance` / `--no-refresh-yfinance` | on | Enables or disables price backfill from `yfinance`.
| `--disable-live-news` | off | Disables live `yfinance` news and falls back to historical sentiment rows only.
| `--live-news-required` | off | Fails the run if live-news coverage is too low.
| `--save-csv` | `default` | Writes the ranked picks CSV into `recommendations/` unless you pass a path.
| `--save-json` | `default` | Writes the full JSON payload into `recommendations/` unless you pass a path.

## Recommended Settings

For higher precision, start with:

```bash
uv run intradaynet picks --mode premarket --risk-profile balanced --long-count 3 --short-count 2
```

For post-open, the default cutoff is usually enough:

```bash
uv run intradaynet picks --mode post-open --post-open-cutoff 09:30 --risk-profile balanced --long-count 3 --short-count 2
```

If you want only preferred picks and are willing to accept fewer total names, leave `--allow-below-preferred` off. If you want the requested count no matter what, add `--allow-below-preferred`.

## Output

The CLI prints ranked LONG and SHORT tables with:

- `Pref` indicating whether the pick passes the preferred filter.
- `Conf` showing the adjusted confidence.
- `Score` showing the ranking score used to sort picks.
- `Target` and `Stop` showing the computed trade levels.

The JSON output also includes the full recommendation payload, readiness check, and summary counts.

## Notes

- Premarket mode uses the latest completed session and refreshed macro/news inputs.
- Live news is the default recommendation input. Historical sentiment CSV data is used as rolling fallback context and for training/backtests.
- Premarket news is cut off at market open (`09:15`) for the target trading day, so overnight articles flow into the next session.
- Post-open mode uses the same base model but reranks with the live opening session up to the price cutoff, and only includes live news up to `--post-open-news-cutoff`.
- If `--max-feature-staleness-bdays 0` is left at the default, stale symbols are filtered out aggressively.
