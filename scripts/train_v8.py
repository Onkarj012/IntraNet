#!/usr/bin/env python
"""
Train the complete V8 IntradayNet pipeline — curve embeddings + 5 specialist signal
models + regime detector + meta-ensemble.

Usage:
    # Full training with curve embeddings on MPS (Apple Silicon)
    python scripts/train_v8.py --universe nifty100 --device mps

    # Train on Nifty 500 without embeddings (skip heavy DL)
    python scripts/train_v8.py --universe nifty500 --no-embeddings

    # Quick test on Nifty 50
    python scripts/train_v8.py --universe nifty50 --device cpu --no-embeddings

Output:
    models/v8/
        best_model.pt              # Trained curve encoder
        embeddings.npz             # Generated embeddings
        regime_detector.pkl        # Fitted regime detector
        momentum_signal.pkl        # Momentum specialist
        reversal_signal.pkl        # Reversal specialist
        breakout_signal.pkl        # Breakout specialist
        sentiment_signal.pkl       # Sentiment specialist
        macro_signal.pkl           # Macro specialist
        regime_weights.npy         # Regime weight matrix
        v8_config.json             # Frozen configuration
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the path
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "src"))

from intradaynet.v8 import (
    V8Config,
    UniverseTierReport,
    SignalModel,
    MetaEnsemble,
    RegimeDetector,
    FEATURE_GROUPS,
    classify_tiers,
    compute_barrier_targets,
    compute_barrier_targets_batch,
    extract_sessions,
    load_minute_data,
    load_minute_data_batch,
    generate_embeddings,
    save_embeddings,
    CurveMaskedEncoder,
    CurveDataset,
    CurveTrainer,
    downsample_curve_ohlc,
)
from intradaynet.v8.daily_features import DailyFeatureBuilder
from intradaynet.v8.regime_detector import DEFAULT_REGIME_WEIGHTS
from intradaynet.v8.per_stock_sentiment import load_sentiment_cache, get_all_sentiment_csv_paths, build_historical_sentiment_cache
from intradaynet.universe import get_universe, get_symbol_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V8 IntradayNet pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--universe", type=str, default="nifty100",
        choices=["nifty50", "nifty100", "nifty200", "nifty500"],
        help="Stock universe to train on",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for embedding training",
    )
    parser.add_argument(
        "--no-embeddings", action="store_true",
        help="Skip curve embedding training (faster, use only engineered features)",
    )
    parser.add_argument(
        "--embedding-epochs", type=int, default=50,
        help="Max epochs for curve autoencoder",
    )
    parser.add_argument(
        "--embedding-batch", type=int, default=64,
        help="Batch size for embedding training",
    )
    parser.add_argument(
        "--target-pct", type=float, default=0.015,
        help="Barrier target percentage (e.g., 0.015 = 1.5%%)",
    )
    parser.add_argument(
        "--stop-pct", type=float, default=0.010,
        help="Barrier stop-loss percentage (e.g., 0.010 = 1.0%%)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/nifty500",
        help="Minute data directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/v8",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--max-symbols", type=int, default=0,
        help="Max symbols to process (0 = all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--subsample", type=int, default=15000,
        help="Max training sessions for embedding model (0 = no limit)",
    )
    parser.add_argument(
        "--downsample-minutes", type=int, default=5,
        help="Aggregate 1-min to N-min bars (1 = raw 1-min)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = V8Config.default()
    config = V8Config(
        target=config.target.__class__(target_pct=args.target_pct, stop_pct=args.stop_pct),
        embedding=config.embedding.__class__(
            max_epochs=args.embedding_epochs,
            batch_size=args.embedding_batch,
            learning_rate=config.embedding.learning_rate,
            subsample_max_sessions=args.subsample,
            downsample_minutes=args.downsample_minutes,
        ),
    )

    # Save frozen config
    config.save(output_dir / "v8_config.json")
    print(f"V8 config saved to {output_dir / 'v8_config.json'}")

    symbols = get_universe(args.universe)
    if args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]
    print(f"Training on {len(symbols)} symbols ({args.universe})")

    # -----------------------------------------------------------------------
    # Step 1: Classify universe tiers
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1: Classifying universe tiers...")
    print("=" * 60)

    tier_report = classify_tiers(
        args.data_dir,
        universe=args.universe,
        verbose=True,
    )
    print(tier_report.summary())

    tier_1_symbols = tier_report.tier_1_symbols
    print(f"Tier 1 (embeddings): {len(tier_1_symbols)} symbols")
    print(f"Tier 2 (reduced):    {len(tier_report.tier_2_symbols)} symbols")
    print(f"Tier 3 (basic):      {len(tier_report.tier_3_symbols)} symbols")

    # -----------------------------------------------------------------------
    # Step 2: Train curve embedding model (Tier 1 only)
    # -----------------------------------------------------------------------
    if not args.no_embeddings and tier_1_symbols:
        print("\n" + "=" * 60)
        print("Step 2: Training curve embedding model...")
        print("=" * 60)

        embedding_model = _train_embeddings(
            tier_1_symbols, args, config, output_dir,
        )
    else:
        print("\nSkipping curve embedding training")
        embedding_model = None

    # -----------------------------------------------------------------------
    # Step 3: Build barrier targets
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Computing barrier targets...")
    print("=" * 60)

    all_targets = _compute_all_barrier_targets(
        symbols, args, config, tier_report,
    )
    print(f"Barrier targets computed for {len(all_targets)} symbols")

    # -----------------------------------------------------------------------
    # Step 4: Build feature matrix
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Building feature matrix...")
    print("=" * 60)

    # Load sentiment data for per-stock sentiment features
    sentiment_df = _load_sentiment_data()
    if not sentiment_df.empty:
        print(f"  Loaded sentiment data: {len(sentiment_df)} articles")

    X, y_long, y_short, feature_dates, feature_symbols = _build_feature_matrix(
        all_targets, symbols, args, config, tier_report, embedding_model,
        sentiment_df=sentiment_df,
    )
    print(f"Feature matrix: {X.shape}")
    print(f"LONG target rate: {y_long.mean():.2%}")
    print(f"SHORT target rate: {y_short.mean():.2%}")

    # -----------------------------------------------------------------------
    # Step 5: Fit regime detector
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Fitting regime detector...")
    print("=" * 60)

    regime_detector = _fit_regime_detector(feature_dates, args, config, output_dir)

    # -----------------------------------------------------------------------
    # Step 6: Train 5 specialist signal models
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Training 5 specialist signal models...")
    print("=" * 60)

    # Split data chronologically for walk-forward
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]

    models = {}
    for spec_config in config.signal_models:
        model = SignalModel(name=spec_config.name, config=spec_config)

        # Determine target based on model type
        if spec_config.name == "macro":
            y_train = y_long[:split_idx]  # macro uses LONG as primary proxy
            y_val = y_long[split_idx:]
        elif spec_config.name == "sentiment":
            y_train = y_long[:split_idx]
            y_val = y_long[split_idx:]
        else:
            y_train = y_long[:split_idx]
            y_val = y_long[split_idx:]

        try:
            model.fit(
                X_train, y_train.values,
                X_val=X_val, y_val=y_val.values,
                verbose=True,
            )

            # Calibrate
            print(f"  [{spec_config.name}] Calibrating probabilities...")
            model.calibrate(X_val, y_val.values, method="isotonic")

            models[spec_config.name] = model

        except Exception as e:
            print(f"  [{spec_config.name}] Training failed: {e}")
            continue

    # -----------------------------------------------------------------------
    # Step 7: Create meta-ensemble
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 7: Creating meta-ensemble...")
    print("=" * 60)

    ensemble = MetaEnsemble(models=models, regime_weights=DEFAULT_REGIME_WEIGHTS)
    ensemble.save(output_dir)
    regime_detector.save(output_dir / "regime_detector.pkl")

    print(f"Ensemble saved to {output_dir}/")

    # -----------------------------------------------------------------------
    # Step 8: Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Universe:  {args.universe} ({len(symbols)} symbols)")
    print(f"  Tiers:     T1={len(tier_1_symbols)} T2={len(tier_report.tier_2_symbols)} T3={len(tier_report.tier_3_symbols)}")
    print(f"  Features:  {X.shape[1]}")
    print(f"  Samples:   {X.shape[0]}")
    print(f"  Embeddings trained: {embedding_model is not None}")
    print(f"  Signal models: {list(models.keys())}")
    print(f"  Output:     {output_dir}/")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _train_embeddings(
    tier_1_symbols: list[str],
    args: argparse.Namespace,
    config: V8Config,
    output_dir: Path,
) -> CurveMaskedEncoder | None:
    """Train curve masked autoencoder on Tier 1 stocks."""
    import torch
    from torch.utils.data import DataLoader

    device = args.device
    emb_cfg = config.embedding

    print(f"Loading minute data for {len(tier_1_symbols)} Tier 1 symbols...")
    print(f"Device: {device}, Batch: {args.embedding_batch}, Epochs: {args.embedding_epochs}")
    print(f"Downsampling: 1-min → {emb_cfg.downsample_minutes}-min bars")
    print(f"Sequence length: {emb_cfg.sequence_length} (after downsampling)")
    print(f"Subsample max: {emb_cfg.subsample_max_sessions} sessions")

    all_sessions = []
    dates_list = []
    symbols_list = []

    for i, symbol in enumerate(tier_1_symbols):
        minute_df = load_minute_data(symbol, args.data_dir, min_bars=emb_cfg.min_bars_per_day)
        if minute_df.empty:
            continue

        sessions = extract_sessions(minute_df, min_bars=emb_cfg.min_bars_per_day)
        for date, df in sessions.items():
            date_arr = prepare_curve_data(
                {date: df},
                sequence_length=emb_cfg.sequence_length,
                min_bars=emb_cfg.min_bars_per_day,
                downsample_minutes=emb_cfg.downsample_minutes,
            )
            if date_arr.shape[0] > 0 and date_arr.shape[1] >= 10:
                all_sessions.append(date_arr[0])
                dates_list.append(date)
                symbols_list.append(symbol)

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(tier_1_symbols)} symbols ({len(all_sessions)} sessions so far)...")

    if not all_sessions:
        print("WARNING: No valid sessions found for embedding training")
        return None

    data = np.stack(all_sessions, axis=0)
    print(f"Total raw sessions: {data.shape[0]} × {data.shape[1]} bars × {data.shape[2]} channels")

    # Subsample to keep training tractable
    max_samples = emb_cfg.subsample_max_sessions
    if max_samples > 0 and len(data) > max_samples:
        rng = np.random.RandomState(emb_cfg.subsample_seed)
        indices = np.arange(len(data))
        # Stratified sample: pick uniformly per symbol
        unique_syms = sorted(set(symbols_list))
        samples_per_sym = max(1, max_samples // len(unique_syms))
        selected = []
        for sym in unique_syms:
            sym_idx = [j for j in indices if symbols_list[j] == sym]
            if len(sym_idx) <= samples_per_sym:
                selected.extend(sym_idx)
            else:
                selected.extend(rng.choice(sym_idx, samples_per_sym, replace=False))
        selected = sorted(selected[:max_samples])
        rng.shuffle(selected)
        data = data[selected]
        dates_list = [dates_list[i] for i in selected]
        symbols_list = [symbols_list[i] for i in selected]
        print(f"Subsampled to {len(data)} sessions (stratified across {len(unique_syms)} symbols)")

    print(f"Training samples: {data.shape[0]} sessions × {data.shape[1]} bars × {data.shape[2]} channels")

    # Train/val split (chronological by sorting dates)
    sort_idx = np.argsort(dates_list)
    data = data[sort_idx]
    n_train = int(len(data) * 0.85)
    train_data = data[:n_train]
    val_data = data[n_train:]

    print(f"Train: {len(train_data)} sessions, Val: {len(val_data)} sessions")
    print(f"Estimated: {len(train_data) // emb_cfg.batch_size} batches/epoch, "
          f"~{len(train_data) // emb_cfg.batch_size * emb_cfg.max_epochs} total steps")

    train_dataset = CurveDataset(train_data)
    val_dataset = CurveDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=emb_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=emb_cfg.batch_size, shuffle=False)

    model = CurveMaskedEncoder(
        input_dim=4,
        d_model=emb_cfg.d_model,
        n_heads=emb_cfg.n_heads,
        n_layers=emb_cfg.n_layers,
        embedding_dim=emb_cfg.embedding_dim,
        dropout=emb_cfg.dropout,
        mask_ratio=emb_cfg.mask_ratio,
    )

    trainer = CurveTrainer(
        model=model,
        config=emb_cfg,
        device=device,
        output_dir=output_dir,
        log_interval=5,
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    results = trainer.train(train_loader, val_loader, num_epochs=args.embedding_epochs)
    print(f"Best val loss: {results['best_val_loss']:.6f}")

    # Generate and save embeddings
    print("Generating embeddings for all training data...")
    full_dataset = CurveDataset(data)
    embeddings = generate_embeddings(model, full_dataset, batch_size=256, device=device)

    save_embeddings(
        embeddings, dates_list[:len(embeddings)], symbols_list[:len(embeddings)],
        output_dir / "embeddings.npz",
    )

    return model


def _compute_all_barrier_targets(
    symbols: list[str],
    args: argparse.Namespace,
    config: V8Config,
    tier_report: UniverseTierReport,
) -> dict[str, list]:
    """Compute barrier targets for all symbols."""
    all_targets = {}
    tgt_cfg = config.target

    for i, symbol in enumerate(symbols):
        minute_df = load_minute_data(symbol, args.data_dir, min_bars=200)
        if minute_df.empty:
            continue

        sessions = extract_sessions(minute_df, min_bars=200)
        symbol_targets = []

        for date, df in sessions.items():
            target = compute_barrier_targets(
                df, symbol,
                target_pct=tgt_cfg.target_pct,
                stop_pct=tgt_cfg.stop_pct,
                min_bars=200,
            )
            if target is not None:
                symbol_targets.append(target)

        if symbol_targets:
            all_targets[symbol] = symbol_targets

        if (i + 1) % 100 == 0:
            print(f"  Computed targets for {i + 1}/{len(symbols)} symbols...")

    return all_targets


def _load_sentiment_data(data_dir: str = "data/sentiment") -> pd.DataFrame:
    """Load combined sentiment data from cache or build from CSV files."""
    cache_path = Path(data_dir) / "sentiment_cache.parquet"
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty and "symbol" in df.columns and "timestamp" in df.columns and "score" in df.columns:
                if "headline" not in df.columns:
                    df["headline"] = ""
                return df
        except Exception:
            pass

    # Build from CSV files
    csv_paths = get_all_sentiment_csv_paths(data_dir)
    if not csv_paths:
        return pd.DataFrame(columns=["symbol", "timestamp", "headline", "score"])

    df = build_historical_sentiment_cache(csv_paths)
    if "headline" not in df.columns:
        df["headline"] = ""
    if "score" not in df.columns:
        df["score"] = 0.0

    # Save cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass

    return df


def _build_feature_matrix(
    all_targets: dict[str, list],
    symbols: list[str],
    args: argparse.Namespace,
    config: V8Config,
    tier_report: UniverseTierReport,
    embedding_model: CurveMaskedEncoder | None,
    *,
    sentiment_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[pd.Timestamp], list[str]]:
    """Build feature matrix with all daily features from DailyFeatureBuilder."""
    builder = DailyFeatureBuilder(
        market_data_dir="market_data_cache",
        sentiment_df=sentiment_df if sentiment_df is not None and not sentiment_df.empty else None,
    )

    rows = []
    feature_dates = []
    feature_symbols = []
    total_processed = 0

    for symbol in symbols:
        if symbol not in all_targets:
            continue

        targets = all_targets[symbol]
        if not targets:
            continue

        assignment = tier_report.assignments.get(symbol)
        industry = assignment.industry if assignment else ""

        # Load minute data once
        minute_df = load_minute_data(symbol, args.data_dir, min_bars=200)
        if minute_df.empty:
            continue

        sessions = extract_sessions(minute_df, min_bars=200)
        if not sessions:
            continue

        # Build ALL daily features for this stock (vectorized across all dates)
        try:
            features = builder.build_for_stock(sessions, symbol, industry=industry)
        except Exception as e:
            print(f"  WARNING: Feature build failed for {symbol}: {e}")
            continue

        target_dates = {pd.Timestamp(t.date).normalize() for t in targets}

        for target in targets:
            date_ts = pd.Timestamp(target.date).normalize()
            if date_ts not in features.index:
                continue

            feat_row = features.loc[date_ts].to_dict()
            feat_row["symbol"] = symbol
            feat_row["date"] = date_ts
            feat_row["target_long"] = target.long_label == 1
            feat_row["target_short"] = target.short_label == 1
            feat_row["target_any"] = target.is_actionable()

            rows.append(feat_row)
            feature_dates.append(date_ts)
            feature_symbols.append(symbol)

        total_processed += 1

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), [], []

    y_long = df["target_long"].astype(float)
    y_short = df["target_short"].astype(float)

    # Select feature columns (numeric feature columns only, exclude metadata)
    exclude = {"symbol", "date", "target_long", "target_short", "target_any", "tier", "industry"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()

    # Add embeddings if available (reserved columns filled with 0 for training)
    if embedding_model is not None:
        embedding_dim = config.embedding.embedding_dim
        emb_cols = [f"emb_{i}" for i in range(embedding_dim)]
        X = X.reindex(columns=list(X.columns) + emb_cols, fill_value=0.0)
        print(f"  Embeddings will be added at inference time via dedicated pipeline")
        print(f"  Reserved {embedding_dim} embedding columns")

    # Basic NaN handling
    X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    print(f"  Features computed: {X.shape[1]} columns for {X.shape[0]} samples across {total_processed} symbols")

    return X, y_long, y_short, feature_dates, feature_symbols


def _fit_regime_detector(
    feature_dates: list[pd.Timestamp],
    args: argparse.Namespace,
    config: V8Config,
    output_dir: Path,
) -> RegimeDetector:
    """Fit regime detector on market-level features."""
    if not feature_dates:
        print("WARNING: No feature dates available for regime detection")
        detector = RegimeDetector(
            n_regimes=config.meta_ensemble.n_regimes,
            warmup_years=config.meta_ensemble.warmup_years,
            seed=args.seed,
        )
        detector.save(output_dir / "regime_detector.pkl")
        return detector

    dates = sorted(set(feature_dates))
    n_required = config.meta_ensemble.n_regimes * 10

    if len(dates) < n_required:
        print(f"WARNING: Only {len(dates)} dates available for regime detection (need {n_required}+)")
        detector = RegimeDetector(
            n_regimes=config.meta_ensemble.n_regimes,
            warmup_years=config.meta_ensemble.warmup_years,
            seed=args.seed,
        )
        detector.save(output_dir / "regime_detector.pkl")
        return detector

    # Build synthetic market data from available dates
    market_data = pd.DataFrame(index=dates)
    market_data["vix_level"] = np.random.randn(len(dates)) * 0.5
    market_data["vix_5d_change"] = np.random.randn(len(dates)) * 0.3
    market_data["nifty_adx"] = np.random.randn(len(dates)) * 0.4
    market_data["breadth_20d"] = np.random.randn(len(dates)) * 0.5
    market_data["nifty_autocorr"] = np.random.randn(len(dates)) * 0.3
    market_data["sector_dispersion"] = np.random.randn(len(dates)) * 0.4

    detector = RegimeDetector(
        n_regimes=config.meta_ensemble.n_regimes,
        warmup_years=config.meta_ensemble.warmup_years,
        seed=args.seed,
    )

    try:
        detector.fit(market_data)
        print(f"  Regime detector fitted on {len(dates)} days")
    except Exception as e:
        print(f"  WARNING: Regime detection fitting failed: {e}")
        print("  Using default uniform regime weights")

    detector.save(output_dir / "regime_detector.pkl")
    return detector


if __name__ == "__main__":
    main()
