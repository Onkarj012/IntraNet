#!/usr/bin/env python3
"""
IntradayNet v3.0 - Integrated Training Pipeline

Combines all Phase 0 and Phase 1 improvements:
- Phase 0: Walk-forward validation, liquid universe, survivorship bias fix
- Phase 1: 4-state regime classifier, regime-conditional models, ATR-based targets

This is the main entry point for training v3.0 models.

Usage:
    # Run full walk-forward validation
    python scripts/train_v3_integrated.py --mode walkforward --start 2015-01-01 --end 2025-12-31
    
    # Train a single model for specific regime
    python scripts/train_v3_integrated.py --mode regime --regime trending_calm --start 2022-01-01 --end 2024-12-31
    
    # Analyze liquid universe evolution
    python scripts/train_v3_integrated.py --mode universe --start 2022-01-01 --end 2025-12-31
    
    # Check survivorship bias
    python scripts/train_v3_integrated.py --mode survivorship --start 2015-01-01 --end 2025-12-31
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_v3")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import v3.0 components
from intradaynet.walkforward_v3 import WalkForwardEngine, WalkForwardConfig
from intradaynet.liquid_universe import LiquidUniverseFilter
from intradaynet.regime_v3 import RegimeClassifierV3, MarketRegime
from intradaynet.survivorship_bias import SurvivorshipBiasFix
from intradaynet.dynamic_targets import DynamicTargetManager


def run_walkforward_validation(args):
    """Run anchored walk-forward validation."""
    logger.info("="*70)
    logger.info("MODE: Anchored Walk-Forward Validation")
    logger.info("="*70)
    
    config = WalkForwardConfig(
        train_months_initial=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
        use_liquid_filter=args.liquid_filter,
        use_regime_models=args.regime_models,
        max_universe_size=args.max_stocks,
        min_universe_size=args.min_stocks,
        data_dir=args.data_dir,
    )
    
    engine = WalkForwardEngine(config)
    
    if args.dry_run:
        # Just show folds
        folds = engine.create_folds(args.start, args.end)
        logger.info(f"\nDry Run: {len(folds)} folds would be created")
        for fold in folds[:5]:  # Show first 5
            logger.info(f"\nFold {fold['fold']}:")
            logger.info(f"  Train: {fold['train_start']} to {fold['train_end']}")
            logger.info(f"  Val:   {fold['val_start']} to {fold['val_end']}")
            logger.info(f"  Test:  {fold['test_start']} to {fold['test_end']}")
        if len(folds) > 5:
            logger.info(f"  ... and {len(folds) - 5} more folds")
        return
    
    # Run full validation
    results = engine.run_full_walkforward(args.start, args.end)
    
    logger.info(f"\nCompleted {len(results)} walk-forward folds")
    logger.info(f"Results saved to: {engine.config.cache_dir}")


def run_regime_specific_training(args):
    """Train models for a specific regime."""
    logger.info("="*70)
    logger.info(f"MODE: Regime-Specific Training ({args.regime})")
    logger.info("="*70)
    
    # Validate regime
    try:
        regime = MarketRegime(args.regime)
    except ValueError:
        logger.error(f"Invalid regime: {args.regime}")
        logger.info(f"Valid regimes: {[r.value for r in MarketRegime]}")
        return
    
    # Initialize components
    classifier = RegimeClassifierV3()
    liquid_filter = LiquidUniverseFilter(data_dir=args.data_dir)
    
    # Get liquid universe as of training end
    logger.info(f"Getting liquid universe as of {args.end}...")
    universe = liquid_filter.get_liquid_universe(
        as_of_date=args.end,
        max_stocks=args.max_stocks,
        min_stocks=args.min_stocks,
    )
    
    logger.info(f"Training universe: {len(universe)} stocks")
    logger.info(f"Target regime: {regime.value}")
    
    # Get regime adjustments
    _, _, adj = classifier.classify(
        vix_level=15,  # Would use actual market data
        vix_change_pct=0,
    )
    
    logger.info(f"\nRegime adjustments:")
    logger.info(f"  Direction threshold: {adj.direction_threshold}")
    logger.info(f"  Min confidence: {adj.min_confidence}")
    logger.info(f"  Max positions: {adj.max_positions}")
    logger.info(f"  Target ATR mult: {adj.target_atr_multiplier}")
    logger.info(f"  Stop ATR mult: {adj.stop_atr_multiplier}")
    logger.info(f"  Allow trading: {adj.allow_trading}")
    
    # This would continue with actual model training
    logger.info("\n[Placeholder] Would train regime-specific model here")
    logger.info("In production, this would:")
    logger.info("  1. Load features for regime-matching days")
    logger.info("  2. Apply regime-specific sampling")
    logger.info("  3. Train LightGBM with regime-tuned params")
    logger.info("  4. Save model to runs/v3_regime_{regime.value}/")


def run_universe_analysis(args):
    """Analyze liquid universe evolution."""
    logger.info("="*70)
    logger.info("MODE: Liquid Universe Analysis")
    logger.info("="*70)
    
    filter = LiquidUniverseFilter(data_dir=args.data_dir)
    
    logger.info(f"Analyzing universe evolution from {args.start} to {args.end}...")
    
    df = filter.analyze_universe_evolution(args.start, args.end)
    
    logger.info(f"\nUniverse Evolution Summary:")
    logger.info(f"  Average stocks per month: {df['count'].mean():.1f}")
    logger.info(f"  Min: {df['count'].min()}, Max: {df['count'].max()}")
    logger.info(f"  Std dev: {df['count'].std():.1f}")
    logger.info(f"  Average monthly entries: {df['new_entries'].mean():.1f}")
    logger.info(f"  Average monthly exits: {df['exits'].mean():.1f}")
    
    # Save results
    output_file = Path("liquid_universe_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'period': f"{args.start} to {args.end}",
                'avg_stocks': float(df['count'].mean()),
                'min_stocks': int(df['count'].min()),
                'max_stocks': int(df['count'].max()),
                'avg_entries': float(df['new_entries'].mean()),
                'avg_exits': float(df['exits'].mean()),
            },
            'monthly_data': df.to_dict('records'),
        }, f, indent=2, default=str)
    
    logger.info(f"\nSaved analysis to: {output_file}")
    
    # Show last 12 months
    logger.info(f"\nLast 12 months:")
    print(df.tail(12).to_string(index=False))


def run_survivorship_analysis(args):
    """Analyze survivorship bias."""
    logger.info("="*70)
    logger.info("MODE: Survivorship Bias Analysis")
    logger.info("="*70)
    
    sbf = SurvivorshipBiasFix(data_dir=args.data_dir)
    
    analysis = sbf.analyze_survivorship_bias(args.start, args.end)
    
    logger.info(f"\nSurvivorship Bias Metrics:")
    logger.info(f"  Period: {args.start} to {args.end}")
    logger.info(f"  Start universe: {analysis['start_universe_size'].iloc[0]} stocks")
    logger.info(f"  End universe: {analysis['end_universe_size'].iloc[0]} stocks")
    logger.info(f"  Survivors: {analysis['survivors'].iloc[0]} stocks")
    logger.info(f"  Delisted: {analysis['delisted_count'].iloc[0]} stocks")
    logger.info(f"  IPOs: {analysis['ipo_count'].iloc[0]} stocks")
    logger.info(f"  Survivorship bias: {analysis['survivorship_bias_pct'].iloc[0]:.1f}%")
    
    # Show delisted stocks
    delisted = sbf.get_delisted_stocks(args.start, args.end)
    if delisted:
        logger.info(f"\nDelisted stocks (would be missing from biased sample):")
        for stock in delisted[:10]:
            logger.info(f"  - {stock['symbol']} (last: {stock['last_date']})")
        if len(delisted) > 10:
            logger.info(f"  ... and {len(delisted) - 10} more")
    
    # Show IPOs
    ipos = sbf.get_ipo_stocks(args.start, args.end)
    if ipos:
        logger.info(f"\nIPO stocks (not in start universe):")
        for stock in ipos[:10]:
            logger.info(f"  - {stock['symbol']} (first: {stock['first_date']})")
        if len(ipos) > 10:
            logger.info(f"  ... and {len(ipos) - 10} more")


def run_demo(args):
    """Run a demo showing all components working together."""
    logger.info("="*70)
    logger.info("MODE: Demo - All Components")
    logger.info("="*70)
    
    # 1. Liquid Universe
    logger.info("\n1. LIQUID UNIVERSE FILTER")
    logger.info("-" * 70)
    liquid_filter = LiquidUniverseFilter(data_dir=args.data_dir)
    universe = liquid_filter.get_liquid_universe(
        as_of_date="2025-01-15",
        max_stocks=10,  # Small for demo
    )
    logger.info(f"Liquid universe (as of 2025-01-15): {len(universe)} stocks")
    for sym in universe:
        logger.info(f"  - {sym}")
    
    # 2. Survivorship Bias Check
    logger.info("\n2. SURVIVORSHIP BIAS FIX")
    logger.info("-" * 70)
    sbf = SurvivorshipBiasFix(data_dir=args.data_dir)
    historical_universe = sbf.get_universe_as_of("2022-06-01")
    logger.info(f"Historical universe (2022-06-01): {len(historical_universe)} stocks")
    logger.info(f"Sample: {', '.join(historical_universe[:5])}")
    
    # 3. Regime Classification
    logger.info("\n3. 4-STATE REGIME CLASSIFIER")
    logger.info("-" * 70)
    classifier = RegimeClassifierV3()
    
    test_scenarios = [
        (12, 0.5, "Low VIX, trending"),
        (18, 0.5, "Medium VIX, trending"),
        (12, -0.5, "Low VIX, choppy"),
        (25, -0.5, "High VIX, choppy"),
    ]
    
    for vix, trend, desc in test_scenarios:
        regime, reason, adj = classifier.classify(
            vix_level=vix,
            vix_change_pct=0,
            gap_pct=0,
        )
        logger.info(f"{desc}:")
        logger.info(f"  → Regime: {regime.value}")
        logger.info(f"  → Trading allowed: {adj.allow_trading}")
        logger.info(f"  → Max positions: {adj.max_positions}")
    
    # 4. Dynamic Targets
    logger.info("\n4. ATR-BASED DYNAMIC TARGETS")
    logger.info("-" * 70)
    target_mgr = DynamicTargetManager()
    
    entry = 1000.0
    atr = 15.0
    
    for regime, _ in test_scenarios[:2]:  # Just show 2 examples
        for conf in [0.55, 0.70]:
            target, stop, meta = target_mgr.compute_levels(
                entry_price=entry,
                atr=atr,
                side="LONG",
                regime=regime,
                confidence=conf,
            )
            
            if not meta.get('skip_trade'):
                logger.info(f"{regime.value}, conf={conf:.0%}:")
                logger.info(f"  → Target: ₹{target:.1f} (+{meta['target_distance_pct']:.2f}%)")
                logger.info(f"  → Stop: ₹{stop:.1f} (-{meta['stop_distance_pct']:.2f}%)")
                logger.info(f"  → R/R: {meta['risk_reward_ratio']:.2f}")
    
    logger.info("\n" + "="*70)
    logger.info("Demo complete! All v3.0 components operational.")
    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="IntradayNet v3.0 - Integrated Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show all components working together
    python scripts/train_v3_integrated.py --mode demo
    
    # Run walk-forward validation (dry run to see folds)
    python scripts/train_v3_integrated.py --mode walkforward --dry-run
    
    # Run actual walk-forward validation
    python scripts/train_v3_integrated.py --mode walkforward --start 2015-01-01 --end 2025-12-31
    
    # Train regime-specific model
    python scripts/train_v3_integrated.py --mode regime --regime trending_calm
    
    # Analyze universe evolution
    python scripts/train_v3_integrated.py --mode universe --start 2022-01-01 --end 2025-12-31
    
    # Check survivorship bias
    python scripts/train_v3_integrated.py --mode survivorship --start 2015-01-01 --end 2025-12-31
        """
    )
    
    parser.add_argument("--mode", type=str, required=True,
                       choices=["walkforward", "regime", "universe", "survivorship", "demo"],
                       help="Operation mode")
    
    # Data settings
    parser.add_argument("--data-dir", type=str, default="nifty500",
                       help="Directory with minute data")
    
    # Date settings
    parser.add_argument("--start", type=str, default="2015-01-01",
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31",
                       help="End date (YYYY-MM-DD)")
    
    # Walk-forward settings
    parser.add_argument("--train-months", type=int, default=84,
                       help="Initial training months")
    parser.add_argument("--val-months", type=int, default=1,
                       help="Validation months")
    parser.add_argument("--test-months", type=int, default=1,
                       help="Test months")
    parser.add_argument("--step-months", type=int, default=3,
                       help="Step size (retrain frequency)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show folds without running")
    
    # Universe settings
    parser.add_argument("--max-stocks", type=int, default=150,
                       help="Maximum universe size")
    parser.add_argument("--min-stocks", type=int, default=120,
                       help="Minimum universe size")
    parser.add_argument("--no-liquid-filter", action="store_true",
                       help="Disable liquid universe filtering")
    
    # Regime settings
    parser.add_argument("--regime", type=str,
                       help="Regime for regime-specific training")
    parser.add_argument("--no-regime-models", action="store_true",
                       help="Disable regime-specific models")
    
    args = parser.parse_args()
    
    # Derived args
    args.liquid_filter = not args.no_liquid_filter
    args.regime_models = not args.no_regime_models
    
    # Route to appropriate handler
    if args.mode == "walkforward":
        run_walkforward_validation(args)
    elif args.mode == "regime":
        run_regime_specific_training(args)
    elif args.mode == "universe":
        run_universe_analysis(args)
    elif args.mode == "survivorship":
        run_survivorship_analysis(args)
    elif args.mode == "demo":
        run_demo(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
