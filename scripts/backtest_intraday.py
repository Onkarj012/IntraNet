#!/usr/bin/env python3
"""
IntradayNet Backtester.

Simulates intraday trading using a trained model.

Usage:
    python scripts/backtest_intraday.py --config configs/intraday_config.yaml --model runs/intraday/best_model.pt
    python scripts/backtest_intraday.py --config configs/intraday_config.yaml --model runs/intraday/best_model.pt --strategy momentum
"""

import argparse
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.config import load_config, IntradayConfig
from intradaynet.models.tcn_attention import IntradayTCNAttention
from intradaynet.dataset.intraday_dataset import IntradayDataset, collate_intraday

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_intraday")


@dataclass
class Trade:
    """A single trade record."""
    symbol: str
    entry_time: str
    exit_time: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    exit_reason: str  # "target", "stop_loss", "eod"


STRATEGY_CONFIGS = {
    "scalp": {"horizon_idx": 0, "threshold": 0.70, "stop_loss": 0.003},
    "momentum": {"horizon_idx": 1, "threshold": 0.65, "stop_loss": 0.005},
    "swing": {"horizon_idx": 2, "threshold": 0.60, "stop_loss": 0.007},
    "eod": {"horizon_idx": 3, "threshold": 0.60, "stop_loss": 0.005},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest IntradayNet")
    parser.add_argument("--config", type=str, default="configs/intraday_config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument("--strategy", type=str, default="momentum",
                        choices=list(STRATEGY_CONFIGS.keys()))
    parser.add_argument("--subset-stocks", type=str, default="")
    parser.add_argument("--max-stocks", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def run_backtest(
    model: IntradayTCNAttention,
    dataloader: DataLoader,
    cfg: IntradayConfig,
    strategy: str = "momentum",
    device: torch.device = torch.device("cpu"),
) -> pd.DataFrame:
    """
    Run backtest on test data.

    Returns DataFrame of trade results.
    """
    model.eval()
    strat = STRATEGY_CONFIGS[strategy]
    horizon_idx = strat["horizon_idx"]
    threshold = strat["threshold"]
    stop_loss = strat["stop_loss"]

    trades = []
    total_predictions = 0
    correct_predictions = 0

    for batch in dataloader:
        per_bar = batch["per_bar"].to(device)
        context = batch["context"].to(device)
        sentiment = batch["sentiment"].to(device)
        targets = batch["targets"]

        predictions = model(per_bar, context, sentiment)

        probs = torch.sigmoid(predictions["direction_logits"])[:, horizon_idx]
        confidences = predictions["confidences"][:, horizon_idx]
        magnitudes = predictions["magnitudes"][:, horizon_idx]

        tgt_dir = targets["direction"][:, horizon_idx]
        tgt_mag = targets["magnitude"][:, horizon_idx]

        for i in range(per_bar.size(0)):
            prob = probs[i].item()
            conf = confidences[i].item()
            magnitude = tgt_mag[i].item()

            total_predictions += 1

            # Direction accuracy tracking
            pred_up = prob > 0.5
            actual_up = tgt_dir[i].item() > 0.5
            if pred_up == actual_up:
                correct_predictions += 1

            # Trade entry decision
            if prob >= threshold and conf >= 0.5:
                direction = "LONG"
                gross_return = magnitude
            elif prob <= (1 - threshold) and conf >= 0.5:
                direction = "SHORT"
                gross_return = -magnitude
            else:
                continue

            # Apply stop loss
            if gross_return < -stop_loss:
                exit_reason = "stop_loss"
                net_return = -stop_loss
            else:
                exit_reason = "target"
                net_return = gross_return

            # Transaction costs
            brokerage = 2 * cfg.backtest.brokerage_per_order / cfg.backtest.position_size
            stt = cfg.backtest.stt_rate  # on sell side
            slippage = cfg.backtest.slippage_rate
            total_cost = brokerage + stt + slippage

            pnl = (net_return - total_cost) * cfg.backtest.position_size

            trades.append(Trade(
                symbol="",
                entry_time="",
                exit_time="",
                direction=direction,
                entry_price=0,
                exit_price=0,
                pnl=pnl,
                return_pct=(net_return - total_cost) * 100,
                exit_reason=exit_reason,
            ))

    # Summary
    if trades:
        pnls = [t.pnl for t in trades]
        returns = [t.return_pct for t in trades]
        winning = [r for r in returns if r > 0]

        logger.info(f"\n{'═' * 60}")
        logger.info(f"BACKTEST RESULTS — Strategy: {strategy.upper()}")
        logger.info(f"{'═' * 60}")
        logger.info(f"  Total predictions:      {total_predictions}")
        logger.info(f"  Direction accuracy:      {correct_predictions / max(total_predictions, 1):.1%}")
        logger.info(f"  Total trades:            {len(trades)}")
        logger.info(f"  Win rate:                {len(winning) / max(len(trades), 1):.1%}")
        logger.info(f"  Avg return/trade:        {np.mean(returns):.3f}%")
        logger.info(f"  Total PnL:               ₹{sum(pnls):,.0f}")
        logger.info(f"  Avg PnL/trade:           ₹{np.mean(pnls):,.0f}")
        logger.info(f"  Max win:                 ₹{max(pnls):,.0f}")
        logger.info(f"  Max loss:                ₹{min(pnls):,.0f}")

        # Sharpe ratio (daily approximation)
        if len(returns) > 1:
            sharpe = np.mean(returns) / max(np.std(returns), 1e-10) * np.sqrt(252)
            logger.info(f"  Sharpe ratio (annualized): {sharpe:.2f}")

        stop_outs = sum(1 for t in trades if t.exit_reason == "stop_loss")
        logger.info(f"  Stop-loss exits:         {stop_outs} ({stop_outs / max(len(trades), 1):.0%})")
        logger.info(f"{'═' * 60}")
    else:
        logger.warning("No trades executed! Try lowering the threshold.")

    return pd.DataFrame([vars(t) for t in trades]) if trades else pd.DataFrame()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Load model
    checkpoint = torch.load(args.model, map_location=cfg.device, weights_only=False)
    model = IntradayTCNAttention(
        num_per_bar_features=cfg.model.num_per_bar_features,
        num_context_features=cfg.model.num_context_features,
        num_sentiment_features=cfg.model.num_sentiment_features,
        hidden_dim=cfg.model.hidden_dim,
        tcn_channels=cfg.model.tcn.channels,
        kernel_size=cfg.model.tcn.kernel_size,
        dilation_base=cfg.model.tcn.dilation_base,
        attn_heads=cfg.model.attn_heads,
        num_horizons=len(cfg.horizons),
        dropout=cfg.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(cfg.device)
    model.to(device)
    logger.info(f"Loaded model from {args.model} (epoch {checkpoint.get('epoch', '?')})")

    # Symbols
    symbols = None
    if args.subset_stocks:
        symbols = [s.strip() for s in args.subset_stocks.split(",")]
    elif args.max_stocks > 0:
        all_csvs = sorted(Path(cfg.data.minute_data_dir).glob("*_minute.csv"))
        symbols = [f.stem.replace("_minute", "") for f in all_csvs[:args.max_stocks]]

    # Test dataset
    test_ds = IntradayDataset(
        minute_data_dir=cfg.data.minute_data_dir,
        sentiment_csv=cfg.data.sentiment_csv,
        symbols=symbols,
        date_start=cfg.splits.test_start,
        date_end=cfg.splits.test_end,
        sequence_length=cfg.model.sequence_length,
        horizons=cfg.horizons,
        sample_interval=cfg.train.sample_interval,
        market_open=cfg.data.market_open,
        market_close=cfg.data.market_close,
    )

    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size,
        shuffle=False, collate_fn=collate_intraday,
    )

    logger.info(f"Test dataset: {len(test_ds)} samples")

    # Run backtest
    results_df = run_backtest(model, test_loader, cfg, args.strategy, device)

    # Save results
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not results_df.empty:
        results_path = out_dir / f"backtest_{args.strategy}.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
