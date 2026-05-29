#!/usr/bin/env python3
"""OptiNet v3 Phase 1b: train + evaluate global model AND per-decision-time submodels
with the new cross-asset features (sector breadth + prior-day bhavcopy)."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA = PROJECT_ROOT / "cache/optinet_v3/intraday_dataset.parquet"
RES = PROJECT_ROOT / "results/optinet_v3"
MOD = PROJECT_ROOT / "models/optinet_v3"

NON_FEATURE = {
    "index", "trade_date", "decision_time",
    "decision_close", "session_open", "session_high", "session_low",
    "ret_1h", "ret_eod",
    "label_long_1h", "label_short_1h", "label_long_eod", "label_short_eod",
}


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    dt_dummies = pd.get_dummies(df["decision_time"], prefix="dt")
    dt_dummies.columns = [c.replace(":", "") for c in dt_dummies.columns]
    df = pd.concat([df, dt_dummies, pd.get_dummies(df["index"], prefix="idx")], axis=1)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
    return df, feat_cols


def _train(X_tr, y_tr):
    import lightgbm as lgb
    n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
    if n_pos < 10 or n_neg < 10:
        return None
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.04, num_leaves=31, min_data_in_leaf=20,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=1.0,
        class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def _eval_picks(test, score_long, score_short, threshold=0.50):
    test = test.copy()
    test["long_p"] = score_long
    test["short_p"] = score_short
    test["picked_long"] = test["long_p"] >= test["short_p"]
    test["conf"] = np.maximum(test["long_p"], test["short_p"])
    test["signed_ret"] = np.where(test["picked_long"], test["ret_1h"], -test["ret_1h"])
    picks = test[test["conf"] >= threshold]
    if picks.empty:
        return {"picks": 0, "win_rate": 0.0, "avg_signed_ret_pct": 0.0,
                "long_win_rate": 0.0, "short_win_rate": 0.0,
                "long_avg_ret_pct": 0.0, "short_avg_ret_pct": 0.0}
    wins = ((picks["picked_long"] & (picks["ret_1h"] > 0))
            | (~picks["picked_long"] & (picks["ret_1h"] < 0)))
    longs = picks[picks["picked_long"]]
    shorts = picks[~picks["picked_long"]]
    return {
        "picks": int(len(picks)),
        "win_rate": float(wins.mean()),
        "avg_signed_ret_pct": float(picks["signed_ret"].mean() * 100),
        "long_count": int(len(longs)),
        "long_win_rate": float(((longs["ret_1h"] > 0).mean()) if len(longs) else 0.0),
        "long_avg_ret_pct": float((longs["signed_ret"].mean() * 100) if len(longs) else 0.0),
        "short_count": int(len(shorts)),
        "short_win_rate": float(((shorts["ret_1h"] < 0).mean()) if len(shorts) else 0.0),
        "short_avg_ret_pct": float((shorts["signed_ret"].mean() * 100) if len(shorts) else 0.0),
    }


def main():
    from sklearn.metrics import roc_auc_score
    print("=== v3 Phase 1b: global + per-decision-time models ===\n")
    ds = pd.read_parquet(DATA)
    print(f"Dataset: {len(ds)} rows, {ds['trade_date'].nunique()} days, "
          f"{len(ds.columns)} cols")

    df, feat_cols = _build_features(ds)
    print(f"Feature count: {len(feat_cols)}\n")

    # 70/30 time split
    days = sorted(df["trade_date"].unique())
    split = int(len(days) * 0.7)
    train_days, test_days = days[:split], days[split:]
    train = df[df["trade_date"].isin(train_days)].copy()
    test = df[df["trade_date"].isin(test_days)].copy()
    X_tr = train[feat_cols].fillna(0.0)
    X_te = test[feat_cols].fillna(0.0)
    print(f"Train: {len(train)} rows ({len(train_days)} days)")
    print(f"Test : {len(test)} rows  ({len(test_days)} days)\n")

    out = {}

    # -- GLOBAL model --
    print("── GLOBAL model ──")
    g_long = _train(X_tr, train["label_long_1h"].astype(int).to_numpy())
    g_short = _train(X_tr, train["label_short_1h"].astype(int).to_numpy())
    g_long_p = g_long.predict_proba(X_te)[:, 1]
    g_short_p = g_short.predict_proba(X_te)[:, 1]
    g_auc_long = roc_auc_score(test["label_long_1h"], g_long_p)
    g_auc_short = roc_auc_score(test["label_short_1h"], g_short_p)
    g_eval = _eval_picks(test, g_long_p, g_short_p, threshold=0.50)
    print(f"  AUC long={g_auc_long:.4f}  short={g_auc_short:.4f}")
    print(f"  picks={g_eval['picks']}  win={g_eval['win_rate']:.2%}  "
          f"avg_ret={g_eval['avg_signed_ret_pct']:+.4f}%")
    print(f"  LONG : {g_eval['long_count']:4d}  win={g_eval['long_win_rate']:.2%}  "
          f"avg_ret={g_eval['long_avg_ret_pct']:+.4f}%")
    print(f"  SHORT: {g_eval['short_count']:4d}  win={g_eval['short_win_rate']:.2%}  "
          f"avg_ret={g_eval['short_avg_ret_pct']:+.4f}%")
    out["global"] = {"auc_long": g_auc_long, "auc_short": g_auc_short, **g_eval}

    # Save global models
    MOD.mkdir(parents=True, exist_ok=True)
    pickle.dump({"model": g_long, "feature_columns": feat_cols},
                open(MOD / "label_long_1h.pkl", "wb"))
    pickle.dump({"model": g_short, "feature_columns": feat_cols},
                open(MOD / "label_short_1h.pkl", "wb"))

    # -- PER-DECISION-TIME submodels --
    print("\n── PER-DECISION-TIME submodels ──")
    per_dt: dict = {}
    test_with_pdt_long = pd.Series(np.nan, index=test.index)
    test_with_pdt_short = pd.Series(np.nan, index=test.index)
    for dt in sorted(train["decision_time"].unique()):
        tr_dt = train[train["decision_time"] == dt]
        te_dt = test[test["decision_time"] == dt]
        if tr_dt.empty or te_dt.empty:
            continue
        X_tr_d = tr_dt[feat_cols].fillna(0.0)
        X_te_d = te_dt[feat_cols].fillna(0.0)
        m_long = _train(X_tr_d, tr_dt["label_long_1h"].astype(int).to_numpy())
        m_short = _train(X_tr_d, tr_dt["label_short_1h"].astype(int).to_numpy())
        if m_long is None or m_short is None:
            continue
        long_p = m_long.predict_proba(X_te_d)[:, 1]
        short_p = m_short.predict_proba(X_te_d)[:, 1]
        test_with_pdt_long.loc[te_dt.index] = long_p
        test_with_pdt_short.loc[te_dt.index] = short_p
        auc_long = roc_auc_score(te_dt["label_long_1h"], long_p)
        auc_short = roc_auc_score(te_dt["label_short_1h"], short_p)
        e = _eval_picks(te_dt, long_p, short_p, threshold=0.50)
        per_dt[dt] = {"auc_long": auc_long, "auc_short": auc_short, **e}
        print(f"  {dt}  AUC L={auc_long:.3f} S={auc_short:.3f}  "
              f"picks={e['picks']:4d}  win={e['win_rate']:.2%}  "
              f"L_win={e['long_win_rate']:.2%}/{e['long_count']:3d}  "
              f"S_win={e['short_win_rate']:.2%}/{e['short_count']:3d}")
        # Save submodel
        pickle.dump({"model": m_long, "feature_columns": feat_cols},
                    open(MOD / f"long_1h_{dt.replace(':','')}.pkl", "wb"))
        pickle.dump({"model": m_short, "feature_columns": feat_cols},
                    open(MOD / f"short_1h_{dt.replace(':','')}.pkl", "wb"))

    out["per_decision_time"] = per_dt

    # Aggregate per-dt across all decision points
    test["pdt_long_p"] = test_with_pdt_long
    test["pdt_short_p"] = test_with_pdt_short
    valid = test[test["pdt_long_p"].notna() & test["pdt_short_p"].notna()]
    if not valid.empty:
        agg_eval = _eval_picks(valid, valid["pdt_long_p"].to_numpy(),
                                 valid["pdt_short_p"].to_numpy(), threshold=0.50)
        print(f"\n  aggregated per-dt: picks={agg_eval['picks']}  "
              f"win={agg_eval['win_rate']:.2%}  "
              f"avg_ret={agg_eval['avg_signed_ret_pct']:+.4f}%")
        print(f"    LONG : {agg_eval['long_count']:4d}  win={agg_eval['long_win_rate']:.2%}  "
              f"avg_ret={agg_eval['long_avg_ret_pct']:+.4f}%")
        print(f"    SHORT: {agg_eval['short_count']:4d}  win={agg_eval['short_win_rate']:.2%}  "
              f"avg_ret={agg_eval['short_avg_ret_pct']:+.4f}%")
        out["per_dt_aggregated"] = agg_eval

    # Per-(dt, index) for the global model — find the strong cells
    print("\n── Strong cells from GLOBAL model (threshold 0.50) ──")
    test["g_long_p"] = g_long_p
    test["g_short_p"] = g_short_p
    test["g_picked_long"] = test["g_long_p"] >= test["g_short_p"]
    test["g_conf"] = np.maximum(test["g_long_p"], test["g_short_p"])
    test["g_signed"] = np.where(test["g_picked_long"], test["ret_1h"], -test["ret_1h"])
    cells = test[test["g_conf"] >= 0.50]
    cell_break = cells.groupby(["decision_time", "index"]).apply(
        lambda g: pd.Series({
            "picks": int(len(g)),
            "win_rate": float(((g["g_picked_long"] & (g["ret_1h"] > 0))
                                | (~g["g_picked_long"] & (g["ret_1h"] < 0))).mean()),
            "avg_signed_ret_pct": float(g["g_signed"].mean() * 100),
        }), include_groups=False).reset_index()
    cell_break = cell_break.sort_values("avg_signed_ret_pct", ascending=False)
    print(cell_break.to_string(index=False))
    out["cell_breakdown"] = cell_break.to_dict(orient="records")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "phase1b_summary.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\nWritten to {RES/'phase1b_summary.json'}")


if __name__ == "__main__":
    main()
