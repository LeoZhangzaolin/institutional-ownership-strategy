#!/usr/bin/env python3
"""Regime Detection"""
import pandas as pd
import numpy as np
import logging
from functools import lru_cache
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _classify_regime_cached(vol: float, ret: float, threshold: float) -> str:
    if np.isnan(vol):
        return "normal"
    if vol > threshold:
        return "high_vol"
    elif ret > 0.02:
        return "bull"
    elif ret < -0.02:
        return "bear"
    else:
        return "normal"


def detect_regime_for_dataframe(
    df: pd.DataFrame,
    vol_col: str = "sp500_ret_next",
    lookback: int = 8,
    vol_threshold: float = 0.15,
    ann: int = 4,
) -> pd.DataFrame:
    logger.info("[Regime Detection] Classifying...")
    if vol_col not in df.columns:
        df["regime"] = "normal"
        return df
    df = df.copy()
    market_data = (
        df[["date_q_end", vol_col]]
        .drop_duplicates("date_q_end")
        .sort_values("date_q_end")
    )
    market_data["rolling_vol"] = market_data[vol_col].rolling(
        window=lookback, min_periods=3
    ).std() * np.sqrt(ann)
    market_data["rolling_ret"] = (
        market_data[vol_col].rolling(window=lookback, min_periods=3).mean()
    )
    market_data["regime"] = market_data.apply(
        lambda row: _classify_regime_cached(
            row["rolling_vol"], row["rolling_ret"], vol_threshold
        ),
        axis=1,
    )
    df = df.merge(market_data[["date_q_end", "regime"]], on="date_q_end", how="left")
    df["regime"] = df["regime"].fillna("normal")
    regime_counts = df.groupby("date_q_end")["regime"].first().value_counts()
    logger.info(f"[Regime] Distribution: {dict(regime_counts)}")
    return df


def detect_current_regime(
    sp500_returns: pd.Series,
    lookback: int = 8,
    vol_threshold: float = 0.15,
    ann: int = 4,
) -> Tuple[str, dict]:
    if len(sp500_returns) < 3:
        return "normal", {"vol": np.nan, "ret": np.nan}
    recent = (
        sp500_returns.iloc[-lookback:]
        if len(sp500_returns) >= lookback
        else sp500_returns
    )
    vol = recent.std() * np.sqrt(ann)
    ret = recent.mean()
    regime = _classify_regime_cached(vol, ret, vol_threshold)
    logger.info(f"[Current Regime] {regime} (vol={vol:.3f}, ret={ret:.4f})")
    return regime, {"vol": vol, "ret": ret, "lookback": len(recent)}


def get_regime_parameters(regime: str, config: dict) -> dict:
    params = config["regimes"].get(regime, config["regimes"]["normal"])
    logger.info(
        f"[Regime Params] {regime}: top_k={params['top_k']}, max_pos={params['max_pos']}"
    )
    return params
