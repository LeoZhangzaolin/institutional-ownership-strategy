#!/usr/bin/env python3
"""Portfolio Optimization"""
import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def to_weights_topk(pred: pd.Series, k: int) -> np.ndarray:
    r = pred.rank(method="first")
    n = len(r)
    if n < 2 * k:
        return np.zeros(n)
    w = np.zeros(n)
    w[r >= n - k + 1] = 1.0 / k
    w[r <= k] = -1.0 / k
    return w


def optimize_portfolio_regime_adaptive(
    predictions: pd.Series,
    regime: str,
    config: Dict,
    w_prev: Optional[np.ndarray] = None,
) -> pd.Series:
    logger.info(f"[Portfolio] Optimizing for {regime} regime...")
    params = config["regimes"][regime]
    n = len(predictions)
    k = params["top_k"]
    if n < 2 * k:
        return pd.Series(np.zeros(n), index=predictions.index)
    w0 = to_weights_topk(predictions, k=k)
    if w_prev is None:
        w_prev = np.zeros(n)
    if not config["models"]["enable_optimization"]:
        return pd.Series(w0, index=predictions.index)
    mu = predictions.values
    risk_aversion = params["risk_aversion"]
    turnover_penalty = params["turnover_penalty"]
    max_position = params["max_pos"]

    def objective(w):
        ret = mu @ w
        risk = np.sum(w**2)
        turnover = np.abs(w - w_prev).sum()
        return -(ret - risk_aversion * risk - turnover_penalty * turnover)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w)},
        {"type": "ineq", "fun": lambda w: 2.0 - np.sum(np.abs(w))},
    ]
    bounds = [(-max_position, max_position) for _ in range(n)]
    try:
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-5},
        )
        if result.success:
            logger.info(f"[Portfolio] Optimized: {(result.x != 0).sum()} positions")
            return pd.Series(result.x, index=predictions.index)
        else:
            logger.warning("[Portfolio] Optimization failed, using top-K")
            return pd.Series(w0, index=predictions.index)
    except Exception as e:
        logger.error(f"[Portfolio] Error: {e}")
        return pd.Series(w0, index=predictions.index)


def apply_quality_filters(pred_series: pd.Series, config: Dict) -> pd.Series:
    if not config["models"]["enable_quality_filters"]:
        return pred_series
    min_conf = config["ensemble"]["min_prediction_confidence"]
    pred_z = (pred_series - pred_series.median()) / (
        1.4826 * (pred_series - pred_series.median()).abs().median()
    )
    mask = pred_z.abs() >= min_conf
    filtered = pred_series.copy()
    filtered[~mask] = pred_series.median()
    logger.info(f"[Quality] Filtered {(~mask).sum()} low-confidence predictions")
    return filtered
