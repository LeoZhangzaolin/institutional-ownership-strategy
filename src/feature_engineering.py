#!/usr/bin/env python3
"""Feature Engineering Module"""
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
import warnings

warnings.filterwarnings("ignore")
try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        return (lambda f: f) if not args else args[0]


logger = logging.getLogger(__name__)


@jit(nopython=True)
def _fast_winsorize(arr, lower_pct, upper_pct):
    n = len(arr)
    if n == 0:
        return arr
    sorted_arr = np.sort(arr)
    return np.clip(arr, sorted_arr[int(n * lower_pct)], sorted_arr[int(n * upper_pct)])


@jit(nopython=True)
def _fast_robust_zscore(arr):
    if len(arr) == 0:
        return arr
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    denom = 1.4826 * mad if mad > 0 else np.std(arr) if np.std(arr) > 0 else 1.0
    return (arr - med) / denom


def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    arr = s.values.astype(np.float64)
    if NUMBA_AVAILABLE:
        return pd.Series(_fast_winsorize(arr, p, 1 - p), index=s.index)
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def robust_zscore(s: pd.Series) -> pd.Series:
    arr = s.values.astype(np.float64)
    if NUMBA_AVAILABLE:
        return pd.Series(_fast_robust_zscore(arr), index=s.index)
    med = s.median()
    mad = (s - med).abs().median()
    denom = 1.4826 * mad if mad > 0 else (s.std() if s.std() > 0 else 1.0)
    return (s - med) / denom


def xsec_clean(
    df: pd.DataFrame, feature_cols: List[str], by: str = "date_q_end"
) -> pd.DataFrame:
    logger.info(f"[Cleaning] {len(feature_cols)} features...")
    for c in feature_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def _clean(g):
        for c in feature_cols:
            if c in g.columns:
                g[c] = robust_zscore(winsorize_series(g[c]))
        return g

    return df.groupby(by, group_keys=False, sort=False).apply(_clean)


def engineer_features_fast(
    df: pd.DataFrame, base_features: List[str], config: dict
) -> Tuple[pd.DataFrame, List[str]]:
    if not config["features"]["enable_feature_engineering"]:
        return df, base_features
    logger.info("[Feature Engineering] Creating features...")
    df = df.copy()
    new_features = []
    mom_feats = [f for f in base_features if f.startswith("mom_")][:5]
    vol_feats = [f for f in base_features if f.startswith("vol_")][:4]
    io_feats = [f for f in base_features if f.startswith("io_")][:4]
    sk_feats = [f for f in base_features if f.startswith("sk_")][:3]
    max_interactions = config["features"]["max_interaction_features"]
    interaction_count = 0
    if len(mom_feats) > 0 and len(vol_feats) > 0:
        for mom in mom_feats[:4]:
            for vol in vol_feats[:3]:
                if interaction_count >= max_interactions:
                    break
                feat_name = f"inter_{mom}_x_{vol}"
                df[feat_name] = df[mom].values * df[vol].values
                new_features.append(feat_name)
                interaction_count += 1
    if len(mom_feats) > 0 and len(io_feats) > 0:
        for mom in mom_feats[:3]:
            for io in io_feats[:3]:
                if interaction_count >= max_interactions:
                    break
                feat_name = f"inter_{mom}_x_{io}"
                df[feat_name] = df[mom].values * df[io].values
                new_features.append(feat_name)
                interaction_count += 1
    if len(sk_feats) > 0 and len(mom_feats) > 0:
        for sk in sk_feats[:2]:
            for mom in mom_feats[:2]:
                if interaction_count >= max_interactions:
                    break
                feat_name = f"inter_{sk}_x_{mom}"
                df[feat_name] = df[sk].values * df[mom].values
                new_features.append(feat_name)
                interaction_count += 1
    max_nonlinear = config["features"]["max_nonlinear_features"]
    nonlinear_count = 0
    priority_feats = mom_feats + vol_feats + io_feats
    for feat in priority_feats[:max_nonlinear]:
        if nonlinear_count >= max_nonlinear:
            break
        if feat in df.columns:
            if df[feat].min() >= 0:
                feat_name = f"log1p_{feat}"
                df[feat_name] = np.log1p(np.clip(df[feat].values, 0, None))
                new_features.append(feat_name)
                nonlinear_count += 1
            elif "mom_" in feat:
                feat_name = f"sq_{feat}"
                df[feat_name] = df[feat].values ** 2
                new_features.append(feat_name)
                nonlinear_count += 1
    max_ranks = config["features"]["max_rank_features"]
    rank_count = 0
    rank_features = []
    for feat in base_features[:max_ranks]:
        if rank_count >= max_ranks:
            break
        rank_features.append((feat, f"rank_{feat}"))
        rank_count += 1
    if rank_features:

        def add_ranks_fast(g):
            for feat, rank_name in rank_features:
                if feat in g.columns:
                    g[rank_name] = g[feat].rank(pct=True, method="first")
            return g

        df = df.groupby("date_q_end", group_keys=False, sort=False).apply(
            add_ranks_fast
        )
        new_features.extend([rn for _, rn in rank_features])
    logger.info(
        f"[Features] Created {len(new_features)} ({interaction_count} inter, {nonlinear_count} nonlinear, {rank_count} ranks)"
    )
    return df, base_features + new_features


def prepare_features_for_training(
    model_df: pd.DataFrame, config: dict
) -> Tuple[pd.DataFrame, List[str]]:
    logger.info("=" * 80)
    logger.info("FEATURE PREPARATION")
    include_prefixes = tuple(config["features"]["include_prefixes"])
    keys_exclude = {
        "permno",
        "date_q_end",
        "ret_next",
        "ret_next_ex",
        "rf_next",
        "sp500_ret_next",
        "regime",
        "industry",
        "sector",
    }
    num_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
    features_base = [
        c for c in num_cols if c not in keys_exclude and c.startswith(include_prefixes)
    ]
    if not features_base:
        raise ValueError("No base features found")
    logger.info(f"Base features: {len(features_base)}")
    model_df, features = engineer_features_fast(model_df, features_base, config)
    model_df = xsec_clean(model_df, features)
    logger.info(f"[Features] Total: {len(features)} (base: {len(features_base)})")
    logger.info("=" * 80)
    return model_df, features
