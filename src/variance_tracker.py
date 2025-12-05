#!/usr/bin/env python3
"""Variance Tracker"""
import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ModelVarianceTracker:
    def __init__(self, lookback_quarters: int = 8):
        self.lookback = lookback_quarters
        self.error_history = {}
        logger.info(f"[VarianceTracker] Init with {lookback_quarters}Q lookback")

    def add_error(self, model_key: str, date: pd.Timestamp, squared_error: float):
        if model_key not in self.error_history:
            self.error_history[model_key] = []
        self.error_history[model_key].append((date, squared_error))
        if len(self.error_history[model_key]) > self.lookback * 2:
            self.error_history[model_key] = self.error_history[model_key][
                -self.lookback * 2 :
            ]

    def get_variance(self, model_key: str, current_date: pd.Timestamp) -> float:
        if model_key not in self.error_history:
            return np.nan
        history = self.error_history[model_key]
        if len(history) < 3:
            return np.nan
        recent = [err for date, err in history if date < current_date][-self.lookback :]
        if len(recent) < 3:
            return np.nan
        return np.mean(recent) if np.mean(recent) > 0 else np.nan

    def get_precision_weight(self, model_key: str, current_date: pd.Timestamp) -> float:
        variance = self.get_variance(model_key, current_date)
        if np.isnan(variance) or variance <= 0:
            return 1.0
        return 1.0 / variance

    def get_ensemble_weights(
        self, model_keys: List[str], current_date: pd.Timestamp
    ) -> Dict[str, float]:
        precisions = {k: self.get_precision_weight(k, current_date) for k in model_keys}
        if all(p == 1.0 for p in precisions.values()):
            return {k: 1.0 / len(model_keys) for k in model_keys}
        total = sum(precisions.values())
        if total == 0:
            return {k: 1.0 / len(model_keys) for k in model_keys}
        weights = {k: v / total for k, v in precisions.items()}
        logger.info(f"[VarianceTracker] Weights: {weights}")
        return weights

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
