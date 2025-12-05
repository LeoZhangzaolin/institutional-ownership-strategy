#!/usr/bin/env python3
"""
Model Training Module
Implements regime-adaptive ensemble with walkforward validation

"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from functools import lru_cache
import time
import gc
import warnings

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import BaseCrossValidator, HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import loguniform, uniform
from joblib import Parallel, delayed, parallel_backend

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    XGBRegressor = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class PurgedExpandingSplit(BaseCrossValidator):
    """
    Custom CV splitter with embargo and expanding window
    Prevents look-ahead bias in time series
    """

    def __init__(self, n_splits=2, min_train_groups=12, embargo=1, val_groups=1):
        self.n_splits = n_splits
        self.min_train_groups = min_train_groups
        self.embargo = embargo
        self.val_groups = val_groups

    def _candidate_anchors(self, U):
        start = self.min_train_groups + self.embargo
        end = len(U) - self.val_groups
        if end < start:
            return np.array([], dtype=int)
        n_anchors = min(self.n_splits, max(1, end - start + 1))
        return np.linspace(start, end, num=n_anchors, dtype=int)

    def get_n_splits(self, X=None, y=None, groups=None):
        if groups is None:
            return self.n_splits
        groups = np.asarray(groups)
        U = np.array(sorted(np.unique(groups)))
        anchors = self._candidate_anchors(U)
        return len(anchors)

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("PurgedExpandingSplit requires `groups`.")

        groups = np.asarray(groups)
        U = np.array(sorted(np.unique(groups)))
        anchors = self._candidate_anchors(U)
        idx = np.arange(len(groups))

        for a in anchors:
            val_mask = np.isin(groups, U[a : a + self.val_groups])
            train_cut = a - self.embargo
            train_mask = np.isin(groups, U[:train_cut])

            if not train_mask.any() or not val_mask.any():
                continue

            yield idx[train_mask], idx[val_mask]


class ModelVarianceTracker:
    """
    Track prediction errors and compute inverse-variance weights
    Better than Sharpe-based weighting for ensemble
    """

    def __init__(self, lookback_q=8):
        self.lookback_q = lookback_q
        self.errors = {}  # {model_key: [(date, sq_error), ...]}

    def add_error(self, model_key, date, squared_error):
        """Add prediction error for a model"""
        if model_key not in self.errors:
            self.errors[model_key] = []
        self.errors[model_key].append((date, squared_error))

    def get_variance(self, model_key, current_date):
        """Get variance (MSE) over lookback window"""
        if model_key not in self.errors or not self.errors[model_key]:
            return 1.0  # Default variance

        # Get errors in lookback window
        recent_errors = [
            err
            for dt, err in self.errors[model_key]
            if dt < current_date  # Only past data
        ]

        if not recent_errors:
            return 1.0

        # Keep only last N quarters
        recent_errors = recent_errors[-self.lookback_q :]

        # Return MSE
        return np.mean(recent_errors)

    def get_precision_weight(self, model_key, current_date):
        """Get inverse-variance weight (precision)"""
        variance = self.get_variance(model_key, current_date)
        return 1.0 / max(variance, 1e-6)  # Avoid division by zero

    def get_ensemble_weights(self, model_keys, current_date):
        """Get normalized ensemble weights for all models"""
        weights = {}

        for key in model_keys:
            weights[key] = self.get_precision_weight(key, current_date)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Equal weights fallback
            weights = {k: 1.0 / len(model_keys) for k in model_keys}

        return weights

    def save(self, path):
        """Save tracker state"""
        with open(path, "wb") as f:
            pickle.dump(self.errors, f)

    def load(self, path):
        """Load tracker state"""
        with open(path, "rb") as f:
            self.errors = pickle.load(f)


class RegimeEnsembleModel:
    """
    Main model class: regime-adaptive ensemble with walkforward training
    """

    def __init__(self, config):
        self.config = config
        self.models = {}  # {regime: {model_name: fitted_model}}
        self.variance_tracker = ModelVarianceTracker(
            lookback_q=config["ensemble"]["var_lookback_q"]
        )
        self.param_cache = {}  # Cache hyperparameters
        self.coef_history = {}  # Track coefficients over time

        # Config
        self.random_seed = config["models"]["random_seed"]
        self.n_splits = config["models"]["n_cv_splits"]
        self.embargo_q = config["models"]["embargo_quarters"]
        self.min_train_q = config["models"]["min_train_quarters"]
        self.max_train_q = config["models"]["max_train_quarters"]
        self.tune_every_q = config["models"]["tune_every_quarters"]
        self.enable_regime = config["models"]["enable_regime_models"]
        self.enable_ensemble = config["ensemble"]["enable_ensemble"]

        np.random.seed(self.random_seed)

    def get_base_models(self):
        """Get base model specifications"""
        cv_inner = PurgedExpandingSplit(
            n_splits=self.n_splits,
            min_train_groups=self.min_train_q,
            embargo=self.embargo_q,
        )

        models = {
            "ElasticNet": HalvingRandomSearchCV(
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "reg",
                            ElasticNet(max_iter=5000, random_state=self.random_seed),
                        ),
                    ]
                ),
                param_distributions={
                    "reg__alpha": loguniform(1e-6, 1e1),
                    "reg__l1_ratio": uniform(0, 1),
                },
                n_candidates=24,
                factor=3,
                resource="n_samples",
                max_resources="auto",
                min_resources="exhaust",
                aggressive_elimination=True,
                cv=cv_inner,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                refit=True,
                random_state=self.random_seed,
                verbose=0,
                error_score=np.nan,
            ),
            "RandomForest": Pipeline(
                [
                    (
                        "reg",
                        RandomForestRegressor(
                            n_estimators=300,
                            max_depth=6,
                            min_samples_leaf=5,
                            max_features="sqrt",
                            n_jobs=-1,
                            random_state=self.random_seed,
                        ),
                    )
                ]
            ),
        }

        # Add XGBoost if available
        if XGB_AVAILABLE and self.config["models"]["enable_xgboost"]:
            xgb_params = dict(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=self.random_seed,
                tree_method="hist",
            )
            models["XGBoost"] = Pipeline([("reg", XGBRegressor(**xgb_params))])

        return models

    def _limit_to_last_quarters(self, df, q_col, max_q):
        """Limit training data to last N quarters"""
        if not max_q or max_q <= 0:
            return df

        unique_quarters = np.sort(df[q_col].unique())
        if len(unique_quarters) <= max_q:
            return df

        keep_quarters = set(unique_quarters[-max_q:])
        return df[df[q_col].isin(keep_quarters)]

    def _train_single_model(
        self,
        model_key,
        name,
        estimator,
        X_train,
        y_train,
        groups,
        X_test,
        param_cache,
        iteration,
        retune,
    ):
        """Train a single model (used in parallel)"""
        try:
            # Use cached params if not retuning
            if not retune and model_key in param_cache:
                est_copy = estimator
                if hasattr(est_copy, "set_params"):
                    est_copy.set_params(**param_cache[model_key])
                est_copy.fit(X_train, y_train)
            else:
                # Full hyperparameter tuning
                if hasattr(estimator, "cv"):
                    # HalvingRandomSearchCV
                    estimator.fit(X_train, y_train, groups=groups)
                    est_copy = estimator.best_estimator_
                    params = estimator.best_params_
                else:
                    # Fixed hyperparameters (e.g., RandomForest)
                    estimator.fit(X_train, y_train)
                    est_copy = estimator
                    params = None

            # Make predictions
            pred = est_copy.predict(X_test)

            # Extract coefficients if linear model
            coef = None
            if hasattr(est_copy, "named_steps") and "reg" in est_copy.named_steps:
                reg = est_copy.named_steps["reg"]
                if hasattr(reg, "coef_"):
                    coef = reg.coef_

            return {
                "model_key": model_key,
                "name": name,
                "pred": pred,
                "coef": coef,
                "params": params if retune else None,
            }

        except Exception as e:
            logger.warning(f"[Model] {model_key} failed: {e}")
            return None

    def train_all_models(self, df, features, target, test_quarters, verbose=True):
        """
        Walkforward training and prediction for all regimes and models

        Args:
            df: Full dataframe with features, target, regime
            features: List of feature column names
            target: Target column name (e.g., 'ret_excess_next')
            test_quarters: List of quarters to predict
            verbose: Print progress

        Returns:
            DataFrame with predictions for all quarters
        """
        logger.info("=" * 80)
        logger.info("[Training] Starting walkforward training")
        logger.info(f"  Quarters: {len(test_quarters)}")
        logger.info(f"  Features: {len(features)}")
        logger.info(f"  Regimes: {self.enable_regime}")
        logger.info("=" * 80)

        all_predictions = []
        t0_start = time.time()

        # Determine which column has the actual returns
        predict_excess = "rf_next" in df.columns
        if predict_excess:
            y_col = target
            logger.info("[Training] Predicting excess returns")
        else:
            y_col = target
            logger.info("[Training] Predicting total returns")

        # Get base models for each regime
        if self.enable_regime:
            regimes = ["bull", "bear", "normal", "high_vol"]
        else:
            regimes = ["ALL"]

        # Initialize models for each regime
        for regime in regimes:
            self.models[regime] = self.get_base_models()

        # Walkforward loop
        for i, test_quarter in enumerate(test_quarters, 1):
            t0 = time.time()

            # Split data
            train_all = df[df["date_q_end"] < test_quarter]

            # Apply embargo
            cutoff = pd.Timestamp(test_quarter) - pd.DateOffset(
                months=3 * self.embargo_q
            )
            train = self._limit_to_last_quarters(
                train_all[train_all["date_q_end"] <= cutoff],
                "date_q_end",
                self.max_train_q,
            )

            test = df[df["date_q_end"] == test_quarter]

            if verbose:
                logger.info(
                    f"  [{i:>2}/{len(test_quarters)}] {pd.Timestamp(test_quarter).date()} | "
                    f"train={len(train):,} test={len(test):,}"
                )

            # Skip if insufficient data
            if len(train) < 50 or test.empty:
                logger.warning(f"    → Skipped (insufficient data)")
                continue

            # Determine test regime
            if self.enable_regime and "regime" in test.columns:
                test_regime = test["regime"].iloc[0]
            else:
                test_regime = "ALL"

            # Get models for this regime
            if test_regime not in self.models:
                test_regime = "normal"  # Fallback

            regime_models = self.models[test_regime]

            # Prepare data
            X_train = train[features].astype(np.float32).values
            y_train = train[y_col].astype(np.float32).values
            X_test = test[features].astype(np.float32).values
            groups = train["date_q_end"].values

            # Shuffle training data (but keep groups aligned)
            rng = np.random.RandomState(self.random_seed + i)
            perm = rng.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]
            groups = groups[perm]

            # Determine if we should retune
            retune = (
                (not self.config["models"].get("reuse_best_params", True))
                or (i == 1)
                or ((i - 1) % self.tune_every_q == 0)
            )

            # Train all models in parallel
            model_tasks = []
            for name, estimator in regime_models.items():
                model_key = f"{test_regime}_{name}"
                model_tasks.append(
                    (
                        model_key,
                        name,
                        estimator,
                        X_train,
                        y_train,
                        groups,
                        X_test,
                        self.param_cache,
                        i,
                        retune,
                    )
                )

            # Parallel training
            n_jobs = min(
                len(model_tasks), self.config["models"].get("n_parallel_models", 4)
            )
            with parallel_backend("loky", n_jobs=n_jobs):
                results = Parallel()(
                    delayed(self._train_single_model)(*task) for task in model_tasks
                )

            # Collect predictions
            ensemble_preds = []
            model_keys = []

            for result in results:
                if result is None:
                    continue

                model_key = result["model_key"]
                name = result["name"]
                pred = result["pred"]
                coef = result["coef"]
                params = result["params"]

                # Cache params
                if params is not None:
                    self.param_cache[model_key] = params

                # Store coefficients
                if coef is not None:
                    if model_key not in self.coef_history:
                        self.coef_history[model_key] = []
                    self.coef_history[model_key].append(
                        pd.Series(coef, index=features, name=test_quarter)
                    )

                # Track prediction error
                if predict_excess:
                    actual = test[target].values - test["rf_next"].fillna(0.0).values
                else:
                    actual = test[target].values

                sq_errors = (pred - actual) ** 2
                avg_sq_error = np.mean(sq_errors)
                self.variance_tracker.add_error(model_key, test_quarter, avg_sq_error)

                # Save individual prediction
                pred_df = test[["permno", "date_q_end", target]].copy()
                pred_df["model"] = name
                pred_df["regime"] = test_regime

                if predict_excess:
                    pred_df["pred_ex"] = pred
                    pred_df["pred"] = pred + test["rf_next"].fillna(0.0).values
                else:
                    pred_df["pred"] = pred
                    pred_df["pred_ex"] = pred - test["rf_next"].fillna(0.0).values

                all_predictions.append(pred_df)

                ensemble_preds.append((model_key, pred))
                model_keys.append(model_key)

            # Create ensemble prediction
            if self.enable_ensemble and len(ensemble_preds) > 0:
                # Get variance weights
                weights = self.variance_tracker.get_ensemble_weights(
                    model_keys, test_quarter
                )

                # Weighted average
                ensemble_pred = np.zeros(len(test))
                for model_key, pred in ensemble_preds:
                    w = weights.get(model_key, 1.0 / len(ensemble_preds))
                    ensemble_pred += w * pred

                # Save ensemble prediction
                ensemble_df = test[["permno", "date_q_end", target]].copy()
                ensemble_df["model"] = "Ensemble"
                ensemble_df["regime"] = test_regime

                if predict_excess:
                    ensemble_df["pred_ex"] = ensemble_pred
                    ensemble_df["pred"] = (
                        ensemble_pred + test["rf_next"].fillna(0.0).values
                    )
                else:
                    ensemble_df["pred"] = ensemble_pred
                    ensemble_df["pred_ex"] = (
                        ensemble_pred - test["rf_next"].fillna(0.0).values
                    )

                all_predictions.append(ensemble_df)

            if verbose:
                logger.info(f"    → {len(results)} models, {time.time()-t0:.1f}s")

            # Memory cleanup
            del train_all, train, test, X_train, y_train, X_test
            gc.collect()

        # Combine all predictions
        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
        else:
            predictions_df = pd.DataFrame()

        elapsed = time.time() - t0_start
        logger.info("=" * 80)
        logger.info(f"[Training] ✓ Complete in {elapsed/60:.1f} minutes")
        logger.info(f"  Total predictions: {len(predictions_df):,}")
        logger.info("=" * 80)

        return predictions_df

    def predict_ensemble(self, df, features, test_quarter):
        """
        Generate ensemble prediction for a single quarter

        Args:
            df: Dataframe with features for the quarter
            features: List of feature names
            test_quarter: Quarter to predict

        Returns:
            Array of predictions
        """
        if not self.enable_ensemble:
            raise ValueError("Ensemble not enabled")

        # Determine regime
        if self.enable_regime and "regime" in df.columns:
            regime = df["regime"].iloc[0]
        else:
            regime = "ALL"

        if regime not in self.models:
            regime = "normal"

        # Get models
        regime_models = self.models[regime]

        # Prepare data
        X = df[features].astype(np.float32).values

        # Get predictions from all models
        predictions = []
        model_keys = []

        for name, model in regime_models.items():
            model_key = f"{regime}_{name}"
            pred = model.predict(X)
            predictions.append(pred)
            model_keys.append(model_key)

        # Get variance weights
        weights = self.variance_tracker.get_ensemble_weights(model_keys, test_quarter)

        # Weighted average
        ensemble_pred = np.zeros(len(df))
        for i, (pred, model_key) in enumerate(zip(predictions, model_keys)):
            w = weights.get(model_key, 1.0 / len(predictions))
            ensemble_pred += w * pred

        return ensemble_pred

    def save(self, path):
        """Save model state"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save models
        with open(path / "models.pkl", "wb") as f:
            pickle.dump(self.models, f)

        # Save variance tracker
        self.variance_tracker.save(path / "variance_tracker.pkl")

        # Save param cache
        with open(path / "param_cache.pkl", "wb") as f:
            pickle.dump(self.param_cache, f)

        logger.info(f"[Model] ✓ Saved to {path}")

    def load(self, path):
        """Load model state"""
        path = Path(path)

        # Load models
        with open(path / "models.pkl", "rb") as f:
            self.models = pickle.load(f)

        # Load variance tracker
        self.variance_tracker.load(path / "variance_tracker.pkl")

        # Load param cache
        with open(path / "param_cache.pkl", "rb") as f:
            self.param_cache = pickle.load(f)

        logger.info(f"[Model] ✓ Loaded from {path}")
