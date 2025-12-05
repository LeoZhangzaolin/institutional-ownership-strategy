#!/usr/bin/env python3
"""
Quarterly Update Script
Complete pipeline orchestration for quarterly rebalancing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
import argparse
from datetime import datetime, timedelta

from src.data_pipeline import DataPipeline
from src.feature_engineering import prepare_features_for_training
from src.regime_detection import detect_regime_for_dataframe
from src.model_training import RegimeEnsembleModel
from src.portfolio_optimization import (
    optimize_portfolio_regime_adaptive,
    apply_quality_filters,
)
from src.utils import load_config, setup_logging, save_checkpoint


def update_13f(config, quarter_end):
    """
    Download new 13F quarter from WRDS

    Args:
        config: Configuration dictionary
        quarter_end: Quarter end date (YYYY-MM-DD)

    Returns:
        Boolean indicating success
    """
    logger.info("=" * 80)
    logger.info(f"[13F UPDATE] Downloading quarter: {quarter_end}")
    logger.info("=" * 80)

    try:
        pipeline = DataPipeline(config)

        # Download 13F data
        logger.info("[13F] Downloading from WRDS...")
        pipeline.download_new_13f_quarter(quarter_end)
        logger.info("[13F] ✓ Download complete")

        logger.info("=" * 80)
        logger.info("[13F UPDATE] ✓ Complete")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"[13F UPDATE] ✗ Error: {e}", exc_info=True)
        return False


def update_data(config):
    """
    Rebuild model_df.parquet with latest data

    Args:
        config: Configuration dictionary

    Returns:
        Boolean indicating success
    """
    logger.info("=" * 80)
    logger.info("[DATA UPDATE] Rebuilding model dataframe")
    logger.info("=" * 80)

    try:
        pipeline = DataPipeline(config)

        # Build complete dataframe
        logger.info("[DATA] Building model dataframe...")
        model_df = pipeline.build_model_dataframe()

        # Add regime detection
        logger.info("[DATA] Detecting market regimes...")
        model_df = detect_regime_for_dataframe(model_df, config)

        # Prepare features
        logger.info("[DATA] Engineering features...")
        model_df, features = prepare_features_for_training(
            model_df, config, include_prefixes=config["features"]["include_prefixes"]
        )

        # Save
        cache_dir = Path(config["data"]["paths"]["cache"])
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_file = cache_dir / "model_df.parquet"
        model_df.to_parquet(output_file)
        logger.info(f"[DATA] ✓ Saved: {output_file}")
        logger.info(f"[DATA] Shape: {model_df.shape}")
        logger.info(f"[DATA] Features: {len(features)}")
        logger.info(
            f"[DATA] Date range: {model_df['date_q_end'].min()} to {model_df['date_q_end'].max()}"
        )

        # Save features list
        features_file = cache_dir / "features.txt"
        with open(features_file, "w") as f:
            f.write("\n".join(features))
        logger.info(f"[DATA] ✓ Saved features: {features_file}")

        logger.info("=" * 80)
        logger.info("[DATA UPDATE] ✓ Complete")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"[DATA UPDATE] ✗ Error: {e}", exc_info=True)
        return False


def train_models(config, test_start_date=None):
    """
    Train all models with walkforward validation

    Args:
        config: Configuration dictionary
        test_start_date: Start of test period (YYYY-MM-DD), default: last 3 years

    Returns:
        Boolean indicating success
    """
    logger.info("=" * 80)
    logger.info("[MODEL TRAINING] Starting walkforward training")
    logger.info("=" * 80)

    try:
        # Load data
        cache_dir = Path(config["data"]["paths"]["cache"])
        model_df = pd.read_parquet(cache_dir / "model_df.parquet")

        with open(cache_dir / "features.txt", "r") as f:
            features = [line.strip() for line in f]

        logger.info(f"[TRAIN] Loaded data: {model_df.shape}")
        logger.info(f"[TRAIN] Features: {len(features)}")

        # Determine test quarters
        if test_start_date is None:
            # Default: last 3 years
            latest_date = model_df["date_q_end"].max()
            test_start_date = (
                pd.Timestamp(latest_date) - pd.DateOffset(years=3)
            ).strftime("%Y-%m-%d")

        test_quarters = sorted(
            model_df[model_df["date_q_end"] >= test_start_date]["date_q_end"].unique()
        )
        logger.info(f"[TRAIN] Test period: {test_start_date} onwards")
        logger.info(f"[TRAIN] Test quarters: {len(test_quarters)}")

        # Initialize model
        model = RegimeEnsembleModel(config)

        # Train
        target = (
            "ret_excess_next" if "ret_excess_next" in model_df.columns else "ret_next"
        )
        logger.info(f"[TRAIN] Target: {target}")

        predictions = model.train_all_models(
            df=model_df,
            features=features,
            target=target,
            test_quarters=test_quarters,
            verbose=True,
        )

        # Save model
        models_dir = Path(config["data"]["paths"]["models"])
        models_dir.mkdir(parents=True, exist_ok=True)
        model.save(models_dir)

        # Save predictions
        predictions_file = models_dir / "predictions.parquet"
        predictions.to_parquet(predictions_file)
        logger.info(f"[TRAIN] ✓ Saved predictions: {predictions_file}")

        # Calculate performance metrics
        ensemble_preds = predictions[predictions["model"] == "Ensemble"].copy()
        if len(ensemble_preds) > 0:
            # Group by quarter
            quarterly_perf = ensemble_preds.groupby("date_q_end").apply(
                lambda g: pd.Series(
                    {
                        "ic": g["pred_ex"].corr(g[target]),
                        "mean_pred": g["pred_ex"].mean(),
                        "mean_actual": g[target].mean(),
                    }
                )
            )

            logger.info(f"[TRAIN] Ensemble Performance:")
            logger.info(f"  Mean IC: {quarterly_perf['ic'].mean():.3f}")
            logger.info(
                f"  IC Sharpe: {quarterly_perf['ic'].mean() / quarterly_perf['ic'].std():.2f}"
            )
            logger.info(
                f"  Hit Rate: {(quarterly_perf['ic'] > 0).sum() / len(quarterly_perf):.1%}"
            )

        logger.info("=" * 80)
        logger.info("[MODEL TRAINING] ✓ Complete")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"[MODEL TRAINING] ✗ Error: {e}", exc_info=True)
        return False


def generate_signals(config, quarter_date=None):
    """
    Generate trading signals for latest quarter

    Args:
        config: Configuration dictionary
        quarter_date: Quarter to generate signals for (YYYY-MM-DD), default: latest

    Returns:
        Boolean indicating success
    """
    logger.info("=" * 80)
    logger.info("[SIGNAL GENERATION] Generating trading signals")
    logger.info("=" * 80)

    try:
        # Load model
        models_dir = Path(config["data"]["paths"]["models"])
        model = RegimeEnsembleModel(config)
        model.load(models_dir)
        logger.info("[SIGNALS] ✓ Loaded model")

        # Load data
        cache_dir = Path(config["data"]["paths"]["cache"])
        model_df = pd.read_parquet(cache_dir / "model_df.parquet")

        with open(cache_dir / "features.txt", "r") as f:
            features = [line.strip() for line in f]

        # Determine quarter
        if quarter_date is None:
            quarter_date = model_df["date_q_end"].max()

        logger.info(f"[SIGNALS] Generating for quarter: {quarter_date}")

        # Get quarter data
        quarter_df = model_df[model_df["date_q_end"] == quarter_date].copy()

        if len(quarter_df) == 0:
            raise ValueError(f"No data for quarter {quarter_date}")

        # Get regime
        regime = (
            quarter_df["regime"].iloc[0] if "regime" in quarter_df.columns else "normal"
        )
        logger.info(f"[SIGNALS] Detected regime: {regime}")

        # Generate predictions
        predictions = model.predict_ensemble(
            df=quarter_df, features=features, test_quarter=quarter_date
        )

        # Add predictions to dataframe
        quarter_df["prediction"] = predictions

        # Apply quality filters
        quarter_df["confidence"] = (
            abs(quarter_df["prediction"]) / quarter_df["prediction"].std()
        )
        quarter_df = apply_quality_filters(
            quarter_df,
            pred_col="prediction",
            min_confidence=config["ensemble"]["min_prediction_confidence"],
        )

        # Optimize portfolio
        logger.info("[SIGNALS] Optimizing portfolio...")
        weights = optimize_portfolio_regime_adaptive(
            predictions=quarter_df["prediction"], regime=regime, config=config
        )

        if weights is None or len(weights) == 0:
            raise ValueError("Portfolio optimization failed")

        # Save signals
        signals_dir = Path(config["data"]["paths"]["signals"])
        signals_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights_df = pd.DataFrame(
            {
                "permno": weights.index,
                "weight": weights.values,
                "date": quarter_date,
                "regime": regime,
            }
        )

        weights_file = signals_dir / f"{quarter_date}_weights.parquet"
        weights_df.to_parquet(weights_file)
        logger.info(f"[SIGNALS] ✓ Saved weights: {weights_file}")

        # Save full predictions with metadata
        signals_df = quarter_df[
            ["permno", "date_q_end", "prediction", "confidence"]
        ].copy()
        signals_df["regime"] = regime
        signals_df = signals_df.merge(
            weights_df[["permno", "weight"]], on="permno", how="left"
        )
        signals_df["weight"] = signals_df["weight"].fillna(0)

        signals_file = signals_dir / f"{quarter_date}_signals.parquet"
        signals_df.to_parquet(signals_file)
        logger.info(f"[SIGNALS] ✓ Saved signals: {signals_file}")

        # Summary
        logger.info(f"[SIGNALS] Summary:")
        logger.info(f"  Regime: {regime}")
        logger.info(f"  Universe: {len(quarter_df)} stocks")
        logger.info(
            f"  After filters: {len(signals_df[signals_df['confidence'] >= config['ensemble']['min_prediction_confidence']])} stocks"
        )
        logger.info(f"  Long positions: {(weights > 0).sum()}")
        logger.info(f"  Short positions: {(weights < 0).sum()}")
        logger.info(f"  Gross exposure: {weights.abs().sum():.2f}")
        logger.info(f"  Net exposure: {weights.sum():.2%}")

        logger.info("=" * 80)
        logger.info("[SIGNAL GENERATION] ✓ Complete")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"[SIGNAL GENERATION] ✗ Error: {e}", exc_info=True)
        return False


def full_update(config, quarter_end=None):
    """
    Run complete quarterly update pipeline

    Args:
        config: Configuration dictionary
        quarter_end: Quarter end date for 13F download (optional)

    Returns:
        Boolean indicating success
    """
    logger.info("=" * 80)
    logger.info("[FULL UPDATE] Starting complete quarterly update")
    logger.info("=" * 80)

    steps = []

    # Step 1: Download 13F (if quarter specified)
    if quarter_end:
        logger.info("[FULL] Step 1/4: Download 13F data")
        success = update_13f(config, quarter_end)
        steps.append(("Download 13F", success))
        if not success:
            logger.error("[FULL] ✗ Step 1 failed, aborting")
            return False
    else:
        logger.info("[FULL] Step 1/4: Skipped (no quarter specified)")
        steps.append(("Download 13F", None))

    # Step 2: Rebuild data
    logger.info("[FULL] Step 2/4: Rebuild data")
    success = update_data(config)
    steps.append(("Rebuild data", success))
    if not success:
        logger.error("[FULL] ✗ Step 2 failed, aborting")
        return False

    # Step 3: Train models
    logger.info("[FULL] Step 3/4: Train models")
    success = train_models(config)
    steps.append(("Train models", success))
    if not success:
        logger.error("[FULL] ✗ Step 3 failed, aborting")
        return False

    # Step 4: Generate signals
    logger.info("[FULL] Step 4/4: Generate signals")
    success = generate_signals(config)
    steps.append(("Generate signals", success))
    if not success:
        logger.error("[FULL] ✗ Step 4 failed, aborting")
        return False

    # Summary
    logger.info("=" * 80)
    logger.info("[FULL UPDATE] Pipeline Summary:")
    for step_name, step_success in steps:
        if step_success is None:
            status = "SKIPPED"
        elif step_success:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        logger.info(f"  {step_name}: {status}")
    logger.info("=" * 80)
    logger.info("[FULL UPDATE] ✓ Complete")
    logger.info("=" * 80)

    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Quarterly update pipeline")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # update-13f command
    parser_13f = subparsers.add_parser("update-13f", help="Download new 13F quarter")
    parser_13f.add_argument(
        "--quarter", required=True, help="Quarter end date (YYYY-MM-DD)"
    )

    # update-data command
    parser_data = subparsers.add_parser("update-data", help="Rebuild model dataframe")

    # train-models command
    parser_train = subparsers.add_parser("train-models", help="Train models")
    parser_train.add_argument(
        "--test-start", default=None, help="Start of test period (YYYY-MM-DD)"
    )

    # generate-signals command
    parser_signals = subparsers.add_parser(
        "generate-signals", help="Generate trading signals"
    )
    parser_signals.add_argument(
        "--quarter", default=None, help="Quarter date (YYYY-MM-DD), default: latest"
    )

    # full-update command
    parser_full = subparsers.add_parser("full-update", help="Run complete pipeline")
    parser_full.add_argument(
        "--quarter", default=None, help="Quarter end date for 13F download (YYYY-MM-DD)"
    )

    # Common arguments
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    global logger
    logger = setup_logging(level=args.log_level)

    # Load config
    config = load_config(args.config)

    # Run command
    if args.command == "update-13f":
        success = update_13f(config, args.quarter)
    elif args.command == "update-data":
        success = update_data(config)
    elif args.command == "train-models":
        success = train_models(config, args.test_start)
    elif args.command == "generate-signals":
        success = generate_signals(config, args.quarter)
    elif args.command == "full-update":
        success = full_update(config, args.quarter)
    else:
        logger.error(f"Unknown command: {args.command}")
        success = False

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
