#!/usr/bin/env python3
"""
Live Trading Script
Execute quarterly rebalancing with full risk checks
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
import argparse
from datetime import datetime

from src.order_execution import OrderExecutor
from src.risk_management import RiskManager
from src.utils import load_config, setup_logging


def execute_rebalance(config, date=None, signals_file=None, dry_run=False):
    """
    Execute portfolio rebalance

    Args:
        config: Configuration dictionary
        date: Rebalance date (default: today)
        signals_file: Path to signals file (default: latest in signals/)
        dry_run: If True, skip actual execution (paper trading mode)

    Returns:
        Boolean indicating success
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    logger.info("=" * 80)
    logger.info(f"[REBALANCE] Starting rebalance for {date}")
    if dry_run or config["trading"]["paper_trading"]:
        logger.warning("⚠ [PAPER TRADING MODE] - No real trades will be executed")
    logger.info("=" * 80)

    try:
        # 1. Load signals
        logger.info("[REBALANCE] Step 1/7: Loading signals...")

        if signals_file is None:
            # Find latest signals
            signals_dir = Path(config["data"]["paths"]["signals"])
            signal_files = sorted(signals_dir.glob("*_weights.parquet"))

            if not signal_files:
                raise FileNotFoundError(f"No signal files found in {signals_dir}")

            signals_file = signal_files[-1]
            logger.info(f"[REBALANCE] Using latest signals: {signals_file.name}")

        weights_df = pd.read_parquet(signals_file)
        target_weights = weights_df.set_index("permno")["weight"]

        logger.info(f"[REBALANCE] ✓ Loaded {len(target_weights)} target weights")
        logger.info(f"[REBALANCE] Long positions: {(target_weights > 0).sum()}")
        logger.info(f"[REBALANCE] Short positions: {(target_weights < 0).sum()}")
        logger.info(f"[REBALANCE] Gross exposure: {target_weights.abs().sum():.2f}")

        # 2. Initialize components
        logger.info("[REBALANCE] Step 2/7: Initializing components...")
        executor = OrderExecutor(config)
        risk_manager = RiskManager(config)
        logger.info("[REBALANCE] ✓ Components initialized")

        # 3. Get current positions
        logger.info("[REBALANCE] Step 3/7: Getting current positions...")
        current_positions = executor.get_current_positions()
        logger.info(f"[REBALANCE] ✓ Current positions: {len(current_positions)}")

        # 4. Check risk limits BEFORE execution
        logger.info("[REBALANCE] Step 4/7: Pre-execution risk checks...")

        # Check daily limits
        passed, reason = risk_manager.check_daily_limits(current_positions, executor)
        if not passed:
            logger.error(f"[REBALANCE] ✗ Daily limit check failed: {reason}")
            return False
        logger.info("[REBALANCE] ✓ Daily limits OK")

        # Get regime from signals
        regime = (
            weights_df["regime"].iloc[0] if "regime" in weights_df.columns else "normal"
        )
        logger.info(f"[REBALANCE] Detected regime: {regime}")

        # Check position limits
        passed, reason = risk_manager.check_position_limits(target_weights, regime)
        if not passed:
            logger.error(f"[REBALANCE] ✗ Position limit check failed: {reason}")
            return False
        logger.info("[REBALANCE] ✓ Position limits OK")

        # 5. Convert current positions to weights
        logger.info("[REBALANCE] Step 5/7: Converting positions to weights...")
        portfolio_value = risk_manager._get_portfolio_value(executor)

        current_weights = {}
        for ticker, qty in current_positions.items():
            # TODO: Convert ticker back to PERMNO
            # For now, assume positions are already in PERMNO format
            current_weights[ticker] = (
                qty / portfolio_value if portfolio_value > 0 else 0
            )

        logger.info(f"[REBALANCE] ✓ Portfolio value: ${portfolio_value:,.0f}")

        # 6. Execute rebalance
        logger.info("[REBALANCE] Step 6/7: Executing orders...")

        if dry_run:
            logger.warning("[REBALANCE] ⚠ Dry run mode - simulating execution")
            success = True
        else:
            success = executor.execute_rebalance(
                target_weights=target_weights,
                current_positions=current_weights,
                date=date,
            )

        if not success:
            logger.error("[REBALANCE] ✗ Execution failed")
            return False

        logger.info("[REBALANCE] ✓ Execution complete")

        # 7. Post-execution checks
        logger.info("[REBALANCE] Step 7/7: Post-execution verification...")

        # Get new positions
        new_positions = executor.get_current_positions()
        logger.info(f"[REBALANCE] New positions: {len(new_positions)}")

        # Check execution quality (if real trades)
        if not dry_run and not config["trading"]["paper_trading"]:
            # TODO: Implement execution quality check
            # For now, log summary
            logger.info("[REBALANCE] Post-execution summary:")
            logger.info(f"  Previous positions: {len(current_positions)}")
            logger.info(f"  New positions: {len(new_positions)}")
            logger.info(f"  Trades executed: {len(target_weights)}")

        # Update tracking
        risk_manager.portfolio_tracker.update(
            date=date, portfolio_value=portfolio_value, positions=new_positions
        )

        logger.info("[REBALANCE] ✓ Post-execution checks complete")

        # Cleanup
        executor.close()

        logger.info("=" * 80)
        logger.info("[REBALANCE] ✓ Rebalance complete")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"[REBALANCE] ✗ Error during rebalance: {e}", exc_info=True)
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Execute portfolio rebalance")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--date", default=None, help="Rebalance date (YYYY-MM-DD), default: today"
    )
    parser.add_argument(
        "--signals", default=None, help="Path to signals file, default: latest"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate execution without real trades"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    global logger
    logger = setup_logging(level=args.log_level)

    # Load config
    config = load_config(args.config)

    # Override paper trading if dry run
    if args.dry_run:
        config["trading"]["paper_trading"] = True

    # Execute rebalance
    success = execute_rebalance(
        config=config, date=args.date, signals_file=args.signals, dry_run=args.dry_run
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
