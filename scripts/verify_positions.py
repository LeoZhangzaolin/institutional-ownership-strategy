#!/usr/bin/env python3
"""
Position Verification Script
Reconcile internal records with broker positions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
import argparse
from datetime import datetime

from src.order_execution import OrderExecutor
from src.utils import load_config, setup_logging


def verify_positions(config, date=None, signals_file=None):
    """
    Verify positions match expected targets

    Args:
        config: Configuration dictionary
        date: Verification date (default: today)
        signals_file: Path to signals file to compare against

    Returns:
        Dictionary with verification results
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    logger.info("=" * 80)
    logger.info(f"[VERIFY] Position verification for {date}")
    logger.info("=" * 80)

    results = {"date": date, "status": "unknown", "discrepancies": [], "summary": {}}

    try:
        # 1. Get broker positions
        logger.info("[VERIFY] Step 1/3: Getting broker positions...")
        executor = OrderExecutor(config)
        broker_positions = executor.get_current_positions()

        logger.info(f"[VERIFY] ✓ Broker positions: {len(broker_positions)}")
        results["summary"]["broker_count"] = len(broker_positions)

        # 2. Load expected positions (if signals file provided)
        if signals_file:
            logger.info("[VERIFY] Step 2/3: Loading expected positions...")
            expected_df = pd.read_parquet(signals_file)
            expected_positions = expected_df.set_index("permno")["weight"].to_dict()
            logger.info(f"[VERIFY] ✓ Expected positions: {len(expected_positions)}")
            results["summary"]["expected_count"] = len(expected_positions)
        else:
            logger.info(
                "[VERIFY] Step 2/3: No expected positions file (verifying consistency only)"
            )
            expected_positions = {}
            results["summary"]["expected_count"] = 0

        # 3. Compare positions
        logger.info("[VERIFY] Step 3/3: Comparing positions...")

        # Find missing positions (in expected but not in broker)
        missing_positions = []
        if expected_positions:
            for permno, weight in expected_positions.items():
                if abs(weight) > 0.001:  # Minimum threshold
                    # TODO: Convert PERMNO to ticker for comparison
                    # For now, assume broker uses same identifiers
                    if permno not in broker_positions:
                        missing_positions.append(
                            {
                                "permno": permno,
                                "type": "missing",
                                "expected_weight": weight,
                                "broker_weight": 0,
                            }
                        )

        # Find extra positions (in broker but not in expected)
        extra_positions = []
        if expected_positions:
            for identifier, qty in broker_positions.items():
                if identifier not in expected_positions:
                    extra_positions.append(
                        {
                            "identifier": identifier,
                            "type": "extra",
                            "expected_weight": 0,
                            "broker_qty": qty,
                        }
                    )

        # Find size mismatches (in both but different sizes)
        size_mismatches = []
        if expected_positions:
            for permno, expected_weight in expected_positions.items():
                if permno in broker_positions:
                    broker_qty = broker_positions[permno]
                    # TODO: Convert qty to weight for proper comparison
                    # For now, compare directly
                    difference = abs(expected_weight - broker_qty)
                    if difference > 0.01:  # 1% threshold
                        size_mismatches.append(
                            {
                                "permno": permno,
                                "type": "mismatch",
                                "expected_weight": expected_weight,
                                "broker_qty": broker_qty,
                                "difference": difference,
                            }
                        )

        # Compile discrepancies
        all_discrepancies = missing_positions + extra_positions + size_mismatches
        results["discrepancies"] = all_discrepancies

        # Determine status
        if len(all_discrepancies) == 0:
            results["status"] = "OK"
            logger.info("[VERIFY] ✓ All positions verified - no discrepancies")
        else:
            results["status"] = "DISCREPANCIES"
            logger.warning(f"[VERIFY] ⚠ Found {len(all_discrepancies)} discrepancies")

        # Log summary
        results["summary"]["missing_count"] = len(missing_positions)
        results["summary"]["extra_count"] = len(extra_positions)
        results["summary"]["mismatch_count"] = len(size_mismatches)
        results["summary"]["total_discrepancies"] = len(all_discrepancies)

        logger.info(f"[VERIFY] Summary:")
        logger.info(f"  Broker positions: {results['summary']['broker_count']}")
        if expected_positions:
            logger.info(f"  Expected positions: {results['summary']['expected_count']}")
            logger.info(f"  Missing positions: {results['summary']['missing_count']}")
            logger.info(f"  Extra positions: {results['summary']['extra_count']}")
            logger.info(f"  Size mismatches: {results['summary']['mismatch_count']}")

        # Log details if discrepancies found
        if all_discrepancies:
            logger.warning("[VERIFY] Discrepancy details:")
            for i, disc in enumerate(all_discrepancies[:10], 1):  # First 10
                if disc["type"] == "missing":
                    logger.warning(
                        f"  {i}. MISSING: PERMNO {disc['permno']} "
                        f"(expected: {disc['expected_weight']:.2%})"
                    )
                elif disc["type"] == "extra":
                    logger.warning(
                        f"  {i}. EXTRA: {disc['identifier']} "
                        f"(qty: {disc['broker_qty']})"
                    )
                elif disc["type"] == "mismatch":
                    logger.warning(
                        f"  {i}. MISMATCH: PERMNO {disc['permno']} "
                        f"(expected: {disc['expected_weight']:.2%}, "
                        f"actual: {disc['broker_qty']}, "
                        f"diff: {disc['difference']:.2%})"
                    )

            if len(all_discrepancies) > 10:
                logger.warning(f"  ... and {len(all_discrepancies) - 10} more")

        # Save verification report
        save_verification_report(config, results)

        # Cleanup
        executor.close()

        logger.info("=" * 80)
        logger.info(f"[VERIFY] ✓ Verification complete - Status: {results['status']}")
        logger.info("=" * 80)

        return results

    except Exception as e:
        logger.error(f"[VERIFY] ✗ Error during verification: {e}", exc_info=True)
        results["status"] = "ERROR"
        results["error"] = str(e)
        return results


def save_verification_report(config, results):
    """
    Save verification report to file

    Args:
        config: Configuration dictionary
        results: Verification results dictionary
    """
    reports_dir = Path(config["data"]["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_file = reports_dir / f"position_verification_{results['date']}.txt"

    report = f"""
{'='*80}
POSITION VERIFICATION REPORT
Date: {results['date']}
Status: {results['status']}
{'='*80}

SUMMARY
-------
Broker Positions: {results['summary']['broker_count']}
Expected Positions: {results['summary']['expected_count']}
Total Discrepancies: {results['summary']['total_discrepancies']}
  - Missing: {results['summary']['missing_count']}
  - Extra: {results['summary']['extra_count']}
  - Mismatches: {results['summary']['mismatch_count']}

"""

    if results["discrepancies"]:
        report += "DISCREPANCIES\n"
        report += "-------------\n"
        for i, disc in enumerate(results["discrepancies"], 1):
            if disc["type"] == "missing":
                report += f"{i}. MISSING: PERMNO {disc['permno']} (expected: {disc['expected_weight']:.2%})\n"
            elif disc["type"] == "extra":
                report += (
                    f"{i}. EXTRA: {disc['identifier']} (qty: {disc['broker_qty']})\n"
                )
            elif disc["type"] == "mismatch":
                report += f"{i}. MISMATCH: PERMNO {disc['permno']} (expected: {disc['expected_weight']:.2%}, actual: {disc['broker_qty']}, diff: {disc['difference']:.2%})\n"
        report += "\n"
    else:
        report += "No discrepancies found - all positions verified ✓\n\n"

    report += f"{'='*80}\n"

    with open(report_file, "w") as f:
        f.write(report)

    logger.info(f"[Report] ✓ Verification report saved to {report_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Verify portfolio positions")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--date", default=None, help="Verification date (YYYY-MM-DD), default: today"
    )
    parser.add_argument(
        "--signals", default=None, help="Path to signals file to compare against"
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

    # Run verification
    results = verify_positions(config=config, date=args.date, signals_file=args.signals)

    # Exit with appropriate code
    if results["status"] == "OK":
        sys.exit(0)  # All good
    elif results["status"] == "DISCREPANCIES":
        sys.exit(1)  # Discrepancies found
    else:
        sys.exit(2)  # Error


if __name__ == "__main__":
    main()
