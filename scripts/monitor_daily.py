#!/usr/bin/env python3
"""
Daily Monitoring Script
Monitor positions, P&L, and risk metrics daily
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import argparse

from src.order_execution import OrderExecutor
from src.risk_management import RiskManager, PortfolioTracker
from src.utils import load_config, setup_logging


def monitor_daily(config, date=None, send_alerts=True):
    """
    Run daily monitoring checks

    Args:
        config: Configuration dictionary
        date: Date to monitor (default: today)
        send_alerts: Whether to send email/Slack alerts

    Returns:
        Dictionary with monitoring results
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    logger.info("=" * 80)
    logger.info(f"[MONITOR] Daily monitoring for {date}")
    logger.info("=" * 80)

    results = {"date": date, "status": "unknown", "alerts": [], "metrics": {}}

    try:
        # Initialize components
        logger.info("[MONITOR] Initializing components...")
        executor = OrderExecutor(config)
        risk_manager = RiskManager(config)
        portfolio_tracker = risk_manager.portfolio_tracker

        # 1. Get current positions from broker
        logger.info("[MONITOR] Step 1/5: Getting current positions...")
        current_positions = executor.get_current_positions()

        if not current_positions:
            logger.warning("[MONITOR] ⚠ No positions found")
            results["alerts"].append("No positions in portfolio")
        else:
            logger.info(f"[MONITOR] ✓ Found {len(current_positions)} positions")
            results["metrics"]["num_positions"] = len(current_positions)

        # 2. Get portfolio value from broker
        logger.info("[MONITOR] Step 2/5: Getting portfolio value...")
        portfolio_value = risk_manager._get_portfolio_value(executor)
        logger.info(f"[MONITOR] ✓ Portfolio value: ${portfolio_value:,.0f}")
        results["metrics"]["portfolio_value"] = portfolio_value

        # 3. Calculate daily P&L
        logger.info("[MONITOR] Step 3/5: Calculating daily P&L...")
        daily_pnl, daily_pnl_pct = risk_manager._calculate_daily_pnl(
            executor, portfolio_value
        )
        logger.info(f"[MONITOR] ✓ Daily P&L: ${daily_pnl:,.0f} ({daily_pnl_pct:.2%})")
        results["metrics"]["daily_pnl"] = daily_pnl
        results["metrics"]["daily_pnl_pct"] = daily_pnl_pct

        # 4. Check risk limits
        logger.info("[MONITOR] Step 4/5: Checking risk limits...")
        passed, reason = risk_manager.check_daily_limits(current_positions, executor)

        if not passed:
            logger.error(f"[MONITOR] ✗ Risk limit breach: {reason}")
            results["status"] = "BREACH"
            results["alerts"].append(f"CRITICAL: {reason}")
        else:
            logger.info(f"[MONITOR] ✓ All risk limits passed")
            results["status"] = "OK"

        # 5. Update portfolio tracking
        logger.info("[MONITOR] Step 5/5: Updating portfolio tracking...")
        portfolio_tracker.update(
            date=date,
            portfolio_value=portfolio_value,
            positions=current_positions,
            daily_pnl=daily_pnl,
        )
        logger.info("[MONITOR] ✓ Portfolio tracking updated")

        # 6. Calculate performance metrics
        logger.info("[MONITOR] Calculating performance metrics...")
        perf_metrics = portfolio_tracker.get_performance_metrics(lookback_days=252)
        results["metrics"].update(perf_metrics)

        logger.info(f"[MONITOR] Performance (1Y):")
        logger.info(f"  Total Return: {perf_metrics['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {perf_metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {perf_metrics['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {perf_metrics['win_rate']:.1%}")

        # 7. Check for warnings
        if daily_pnl_pct < -0.01:  # -1% or worse
            warning = f"Significant daily loss: {daily_pnl_pct:.2%}"
            logger.warning(f"[MONITOR] ⚠ {warning}")
            results["alerts"].append(warning)

        if perf_metrics["sharpe_ratio"] < 1.0 and perf_metrics["num_days"] > 60:
            warning = f"Sharpe ratio below 1.0: {perf_metrics['sharpe_ratio']:.2f}"
            logger.warning(f"[MONITOR] ⚠ {warning}")
            results["alerts"].append(warning)

        # 8. Send alerts if configured
        if send_alerts and results["alerts"]:
            send_monitoring_alerts(config, results)

        # 9. Generate daily report
        generate_daily_report(config, results, portfolio_tracker)

        # Cleanup
        executor.close()

        logger.info("=" * 80)
        logger.info(
            f"[MONITOR] ✓ Daily monitoring complete - Status: {results['status']}"
        )
        logger.info("=" * 80)

        return results

    except Exception as e:
        logger.error(f"[MONITOR] ✗ Error during monitoring: {e}", exc_info=True)
        results["status"] = "ERROR"
        results["alerts"].append(f"Monitoring error: {str(e)}")
        return results


def send_monitoring_alerts(config, results):
    """
    Send email or Slack alerts for monitoring issues

    Args:
        config: Configuration dictionary
        results: Monitoring results
    """
    if not config.get("alerts", {}).get("enabled", False):
        logger.info("[Alerts] Alerts disabled in config")
        return

    alert_config = config["alerts"]

    # Build alert message
    message = f"""
Daily Monitoring Alert - {results['date']}
Status: {results['status']}

Alerts:
{chr(10).join(f"  • {alert}" for alert in results['alerts'])}

Metrics:
  Portfolio Value: ${results['metrics'].get('portfolio_value', 0):,.0f}
  Daily P&L: ${results['metrics'].get('daily_pnl', 0):,.0f} ({results['metrics'].get('daily_pnl_pct', 0):.2%})
  Sharpe Ratio: {results['metrics'].get('sharpe_ratio', 0):.2f}
  Max Drawdown: {results['metrics'].get('max_drawdown', 0):.2%}
"""

    # Send email
    if alert_config.get("email", {}).get("enabled", False):
        try:
            send_email_alert(alert_config["email"], message)
            logger.info("[Alerts] ✓ Email alert sent")
        except Exception as e:
            logger.error(f"[Alerts] ✗ Email failed: {e}")

    # Send Slack
    if alert_config.get("slack", {}).get("enabled", False):
        try:
            send_slack_alert(alert_config["slack"], message)
            logger.info("[Alerts] ✓ Slack alert sent")
        except Exception as e:
            logger.error(f"[Alerts] ✗ Slack failed: {e}")


def send_email_alert(email_config, message):
    """Send email alert"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart()
    msg["From"] = email_config["from_address"]
    msg["To"] = ", ".join(email_config["to_addresses"])
    msg["Subject"] = "Trading System Alert"

    msg.attach(MIMEText(message, "plain"))

    with smtplib.SMTP(email_config["smtp_host"], email_config["smtp_port"]) as server:
        if email_config.get("use_tls", True):
            server.starttls()
        if email_config.get("username") and email_config.get("password"):
            server.login(email_config["username"], email_config["password"])
        server.send_message(msg)


def send_slack_alert(slack_config, message):
    """Send Slack alert"""
    import requests

    webhook_url = slack_config["webhook_url"]
    payload = {
        "text": message,
        "username": "Trading Monitor",
        "icon_emoji": ":chart_with_upwards_trend:",
    }

    response = requests.post(webhook_url, json=payload)
    response.raise_for_status()


def generate_daily_report(config, results, portfolio_tracker):
    """
    Generate daily report file

    Args:
        config: Configuration dictionary
        results: Monitoring results
        portfolio_tracker: PortfolioTracker instance
    """
    reports_dir = Path(config["data"]["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_file = reports_dir / f"daily_report_{results['date']}.txt"

    # Get recent performance summary
    recent_summary = portfolio_tracker.get_recent_summary(days=30)

    report = f"""
{'='*80}
DAILY MONITORING REPORT
Date: {results['date']}
Status: {results['status']}
{'='*80}

CURRENT METRICS
---------------
Portfolio Value: ${results['metrics'].get('portfolio_value', 0):,.0f}
Daily P&L: ${results['metrics'].get('daily_pnl', 0):,.0f} ({results['metrics'].get('daily_pnl_pct', 0):.2%})
Number of Positions: {results['metrics'].get('num_positions', 0)}

PERFORMANCE METRICS (1 YEAR)
----------------------------
Total Return: {results['metrics'].get('total_return', 0):.2%}
Sharpe Ratio: {results['metrics'].get('sharpe_ratio', 0):.2f}
Max Drawdown: {results['metrics'].get('max_drawdown', 0):.2%}
Win Rate: {results['metrics'].get('win_rate', 0):.1%}
Avg Daily Return: {results['metrics'].get('avg_daily_return', 0):.3%}

{recent_summary}

ALERTS
------
"""

    if results["alerts"]:
        report += "\n".join(f"  • {alert}" for alert in results["alerts"])
    else:
        report += "  No alerts\n"

    report += f"\n\n{'='*80}\n"

    with open(report_file, "w") as f:
        f.write(report)

    logger.info(f"[Report] ✓ Daily report saved to {report_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Daily portfolio monitoring")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--date", default=None, help="Date to monitor (YYYY-MM-DD), default: today"
    )
    parser.add_argument(
        "--no-alerts", action="store_true", help="Disable alert sending"
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

    # Run monitoring
    results = monitor_daily(
        config=config, date=args.date, send_alerts=not args.no_alerts
    )

    # Exit with appropriate code
    if results["status"] == "BREACH":
        sys.exit(1)  # Risk limit breached
    elif results["status"] == "ERROR":
        sys.exit(2)  # Monitoring error
    else:
        sys.exit(0)  # All OK


if __name__ == "__main__":
    main()
