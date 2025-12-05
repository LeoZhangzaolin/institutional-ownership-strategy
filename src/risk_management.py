#!/usr/bin/env python3
"""
Risk Management Module - Production Ready
Circuit breakers, position limits, and portfolio tracking
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enforce risk limits and circuit breakers
    Monitor portfolio health in real-time
    """

    def __init__(self, config):
        self.config = config
        self.risk_limits = config["risk"]
        self.portfolio_tracker = PortfolioTracker(config)

        # Load last check state
        self.state_file = Path(config["data"]["paths"]["logs"]) / "risk_state.json"
        self.load_state()

    def load_state(self):
        """Load previous risk state"""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                self.state = json.load(f)
        else:
            self.state = {
                "last_check": None,
                "breaches_today": 0,
                "max_drawdown": 0.0,
                "peak_value": 0.0,
            }

    def save_state(self):
        """Save current risk state"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def check_daily_limits(self, current_positions, executor):
        """
        Check daily loss and drawdown limits

        Args:
            current_positions: Current portfolio positions
            executor: OrderExecutor instance for broker queries

        Returns:
            (bool, str): (passed, reason if failed)
        """
        logger.info("[Risk] Checking daily limits...")

        try:
            # Get current portfolio value from broker
            portfolio_value = self._get_portfolio_value(executor)

            # Update peak value
            if portfolio_value > self.state.get("peak_value", 0):
                self.state["peak_value"] = portfolio_value

            # Calculate daily P&L
            daily_pnl, daily_pnl_pct = self._calculate_daily_pnl(
                executor, portfolio_value
            )

            # Check daily loss limit
            max_daily_loss_pct = self.risk_limits["max_daily_loss_pct"]
            if daily_pnl_pct < -max_daily_loss_pct:
                msg = (
                    f"Daily loss limit breached: {daily_pnl_pct:.2%} "
                    f"(limit: -{max_daily_loss_pct:.2%})"
                )
                logger.error(f"[Risk] ✗ {msg}")
                self.state["breaches_today"] += 1
                self.save_state()
                return False, msg

            # Calculate drawdown from peak
            if self.state["peak_value"] > 0:
                drawdown = (portfolio_value - self.state["peak_value"]) / self.state[
                    "peak_value"
                ]
            else:
                drawdown = 0.0

            # Check max drawdown limit
            max_drawdown_pct = self.risk_limits["max_drawdown_pct"]
            if drawdown < -max_drawdown_pct:
                msg = (
                    f"Max drawdown limit breached: {drawdown:.2%} "
                    f"(limit: -{max_drawdown_pct:.2%})"
                )
                logger.error(f"[Risk] ✗ {msg}")
                self.state["breaches_today"] += 1
                self.save_state()
                return False, msg

            # Log status
            logger.info(f"[Risk] ✓ Daily P&L: ${daily_pnl:,.0f} ({daily_pnl_pct:.2%})")
            logger.info(f"[Risk] ✓ Portfolio value: ${portfolio_value:,.0f}")
            logger.info(
                f"[Risk] ✓ Drawdown: {drawdown:.2%} (limit: -{max_drawdown_pct:.2%})"
            )

            self.state["last_check"] = datetime.now().isoformat()
            self.save_state()

            return True, "All daily limits passed"

        except Exception as e:
            logger.error(f"[Risk] Error checking daily limits: {e}")
            return False, f"Error: {e}"

    def check_position_limits(self, target_weights, regime="normal"):
        """
        Check position concentration limits

        Args:
            target_weights: Series of target position weights
            regime: Current market regime

        Returns:
            (bool, str): (passed, reason if failed)
        """
        logger.info("[Risk] Checking position limits...")

        try:
            # Get regime-specific limits
            regime_params = self.config["regimes"][regime]
            max_position = regime_params["max_pos"]

            # Check individual position limits
            violations = target_weights[target_weights.abs() > max_position]

            if len(violations) > 0:
                msg = f"Position limit violations: {len(violations)} positions exceed {max_position:.1%}"
                logger.error(f"[Risk] ✗ {msg}")
                for permno, weight in violations.items():
                    logger.error(
                        f"  PERMNO {permno}: {weight:.2%} (limit: {max_position:.1%})"
                    )
                return False, msg

            # Check gross exposure
            gross_exposure = target_weights.abs().sum()
            max_gross = self.risk_limits.get("max_gross_exposure", 2.0)

            if gross_exposure > max_gross:
                msg = f"Gross exposure too high: {gross_exposure:.2f} (limit: {max_gross:.2f})"
                logger.error(f"[Risk] ✗ {msg}")
                return False, msg

            # Check net exposure (dollar-neutral constraint)
            net_exposure = target_weights.sum()
            max_net = self.risk_limits.get("max_net_exposure", 0.05)

            if abs(net_exposure) > max_net:
                msg = (
                    f"Net exposure too high: {net_exposure:.2%} (limit: {max_net:.1%})"
                )
                logger.warning(f"[Risk] ⚠ {msg}")
                # Warning only, don't fail

            logger.info(
                f"[Risk] ✓ Max position: {target_weights.abs().max():.2%} (limit: {max_position:.1%})"
            )
            logger.info(
                f"[Risk] ✓ Gross exposure: {gross_exposure:.2f} (limit: {max_gross:.2f})"
            )
            logger.info(f"[Risk] ✓ Net exposure: {net_exposure:.2%}")

            return True, "All position limits passed"

        except Exception as e:
            logger.error(f"[Risk] Error checking position limits: {e}")
            return False, f"Error: {e}"

    def check_execution_quality(self, filled_orders, target_orders):
        """
        Check execution quality metrics

        Args:
            filled_orders: List of filled orders
            target_orders: List of target orders

        Returns:
            (bool, str): (passed, reason if failed)
        """
        logger.info("[Risk] Checking execution quality...")

        try:
            if not target_orders:
                return True, "No orders to check"

            # Calculate fill rate
            fill_rate = len(filled_orders) / len(target_orders)
            min_fill_rate = self.risk_limits.get("min_fill_rate", 0.85)

            if fill_rate < min_fill_rate:
                msg = (
                    f"Fill rate too low: {fill_rate:.1%} (minimum: {min_fill_rate:.1%})"
                )
                logger.warning(f"[Risk] ⚠ {msg}")
                return False, msg

            # Calculate slippage (if limit orders)
            total_slippage = 0
            for order in filled_orders:
                if "limit_price" in order and "fill_price" in order:
                    slippage = (
                        abs(order["fill_price"] - order["limit_price"])
                        / order["limit_price"]
                    )
                    total_slippage += slippage

            avg_slippage = total_slippage / len(filled_orders) if filled_orders else 0
            max_slippage = self.risk_limits.get("max_slippage_bps", 50) / 10000

            if avg_slippage > max_slippage:
                msg = (
                    f"Slippage too high: {avg_slippage:.2%} (limit: {max_slippage:.2%})"
                )
                logger.warning(f"[Risk] ⚠ {msg}")
                # Warning only

            logger.info(
                f"[Risk] ✓ Fill rate: {fill_rate:.1%} (minimum: {min_fill_rate:.1%})"
            )
            logger.info(f"[Risk] ✓ Avg slippage: {avg_slippage*10000:.1f} bps")

            return True, "Execution quality acceptable"

        except Exception as e:
            logger.error(f"[Risk] Error checking execution quality: {e}")
            return False, f"Error: {e}"

    def _get_portfolio_value(self, executor):
        """Get current portfolio value from broker"""
        try:
            if executor.broker == "interactive_brokers":
                # Get account summary
                account = executor.ib.accountSummary()
                for item in account:
                    if item.tag == "NetLiquidation" and item.currency == "USD":
                        return float(item.value)

                # Fallback: sum positions
                total_value = 0
                for position in executor.ib.positions():
                    total_value += position.position * position.marketPrice
                return total_value

            elif executor.broker == "alpaca":
                # Get account
                account = executor.api.get_account()
                return float(account.portfolio_value)

            else:
                # Fallback to config
                logger.warning("[Risk] Using config portfolio value (no broker query)")
                return self.config["portfolio"]["initial_capital"]

        except Exception as e:
            logger.error(f"[Risk] Error getting portfolio value: {e}")
            # Fallback to config
            return self.config["portfolio"]["initial_capital"]

    def _calculate_daily_pnl(self, executor, current_value):
        """Calculate daily P&L"""
        try:
            # Get previous day's value from state or tracking
            prev_value = self.portfolio_tracker.get_previous_value()

            if prev_value is None:
                # First run or no history
                logger.warning("[Risk] No previous value for P&L calculation")
                prev_value = current_value

            # Calculate P&L
            daily_pnl = current_value - prev_value
            daily_pnl_pct = (
                (current_value - prev_value) / prev_value if prev_value > 0 else 0
            )

            return daily_pnl, daily_pnl_pct

        except Exception as e:
            logger.error(f"[Risk] Error calculating daily P&L: {e}")
            return 0, 0


class PortfolioTracker:
    """
    Track portfolio state and performance over time
    """

    def __init__(self, config):
        self.config = config
        self.history_file = (
            Path(config["data"]["paths"]["logs"]) / "portfolio_history.csv"
        )
        self.load_history()

    def load_history(self):
        """Load portfolio history"""
        if self.history_file.exists():
            self.history = pd.read_csv(self.history_file, parse_dates=["date"])
        else:
            self.history = pd.DataFrame(
                columns=[
                    "date",
                    "portfolio_value",
                    "daily_pnl",
                    "daily_return",
                    "gross_exposure",
                    "net_exposure",
                    "num_positions",
                ]
            )

    def save_history(self):
        """Save portfolio history"""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history.to_csv(self.history_file, index=False)

    def update(self, date, portfolio_value, positions, daily_pnl=0):
        """
        Update portfolio tracking

        Args:
            date: Current date
            portfolio_value: Current portfolio value
            positions: Current positions dict
            daily_pnl: Daily P&L
        """
        # Calculate metrics
        prev_value = self.get_previous_value()
        daily_return = (portfolio_value - prev_value) / prev_value if prev_value else 0

        # Calculate exposures
        total_long = sum(v for v in positions.values() if v > 0)
        total_short = sum(abs(v) for v in positions.values() if v < 0)
        gross_exposure = total_long + total_short
        net_exposure = total_long - total_short

        # Add to history
        new_row = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(date),
                    "portfolio_value": portfolio_value,
                    "daily_pnl": daily_pnl,
                    "daily_return": daily_return,
                    "gross_exposure": gross_exposure,
                    "net_exposure": net_exposure,
                    "num_positions": len(positions),
                }
            ]
        )

        self.history = pd.concat([self.history, new_row], ignore_index=True)
        self.save_history()

        logger.info(
            f"[Tracking] Updated portfolio: ${portfolio_value:,.0f}, "
            f"P&L: ${daily_pnl:,.0f}, positions: {len(positions)}"
        )

    def get_previous_value(self):
        """Get previous day's portfolio value"""
        if len(self.history) == 0:
            return None
        return self.history.iloc[-1]["portfolio_value"]

    def get_performance_metrics(self, lookback_days=252):
        """
        Calculate performance metrics

        Args:
            lookback_days: Number of days to look back

        Returns:
            Dictionary of metrics
        """
        if len(self.history) < 2:
            return {
                "total_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "avg_daily_return": 0,
            }

        # Get recent history
        recent = self.history.tail(lookback_days).copy()

        # Total return
        first_value = recent.iloc[0]["portfolio_value"]
        last_value = recent.iloc[-1]["portfolio_value"]
        total_return = (last_value - first_value) / first_value

        # Sharpe ratio (annualized)
        returns = recent["daily_return"].dropna()
        if len(returns) > 0:
            sharpe = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0
                else 0
            )
        else:
            sharpe = 0

        # Max drawdown
        cummax = recent["portfolio_value"].cummax()
        drawdown = (recent["portfolio_value"] - cummax) / cummax
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        # Average daily return
        avg_daily_return = returns.mean() if len(returns) > 0 else 0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_daily_return": avg_daily_return,
            "num_days": len(recent),
        }

    def get_recent_summary(self, days=30):
        """Get summary of recent performance"""
        if len(self.history) < 2:
            return "No performance history available"

        recent = self.history.tail(days)

        summary = f"""
Portfolio Performance (Last {days} days):
  Current Value: ${recent.iloc[-1]['portfolio_value']:,.0f}
  Total P&L: ${recent['daily_pnl'].sum():,.0f}
  Avg Daily Return: {recent['daily_return'].mean():.3%}
  Volatility: {recent['daily_return'].std():.3%}
  Best Day: ${recent['daily_pnl'].max():,.0f}
  Worst Day: ${recent['daily_pnl'].min():,.0f}
  Sharpe (annualized): {recent['daily_return'].mean() / recent['daily_return'].std() * np.sqrt(252):.2f}
"""
        return summary
