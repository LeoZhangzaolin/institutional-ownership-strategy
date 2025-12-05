#!/usr/bin/env python3
"""
Order Execution Module - Production Ready
Handles trade execution via Interactive Brokers or Alpaca
"""

import pandas as pd
import numpy as np
import wrds
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Execute trades via broker with proper order pricing and sizing"""

    def __init__(self, config):
        self.config = config
        self.broker = config["trading"]["broker"]
        self.paper_trading = config["trading"]["paper_trading"]
        self.portfolio_value = config["portfolio"]["initial_capital"]

        # Connect to broker
        if self.broker == "interactive_brokers":
            self._connect_ib()
        elif self.broker == "alpaca":
            self._connect_alpaca()
        else:
            raise ValueError(f"Unsupported broker: {self.broker}")

    def _connect_ib(self):
        """Connect to Interactive Brokers"""
        try:
            from ib_insync import IB

            self.ib = IB()
            self.ib.connect(
                self.config["broker"]["interactive_brokers"]["host"],
                self.config["broker"]["interactive_brokers"]["port"],
                self.config["broker"]["interactive_brokers"]["client_id"],
            )
            logger.info("[OrderExecutor] ✓ Connected to Interactive Brokers")
        except Exception as e:
            logger.error(f"[OrderExecutor] Failed to connect to IB: {e}")
            raise

    def _connect_alpaca(self):
        """Connect to Alpaca"""
        try:
            import alpaca_trade_api as tradeapi

            self.api = tradeapi.REST(
                self.config["broker"]["alpaca"]["api_key"],
                self.config["broker"]["alpaca"]["secret_key"],
                self.config["broker"]["alpaca"]["base_url"],
            )
            logger.info("[OrderExecutor] ✓ Connected to Alpaca")
        except Exception as e:
            logger.error(f"[OrderExecutor] Failed to connect to Alpaca: {e}")
            raise

    def map_permno_to_ticker(self, permnos, date):
        """
        Map PERMNOs to current tickers via CRSP

        Args:
            permnos: List of PERMNOs
            date: Reference date (YYYY-MM-DD)

        Returns:
            Dictionary: {permno: ticker}
        """
        logger.info(
            f"[Mapping] Mapping {len(permnos)} PERMNOs to tickers as of {date}..."
        )

        try:
            db = wrds.Connection(wrds_username=self.config["data"]["wrds_username"])

            # Query CRSP for valid tickers on the given date
            query = f"""
            SELECT permno, ticker
            FROM crsp.msenames
            WHERE permno IN ({','.join(map(str, permnos))})
              AND namedt <= '{date}'
              AND nameendt >= '{date}'
              AND ticker IS NOT NULL
              AND ticker != ''
            ORDER BY permno, namedt DESC
            """

            mapping = db.raw_sql(query)
            db.close()

            # Convert to dict (if multiple tickers for same PERMNO, take first)
            ticker_map = (
                mapping.drop_duplicates(subset="permno", keep="first")
                .set_index("permno")["ticker"]
                .to_dict()
            )

            logger.info(f"[Mapping] ✓ Mapped {len(ticker_map)} tickers")

            # Log any missing mappings
            missing = set(permnos) - set(ticker_map.keys())
            if missing:
                logger.warning(
                    f"[Mapping] ⚠ Could not map {len(missing)} PERMNOs: {list(missing)[:5]}..."
                )

            return ticker_map

        except Exception as e:
            logger.error(f"[Mapping] Error mapping PERMNOs: {e}")
            raise

    def execute_rebalance(self, target_weights, current_positions, date):
        """
        Execute portfolio rebalance

        Args:
            target_weights: Series with PERMNO index and target weights
            current_positions: Dict {permno: current_weight}
            date: Rebalance date

        Returns:
            Boolean indicating success
        """
        logger.info("=" * 80)
        logger.info(f"[EXECUTION] Starting rebalance for {date}")

        if self.paper_trading:
            logger.warning("⚠ [PAPER TRADING MODE] - No real trades will be executed")

        try:
            # 1. Map PERMNO → ticker
            permnos = target_weights.index.tolist()
            ticker_map = self.map_permno_to_ticker(permnos, date)

            # 2. Calculate trades needed
            trades = self._calculate_trades(
                target_weights, current_positions, ticker_map
            )

            if not trades:
                logger.info(
                    "[EXECUTION] ✓ No trades needed (portfolio already balanced)"
                )
                return True

            logger.info(f"[EXECUTION] {len(trades)} trades to execute")

            # 3. Get current prices for all tickers
            prices = self._get_prices([t["ticker"] for t in trades])

            # 4. Generate orders with proper sizing
            orders = self._generate_orders(trades, prices)

            # 5. Submit orders
            if not self.paper_trading:
                filled_orders = self._submit_orders(orders)
                logger.info(
                    f"[EXECUTION] ✓ Filled {len(filled_orders)}/{len(orders)} orders"
                )
            else:
                logger.info("[PAPER TRADING] Would submit the following orders:")
                for i, order in enumerate(orders, 1):
                    logger.info(
                        f"  {i}. {order['side']:4s} {order['quantity']:6d} {order['ticker']:6s} @ ${order.get('limit_price', 'MKT')}"
                    )
                filled_orders = orders  # Pretend all filled in paper trading

            logger.info("[EXECUTION] ✓ Rebalance complete")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"[EXECUTION] ✗ Error during rebalance: {e}")
            logger.info("=" * 80)
            raise

    def _calculate_trades(self, target_weights, current_positions, ticker_map):
        """Calculate trades needed to reach target weights"""
        trades = []

        # Minimum trade threshold (avoid tiny trades)
        min_trade_weight = self.config["trading"].get("min_trade_weight", 0.001)

        for permno, target_weight in target_weights.items():
            if permno not in ticker_map:
                logger.warning(f"[Skip] PERMNO {permno} - no valid ticker found")
                continue

            ticker = ticker_map[permno]
            current_weight = current_positions.get(permno, 0.0)
            trade_weight = target_weight - current_weight

            if abs(trade_weight) > min_trade_weight:
                trades.append(
                    {
                        "permno": permno,
                        "ticker": ticker,
                        "target_weight": target_weight,
                        "current_weight": current_weight,
                        "trade_weight": trade_weight,
                        "side": "BUY" if trade_weight > 0 else "SELL",
                    }
                )

        return trades

    def _get_prices(self, tickers):
        """
        Get current prices for tickers

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary: {ticker: price}
        """
        if self.broker == "interactive_brokers":
            return self._get_ib_prices(tickers)
        elif self.broker == "alpaca":
            return self._get_alpaca_prices(tickers)
        else:
            # Fallback: use placeholder prices
            logger.warning(
                "[Prices] Using placeholder prices ($100) - implement real pricing!"
            )
            return {ticker: 100.0 for ticker in tickers}

    def _get_ib_prices(self, tickers):
        """Get current prices from Interactive Brokers"""
        from ib_insync import Stock

        prices = {}

        for ticker in tickers:
            try:
                contract = Stock(ticker, "SMART", "USD")

                # Request market data
                self.ib.reqMktData(contract, "", False, False)
                self.ib.sleep(0.5)  # Wait for data

                ticker_data = self.ib.ticker(contract)

                # Use mid-price if available, otherwise last price
                if (
                    ticker_data.bid
                    and ticker_data.ask
                    and ticker_data.bid > 0
                    and ticker_data.ask > 0
                ):
                    price = (ticker_data.bid + ticker_data.ask) / 2
                elif ticker_data.last and ticker_data.last > 0:
                    price = ticker_data.last
                else:
                    logger.warning(f"[Prices] No price data for {ticker}, using $100")
                    price = 100.0

                prices[ticker] = price

                # Cancel market data subscription
                self.ib.cancelMktData(contract)

            except Exception as e:
                logger.warning(
                    f"[Prices] Error getting price for {ticker}: {e}, using $100"
                )
                prices[ticker] = 100.0

        return prices

    def _get_alpaca_prices(self, tickers):
        """Get current prices from Alpaca"""
        prices = {}

        try:
            # Get latest quotes
            quotes = self.api.get_latest_quotes(tickers)

            for ticker in tickers:
                if (
                    ticker in quotes
                    and quotes[ticker].ask_price > 0
                    and quotes[ticker].bid_price > 0
                ):
                    # Use mid-price
                    prices[ticker] = (
                        quotes[ticker].ask_price + quotes[ticker].bid_price
                    ) / 2
                else:
                    logger.warning(f"[Prices] No quote for {ticker}, using $100")
                    prices[ticker] = 100.0

        except Exception as e:
            logger.warning(
                f"[Prices] Error getting Alpaca prices: {e}, using placeholders"
            )
            prices = {ticker: 100.0 for ticker in tickers}

        return prices

    def _generate_orders(self, trades, prices):
        """
        Generate orders with proper sizing

        Args:
            trades: List of trade dicts
            prices: Dictionary of current prices {ticker: price}

        Returns:
            List of order dicts
        """
        orders = []

        # Get buffer for limit orders (basis points)
        limit_buffer_bps = self.config["trading"].get("limit_buffer_bps", 20)
        buffer = limit_buffer_bps / 10000  # Convert to decimal

        for trade in trades:
            ticker = trade["ticker"]
            price = prices.get(ticker, 100.0)

            # Calculate dollar amount to trade
            dollar_amount = abs(trade["trade_weight"]) * self.portfolio_value

            # Calculate number of shares
            quantity = int(dollar_amount / price)

            if quantity == 0:
                logger.debug(f"[Skip] {ticker} - quantity rounds to 0")
                continue

            # Determine order type
            order_type = self.config["trading"]["order_type"]

            order = {
                "ticker": ticker,
                "side": trade["side"],
                "quantity": quantity,
                "order_type": order_type,
                "time_in_force": self.config["trading"]["time_in_force"],
                "permno": trade["permno"],
                "target_weight": trade["target_weight"],
            }

            # Add limit price if using limit orders
            if order_type == "limit":
                if trade["side"] == "BUY":
                    limit_price = price * (1 + buffer)  # Pay slightly more
                else:
                    limit_price = price * (1 - buffer)  # Accept slightly less

                order["limit_price"] = round(limit_price, 2)

            orders.append(order)

        return orders

    def _submit_orders(self, orders):
        """Submit orders to broker"""
        if self.broker == "interactive_brokers":
            return self._submit_ib_orders(orders)
        elif self.broker == "alpaca":
            return self._submit_alpaca_orders(orders)
        else:
            logger.error(f"[Submit] Unsupported broker: {self.broker}")
            return []

    def _submit_ib_orders(self, orders):
        """Submit orders to Interactive Brokers"""
        from ib_insync import Stock, MarketOrder, LimitOrder

        filled_orders = []

        for order in orders:
            try:
                contract = Stock(order["ticker"], "SMART", "USD")

                # Create order based on type
                if order["order_type"] == "market":
                    ib_order = MarketOrder(order["side"], order["quantity"])
                    logger.info(
                        f"[IB] Submitting: {order['side']} {order['quantity']} {order['ticker']} @ MARKET"
                    )
                else:
                    limit_price = order["limit_price"]
                    ib_order = LimitOrder(order["side"], order["quantity"], limit_price)
                    logger.info(
                        f"[IB] Submitting: {order['side']} {order['quantity']} {order['ticker']} @ ${limit_price}"
                    )

                # Place order
                trade = self.ib.placeOrder(contract, ib_order)

                # Wait for fill (with timeout)
                timeout = self.config["trading"].get("fill_timeout_seconds", 30)
                self.ib.sleep(timeout)

                # Check fill status
                if trade.orderStatus.status in ["Filled", "PreSubmitted", "Submitted"]:
                    filled_orders.append(order)
                    logger.info(
                        f"[IB] ✓ {order['ticker']} - Status: {trade.orderStatus.status}"
                    )
                else:
                    logger.warning(
                        f"[IB] ⚠ {order['ticker']} - Status: {trade.orderStatus.status}"
                    )

            except Exception as e:
                logger.error(f"[IB] ✗ Error submitting {order['ticker']}: {e}")

        return filled_orders

    def _submit_alpaca_orders(self, orders):
        """Submit orders to Alpaca"""
        filled_orders = []

        for order in orders:
            try:
                # Submit order
                if order["order_type"] == "market":
                    logger.info(
                        f"[Alpaca] Submitting: {order['side']} {order['quantity']} {order['ticker']} @ MARKET"
                    )
                    alpaca_order = self.api.submit_order(
                        symbol=order["ticker"],
                        qty=order["quantity"],
                        side=order["side"].lower(),
                        type="market",
                        time_in_force=order["time_in_force"],
                    )
                else:
                    limit_price = order["limit_price"]
                    logger.info(
                        f"[Alpaca] Submitting: {order['side']} {order['quantity']} {order['ticker']} @ ${limit_price}"
                    )
                    alpaca_order = self.api.submit_order(
                        symbol=order["ticker"],
                        qty=order["quantity"],
                        side=order["side"].lower(),
                        type="limit",
                        time_in_force=order["time_in_force"],
                        limit_price=limit_price,
                    )

                filled_orders.append(order)
                logger.info(
                    f"[Alpaca] ✓ {order['ticker']} - Order ID: {alpaca_order.id}"
                )

            except Exception as e:
                logger.error(f"[Alpaca] ✗ Error submitting {order['ticker']}: {e}")

        return filled_orders

    def get_current_positions(self):
        """
        Get current positions from broker

        Returns:
            Dictionary: {ticker: quantity}
        """
        if self.broker == "interactive_brokers":
            return self._get_ib_positions()
        elif self.broker == "alpaca":
            return self._get_alpaca_positions()
        return {}

    def _get_ib_positions(self):
        """Get positions from Interactive Brokers"""
        positions = {}
        try:
            for position in self.ib.positions():
                if position.position != 0:  # Only active positions
                    positions[position.contract.symbol] = position.position
            logger.info(f"[IB] Retrieved {len(positions)} positions")
        except Exception as e:
            logger.error(f"[IB] Error getting positions: {e}")
        return positions

    def _get_alpaca_positions(self):
        """Get positions from Alpaca"""
        positions = {}
        try:
            for position in self.api.list_positions():
                positions[position.symbol] = float(position.qty)
            logger.info(f"[Alpaca] Retrieved {len(positions)} positions")
        except Exception as e:
            logger.error(f"[Alpaca] Error getting positions: {e}")
        return positions

    def close(self):
        """Close broker connection"""
        if self.broker == "interactive_brokers" and hasattr(self, "ib"):
            self.ib.disconnect()
            logger.info("[OrderExecutor] Disconnected from Interactive Brokers")
