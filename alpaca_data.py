import asyncio
import websockets
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import os
import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from collections import deque

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
# from alpaca.trading.client import TradingClient # Not used in this data-focused class
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.common.exceptions import APIError # More specific exception

# --- Configuration & Constants ---
DEFAULT_WEBSOCKET_RECONNECT_DELAY = 5  # seconds
MAX_WEBSOCKET_RECONNECT_DELAY = 60 # seconds
WEBSOCKET_TIMEOUT = 60  # seconds for receiving a message
WEBSOCKET_PING_INTERVAL = 30 # seconds to send a ping to keep connection alive (Alpaca might do this, but good practice)
MAX_TRADE_CACHE_SIZE = 5000
MAX_QUOTE_CACHE_SIZE = 5000 # If you decide to cache quotes

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class AlpacaData:
    """
    Manages data fetching from Alpaca (both REST API for historical data
    and WebSocket for real-time data) with a fallback to yfinance.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        feed: DataFeed = DataFeed.SIP # SIP for paid, IEX for free on Alpaca
    ):
        """
        Initializes the AlpacaData client.

        Args:
            api_key (Optional[str]): Alpaca API key. Defaults to APCA_API_KEY_ID env var.
            secret_key (Optional[str]): Alpaca API secret key. Defaults to APCA_API_SECRET_KEY env var.
            paper (bool): Whether to use paper trading base URL. Defaults to True.
            feed (DataFeed): Data feed to use for WebSocket (SIP, IEX, OTC).
                             SIP is generally for paid plans. IEX for free.
        """
        self.api_key = api_key or os.environ.get("APCA_API_KEY_ID")
        self.secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY")
        self.paper = paper
        self.feed = feed # SIP, IEX, or OTC

        if not self.api_key or not self.secret_key:
            logger.critical("Alpaca API Key/Secret not found. Please provide them or set environment variables.")
            raise ValueError("Alpaca API Key/Secret not found.")

        try:
            # Note: StockHistoricalDataClient doesn't take a 'paper' argument directly.
            # The URL selection (paper/live) is implicit in how Alpaca SDK handles it,
            # or you might need to pass full base_url if overriding.
            # For now, let's assume SDK handles paper/live for data based on key type or future config.
            # If using a specific paper data URL:
            # self.rest_client = StockHistoricalDataClient(self.api_key, self.secret_key, url_override="https://data.paper-api.alpaca.markets/v2")
            self.rest_client = StockHistoricalDataClient(self.api_key, self.secret_key)
            logger.info("Alpaca StockHistoricalDataClient initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize Alpaca REST API client: {e}", exc_info=True)
            raise

        # WebSocket URL depends on the feed (SIP for paid, IEX for free)
        # Alpaca documentation: https://alpaca.markets/docs/api-references/market-data-api/stock-pricing-data/realtime/
        if self.feed == DataFeed.SIP:
            self.websocket_base_url = "wss://stream.data.alpaca.markets/v2/sip"
        elif self.feed == DataFeed.IEX:
            self.websocket_base_url = "wss://stream.data.alpaca.markets/v2/iex"
        elif self.feed == DataFeed.OTC:
            self.websocket_base_url = "wss://stream.data.alpaca.markets/v2/otc"
        else:
            raise ValueError(f"Unsupported data feed: {self.feed}")

        logger.info(f"Using WebSocket URL: {self.websocket_base_url}")

        # Use deque for efficient fixed-size caches
        self.trade_cache: Dict[str, deque] = {} # Symbol -> deque of trades
        # self.quote_cache: Dict[str, deque] = {} # If you implement quote handling

        self._websocket_running = False
        self._websocket_task: Optional[asyncio.Task] = None

    async def _authenticate_websocket(self, ws: websockets.WebSocketClientProtocol) -> bool:
        """Authenticates the WebSocket connection."""
        try:
            auth_data = {"action": "auth", "key": self.api_key, "secret": self.secret_key}
            await ws.send(json.dumps(auth_data))
            response_str = await asyncio.wait_for(ws.recv(), timeout=10)
            response = json.loads(response_str)
            logger.debug(f"WebSocket Auth Response: {response}")

            if isinstance(response, list) and response[0].get("T") == "success" and response[0].get("msg") == "authenticated":
                logger.info("WebSocket authenticated successfully.")
                # Some Alpaca streams send a second message immediately after auth, also related to connection status.
                # It's good to consume it if it's part of the auth handshake.
                # Example: [{"T":"subscription","trades":[],"quotes":["AAPL"],"bars":[],"dailyBars":[],"updatedBars":[]}]
                # This indicates current subscriptions. If you subscribe later, this initial message might be different.
                try:
                    # Try to receive a second message, but don't fail if it's not there or times out quickly
                    second_response_str = await asyncio.wait_for(ws.recv(), timeout=5)
                    logger.debug(f"WebSocket Post-Auth/Subscription Info: {json.loads(second_response_str)}")
                except asyncio.TimeoutError:
                    logger.debug("No immediate second message after WebSocket auth.")
                except Exception as e:
                    logger.warning(f"Error processing second message after WebSocket auth: {e}")
                return True
            else:
                logger.error(f"WebSocket authentication failed: {response}")
                return False
        except asyncio.TimeoutError:
            logger.error("WebSocket authentication timed out.")
            return False
        except Exception as e:
            logger.error(f"Error during WebSocket authentication: {e}", exc_info=True)
            return False

    async def _subscribe_to_streams(self, ws: websockets.WebSocketClientProtocol, tickers: List[str]):
        """Subscribes to trades and quotes for the given tickers."""
        if not tickers:
            logger.warning("No tickers provided for WebSocket subscription.")
            return

        subscription_data = {"action": "subscribe", "trades": tickers, "quotes": tickers} # Or subscribe to bars if needed
        await ws.send(json.dumps(subscription_data))
        logger.info(f"Sent subscription request for tickers: {tickers}")
        # It's good to confirm subscription, Alpaca usually sends a message like:
        # [{"T":"subscription","trades":["AAPL"],"quotes":["AAPL"],"bars":[],"dailyBars":[],"updatedBars":[]}]
        try:
            sub_response_str = await asyncio.wait_for(ws.recv(), timeout=10)
            sub_response = json.loads(sub_response_str)
            logger.info(f"WebSocket Subscription Response: {sub_response}")
            # Basic check, can be more thorough based on expected response format
            if isinstance(sub_response, list) and sub_response[0].get("T") == "subscription":
                logger.info("Subscription confirmed by server.")
            else:
                logger.warning(f"Unexpected subscription response: {sub_response}")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for WebSocket subscription confirmation.")
        except Exception as e:
            logger.error(f"Error processing subscription confirmation: {e}", exc_info=True)


    async def _websocket_handler(self, tickers: List[str]):
        """Handles WebSocket connection, authentication, subscription, and message processing."""
        current_delay = DEFAULT_WEBSOCKET_RECONNECT_DELAY
        self._websocket_running = True

        while self._websocket_running:
            try:
                async with websockets.connect(self.websocket_base_url, ping_interval=WEBSOCKET_PING_INTERVAL) as ws:
                    logger.info("WebSocket connection established.")
                    current_delay = DEFAULT_WEBSOCKET_RECONNECT_DELAY # Reset delay on successful connect

                    if not await self._authenticate_websocket(ws):
                        # If auth fails, usually means bad keys or server issue.
                        # Could stop retrying or have a different retry strategy.
                        logger.error("WebSocket authentication failed. Will attempt reconnect with delay.")
                        await asyncio.sleep(current_delay)
                        current_delay = min(current_delay * 2, MAX_WEBSOCKET_RECONNECT_DELAY)
                        continue

                    await self._subscribe_to_streams(ws, tickers)

                    while self._websocket_running:
                        try:
                            message_str = await asyncio.wait_for(ws.recv(), timeout=WEBSOCKET_TIMEOUT)
                            messages = json.loads(message_str)
                            for message in messages:
                                self._process_websocket_message(message)
                        except asyncio.TimeoutError:
                            logger.debug(f"WebSocket: No message received in {WEBSOCKET_TIMEOUT}s. Sending ping (if not auto).")
                            # websockets library handles pings automatically if ping_interval is set
                            # If not, you might need to send a manual ping: await ws.ping()
                        except websockets.exceptions.ConnectionClosed as e:
                            logger.warning(f"WebSocket connection closed: {e}. Attempting to reconnect...")
                            break # Break inner loop to trigger reconnection
                        except Exception as e:
                            logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
                            # Depending on error, you might want to break or continue

            except (websockets.exceptions.InvalidURI, websockets.exceptions.WebSocketException, ConnectionRefusedError, OSError) as e:
                logger.error(f"WebSocket connection/setup failed: {e}. Retrying in {current_delay}s...")
            except Exception as e:
                logger.critical(f"Unexpected error in WebSocket handler: {e}. Retrying in {current_delay}s...", exc_info=True)

            if not self._websocket_running: # Check if stop was requested
                break

            await asyncio.sleep(current_delay)
            current_delay = min(current_delay * 2, MAX_WEBSOCKET_RECONNECT_DELAY)

        logger.info("WebSocket handler stopped.")

    def _process_websocket_message(self, message: Dict[str, Any]):
        """Processes a single message from the WebSocket stream."""
        msg_type = message.get("T")
        if msg_type == "t":  # Trade
            self._store_trade(message)
        elif msg_type == "q": # Quote
            # self._store_quote(message) # Implement if needed
            logger.debug(f"Received quote (not stored): {message}")
        elif msg_type == "success" or msg_type == "subscription" or msg_type == "error":
            # These are usually handled during auth/sub or indicate issues
            logger.info(f"Control message: {message}")
        else:
            logger.debug(f"Received unhandled WebSocket message type '{msg_type}': {message}")


    def _store_trade(self, trade_data: Dict[str, Any]):
        """Stores trade data into the cache."""
        symbol = trade_data.get("S")
        timestamp_str = trade_data.get("t") # e.g., "2023-10-26T14:30:00.123456789Z"
        price = trade_data.get("p")
        volume = trade_data.get("s") # 'x' in Alpaca docs is exchange, 's' is size

        if not all([symbol, timestamp_str, price is not None, volume is not None]):
            logger.warning(f"Skipping incomplete trade data: {trade_data}")
            return
        try:
            # Alpaca timestamps are RFC3339 format
            timestamp = pd.to_datetime(timestamp_str)
        except Exception as e:
            logger.error(f"Error parsing trade timestamp '{timestamp_str}': {e}", exc_info=True)
            return

        if symbol not in self.trade_cache:
            self.trade_cache[symbol] = deque(maxlen=MAX_TRADE_CACHE_SIZE)

        self.trade_cache[symbol].append({
            "timestamp": timestamp,
            "price": float(price),
            "volume": int(volume)
        })
        # logger.debug(f"Stored trade for {symbol}: P={price}, V={volume} @ {timestamp}")

    async def start_websocket_stream(self, tickers: List[str]):
        """Starts the WebSocket data stream in a background task."""
        if not self.api_key or not self.secret_key:
            logger.error("Cannot start WebSocket: API Key/Secret missing.")
            return
        if not tickers:
            logger.warning("Cannot start WebSocket: No tickers provided.")
            return
        if self._websocket_running:
            logger.warning("WebSocket stream is already running or starting.")
            return

        logger.info(f"Starting WebSocket stream for tickers: {tickers}")
        self._websocket_task = asyncio.create_task(self._websocket_handler(tickers))
        # Give it a moment to start up and potentially fail early
        await asyncio.sleep(0.1)
        if self._websocket_task.done() and self._websocket_task.exception():
            logger.error(f"WebSocket task failed on startup: {self._websocket_task.exception()}")
            self._websocket_running = False # Ensure flag is reset

    async def stop_websocket_stream(self):
        """Stops the WebSocket data stream."""
        if not self._websocket_running or not self._websocket_task:
            logger.info("WebSocket stream is not running.")
            return

        logger.info("Stopping WebSocket stream...")
        self._websocket_running = False
        if self._websocket_task:
            try:
                # Give the handler a chance to exit gracefully
                await asyncio.wait_for(self._websocket_task, timeout=WEBSOCKET_TIMEOUT + 5)
            except asyncio.TimeoutError:
                logger.warning("WebSocket task did not stop in time, cancelling.")
                self._websocket_task.cancel()
                try:
                    await self._websocket_task # Await cancellation
                except asyncio.CancelledError:
                    logger.info("WebSocket task successfully cancelled.")
            except Exception as e:
                logger.error(f"Error during WebSocket task shutdown: {e}", exc_info=True)
        self._websocket_task = None
        logger.info("WebSocket stream stopped.")

    def get_latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Gets the latest trade for a symbol from the cache.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with 'timestamp', 'price', 'volume', or None.
        """
        if symbol in self.trade_cache and self.trade_cache[symbol]:
            return self.trade_cache[symbol][-1]
        return None

    def _map_str_to_timeframe(self, timeframe_str: str) -> TimeFrame:
        """Maps a string like "5Min", "1H", "1D" to Alpaca TimeFrame object."""
        tf_str = timeframe_str.lower()
        try:
            if "min" in tf_str:
                minutes = int(tf_str.replace("min", ""))
                return TimeFrame(minutes, TimeFrameUnit.Minute)
            elif "h" in tf_str:
                hours = int(tf_str.replace("h", ""))
                return TimeFrame(hours, TimeFrameUnit.Hour)
            elif "d" in tf_str:
                days = int(tf_str.replace("d", ""))
                return TimeFrame(days, TimeFrameUnit.Day)
            else:
                logger.warning(f"Unsupported timeframe string: {timeframe_str}. Defaulting to 5Min.")
                return TimeFrame(5, TimeFrameUnit.Minute)
        except ValueError:
            logger.warning(f"Invalid format for timeframe string: {timeframe_str}. Defaulting to 5Min.")
            return TimeFrame(5, TimeFrameUnit.Minute)

    def _map_timeframe_to_yfinance_interval(self, timeframe_str: str) -> str:
        """Maps our timeframe string to yfinance interval string (e.g., "5m", "1h", "1d")."""
        tf_str = timeframe_str.lower()
        if "min" in tf_str:
            return tf_str.replace("min", "m")
        elif "h" in tf_str:
            return tf_str.replace("h", "h") # yfinance uses 'h' for hour
        elif "d" in tf_str:
            return tf_str.replace("d", "d")
        logger.warning(f"Could not map {timeframe_str} to yfinance interval. Defaulting to '5m'.")
        return "5m" # Default

    def get_historical_data(
        self,
        tickers: List[str],
        timeframe_str: str = "5Min",
        limit_per_symbol: int = 200,
        days_to_look_back_for_intraday: int = 30, # How far back to start scan for intraday Alpaca data
        max_days_for_yfinance_intraday: int = 59 # yfinance intraday limit
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches historical OHLCV data for given tickers.
        Tries Alpaca first, then falls back to yfinance.

        Args:
            tickers (List[str]): List of stock symbols.
            timeframe_str (str): Timeframe string (e.g., "1Min", "5Min", "1H", "1D").
            limit_per_symbol (int): Desired number of bars per symbol.
            days_to_look_back_for_intraday (int): For Alpaca intraday, how many calendar days to check.
            max_days_for_yfinance_intraday (int): Max period for yfinance intraday (usually < 60d).

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbol to DataFrame of OHLCV data.
                                     DataFrame will be empty if data couldn't be fetched.
        """
        results: Dict[str, pd.DataFrame] = {}
        alpaca_tf = self._map_str_to_timeframe(timeframe_str)

        # Calculate a dynamic start_date for Alpaca requests to try and get enough data
        # This is a rough estimate; market hours, holidays, etc., affect bar count.
        now_utc = datetime.now(timezone.utc)
        if alpaca_tf.unit == TimeFrameUnit.Minute:
            # Estimate days needed: (bars * minutes_per_bar) / (minutes_in_trading_day)
            # Add buffer for non-trading days.
            est_days = (limit_per_symbol * alpaca_tf.amount) / (6.5 * 60) * 1.7 # 1.7 for weekends/sparse data
            start_dt_alpaca = now_utc - timedelta(days=max(int(est_days) + 2, days_to_look_back_for_intraday))
        elif alpaca_tf.unit == TimeFrameUnit.Hour:
            est_days = (limit_per_symbol * alpaca_tf.amount) / 6.5 * 1.7
            start_dt_alpaca = now_utc - timedelta(days=max(int(est_days) + 2, days_to_look_back_for_intraday * 2))
        else: # Daily
            est_days = limit_per_symbol * 1.7 # For weekends/holidays
            start_dt_alpaca = now_utc - timedelta(days=max(int(est_days) + 20, 90)) # Longer default for daily

        # Alpaca data feed for historical bars (IEX is free, SIP requires paid plan)
        # The StockHistoricalDataClient uses the feed associated with your API key type by default.
        # You can specify if needed for `get_stock_bars`.
        # For historical, `feed` argument in StockBarsRequest refers to the source.
        # It might be different from the real-time feed. Usually SIP for paid, IEX for free.
        historical_feed = self.feed # Align with WebSocket feed by default or choose explicitly.
        if self.feed not in [DataFeed.SIP, DataFeed.IEX, DataFeed.OTC]: # Ensure it's a valid data feed for historical
            logger.warning(f"Historical data feed {self.feed} might not be optimal. Consider SIP or IEX.")
            historical_feed = DataFeed.IEX # Default to IEX for broad compatibility if config is odd

        for symbol in tickers:
            df = pd.DataFrame()
            try:
                logger.info(f"Fetching {timeframe_str} bars for {symbol} from Alpaca (limit ~{limit_per_symbol}, start {start_dt_alpaca.strftime('%Y-%m-%d')}).")
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=alpaca_tf,
                    start=start_dt_alpaca, # UTC datetime object
                    # end=now_utc, # Optional: defaults to current time
                    limit=min(limit_per_symbol + 50, 10000), # Fetch a bit more, respect API max limit (10k)
                    adjustment=Adjustment.SPLIT, # Or RAW, DIVIDEND etc.
                    feed=historical_feed # Use IEX for free accounts, SIP for paid
                )
                bars_data = self.rest_client.get_stock_bars(request_params)

                if bars_data and symbol in bars_data.df.index.get_level_values('symbol').unique():
                    # Alpaca SDK returns a multi-index DataFrame if multiple symbols requested,
                    # or single index if one symbol. Ensure we handle both.
                    if isinstance(bars_data.df.index, pd.MultiIndex):
                        df_symbol = bars_data.df.loc[symbol].copy()
                    else: # Single symbol in request, or data for only one symbol returned
                        df_symbol = bars_data.df.copy()

                    df_symbol.columns = df_symbol.columns.str.lower()
                    df = df_symbol.tail(limit_per_symbol) # Ensure we only take the desired limit
                    logger.info(f"SUCCESS (Alpaca): Fetched {len(df)} '{timeframe_str}' bars for {symbol}.")
                else:
                    logger.info(f"INFO (Alpaca): No data returned for {symbol} (Timeframe: {timeframe_str}).")

            except APIError as e: # Alpaca specific API errors
                 logger.warning(f"WARNING (Alpaca API Error for {symbol}): {e}. Status: {e.status_code if hasattr(e, 'status_code') else 'N/A'}")
                 if hasattr(e, 'status_code') and e.status_code == 422: # Unprocessable Entity (e.g. symbol not found for feed)
                     logger.warning(f"Symbol {symbol} might not be available on Alpaca {historical_feed} feed.")
            except Exception as e:
                logger.warning(f"WARNING (Alpaca general error for {symbol}): {e}", exc_info=False) # exc_info=False to reduce noise for common fallbacks

            if df.empty:
                logger.info(f"FALLBACK: Attempting to fetch {symbol} from yfinance (Timeframe: {timeframe_str}).")
                yf_interval = self._map_timeframe_to_yfinance_interval(timeframe_str)
                yf_period = "1y" # Default period, adjust based on interval and limit

                try: # Calculate yfinance period
                    if 'm' in yf_interval or 'h' in yf_interval: # Intraday
                        # For intraday, yfinance period is max 730d for some intervals, but often 60d for <1h
                        # Let's calculate days needed, respecting yfinance limits
                        minutes_per_bar = int(yf_interval[:-1]) if yf_interval[:-1].isdigit() else (60 if 'h' in yf_interval else 1)
                        if 'h' in yf_interval: minutes_per_bar *= 60

                        trading_minutes_per_day = 6.5 * 60
                        days_needed_for_yfinance = (limit_per_symbol * minutes_per_bar) / trading_minutes_per_day
                        # Add buffer for non-trading days & ensure it's at least a few days
                        yf_days_param = min(max(int(days_needed_for_yfinance * 1.7) + 2, 5), max_days_for_yfinance_intraday)
                        yf_period = f"{yf_days_param}d"
                    else: # Daily
                        yf_days_param = int(limit_per_symbol * 1.7) + 20 # More buffer for daily
                        yf_period = f"{yf_days_param}d"
                except Exception as e_calc:
                    logger.error(f"Error calculating yfinance period: {e_calc}. Defaulting period.")


                for attempt in range(3): # Retry yfinance
                    try:
                        logger.debug(f"yfinance.download attempt {attempt+1} for {symbol} with period='{yf_period}', interval='{yf_interval}'")
                        temp_df = yf.download(
                            tickers=symbol,
                            period=yf_period,
                            interval=yf_interval,
                            progress=False,
                            timeout=15
                        )
                        if not temp_df.empty:
                            temp_df = temp_df.rename(columns=str.lower).dropna()
                            # Ensure timezone localization to UTC if not already (yfinance often returns tz-aware)
                            if temp_df.index.tz is None:
                                temp_df.index = temp_df.index.tz_localize('UTC') # Assuming market's tz, then convert
                            else:
                                temp_df.index = temp_df.index.tz_convert('UTC')
                            df = temp_df.tail(limit_per_symbol)
                            logger.info(f"SUCCESS (yfinance attempt {attempt+1}): Fetched {len(df)} bars for {symbol}.")
                            break
                        else:
                            logger.warning(f"WARNING (yfinance attempt {attempt+1}): Empty data for {symbol}.")
                    except Exception as yf_e:
                        logger.error(f"ERROR (yfinance attempt {attempt+1} for {symbol}): {yf_e}", exc_info=False)
                    
                    if not df.empty: break # Exit retry loop if successful
                    await asyncio.sleep(1.5 ** attempt) # Exponential backoff for yfinance retries

            # Ensure DataFrame index is DatetimeIndex and UTC for consistency
            if not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC') # Fallback if somehow not set
                elif df.index.tz.zone != 'UTC':
                    df.index = df.index.tz_convert('UTC')
                df = df.sort_index() # Ensure chronological order

            results[symbol] = df
        return results

async def main():
    """Example usage of AlpacaData class."""
    # For free accounts, use feed=DataFeed.IEX for websockets
    # For paid accounts, use feed=DataFeed.SIP
    alpaca_client = AlpacaData(paper=True, feed=DataFeed.IEX) # Or DataFeed.SIP if you have a paid plan

    tickers_to_stream = ["AAPL", "MSFT"]
    tickers_for_historical = ["SPY", "AAPL", "GOOG", "TSLA", "INVALID"]

    # --- Test Historical Data ---
    logger.info("\n--- Testing Historical Data ---")
    historical_data_5min = alpaca_client.get_historical_data(
        tickers_for_historical,
        timeframe_str="5Min",
        limit_per_symbol=50
    )
    for symbol, df in historical_data_5min.items():
        logger.info(f"\nHistorical 5Min data for {symbol} (first 3 rows, last 3 rows if > 6 rows):")
        if not df.empty:
            if len(df) > 6:
                print(df.head(3))
                print("...")
                print(df.tail(3))
            else:
                print(df)
            logger.info(f"Shape: {df.shape}")
        else:
            logger.info("No data fetched.")

    historical_data_1d = alpaca_client.get_historical_data(
        tickers_for_historical,
        timeframe_str="1D",
        limit_per_symbol=30
    )
    for symbol, df in historical_data_1d.items():
        logger.info(f"\nHistorical 1D data for {symbol} (last 5 rows):")
        if not df.empty:
            print(df.tail())
            logger.info(f"Shape: {df.shape}")
        else:
            logger.info("No data fetched.")


    # --- Test WebSocket Stream ---
    logger.info("\n--- Testing WebSocket Stream ---")
    await alpaca_client.start_websocket_stream(tickers_to_stream)

    try:
        for i in range(20): # Run for 20 seconds, printing latest trades
            await asyncio.sleep(1)
            if i % 5 == 0: # Print every 5 seconds
                for ticker in tickers_to_stream:
                    latest_trade = alpaca_client.get_latest_trade(ticker)
                    if latest_trade:
                        logger.info(f"Latest trade for {ticker}: {latest_trade}")
                    else:
                        logger.info(f"No trades received yet for {ticker} via WebSocket.")
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Stopping WebSocket stream...")
        await alpaca_client.stop_websocket_stream()
        logger.info("Program finished.")

if __name__ == "__main__":
    # Ensure you have APCA_API_KEY_ID and APCA_API_SECRET_KEY in your environment
    # For paper trading with a free plan, ensure your keys are for paper.
    # Example:
    # export APCA_API_KEY_ID="YOUR_PAPER_KEY_ID"
    # export APCA_API_SECRET_KEY="YOUR_PAPER_SECRET_KEY"
    try:
        asyncio.run(main())
    except ValueError as e: # Catch API key/secret missing error
        logger.critical(f"Initialization failed: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
