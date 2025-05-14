# alpaca_data.py
import asyncio
import websockets
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import time # For retry delay
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment
class AlpacaData:
    def __init__(self):
        self.api_key = os.environ.get("APCA_API_KEY_ID")
        self.secret_key = os.environ.get("APCA_API_SECRET_KEY")
        self.base_rest_url = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets") # Default to paper

        if not self.api_key or not self.secret_key:
            # This should ideally stop the app or be handled more gracefully in app.py
            print("FATAL: Alpaca API Key/Secret not found in environment variables. Cannot initialize AlpacaData.")
            raise ValueError("Alpaca API Key/Secret not found. Please set in secrets.toml.")

        try:
            self.rest_api = StockHistoricalDataClient(self.api_key, self.secret_key)
            # Test connection
            #self.rest_api.get_account() 
            #print(f"Successfully connected to Alpaca REST API at {self.base_rest_url}.")
        except Exception as e:
            print(f"FATAL: Failed to connect to Alpaca REST API: {e}. Check credentials and API URL.")
            raise
        except Exception as e:
            print(f"FATAL: An unexpected error occurred during Alpaca REST API client initialization: {e}")
            raise


        self.websocket_base_url = "wss://stream.data.alpaca.markets/v2/sip" # SIP for US Equities
        self.data_cache = {} # Stores raw trades from WebSocket for get_latest_trade or very recent 1min bars

    async def start_websocket(self, tickers):
        if not tickers:
            print("WebSocket: No tickers provided to subscribe.")
            return
        if not self.api_key or not self.secret_key: # Redundant check, __init__ should catch
            print("WebSocket: API Key/Secret missing.")
            return
        try:
            async with websockets.connect(self.websocket_base_url) as ws:
                auth_data = {"action": "auth", "key": self.api_key, "secret": self.secret_key}
                await ws.send(json.dumps(auth_data))
                auth_response_raw = await ws.recv()
                print(f"WebSocket Auth Response: {auth_response_raw}")
                auth_msg = json.loads(auth_response_raw)
                if not (isinstance(auth_msg, list) and auth_msg[0].get("T") == "success" and auth_msg[0].get("msg") == "connected"):
                    print(f"❌ WebSocket authentication failed: {auth_msg}")
                    return # Stop if auth fails

                # Check for second success message if Alpaca sends two-part auth
                second_auth_msg_raw = await ws.recv()
                print(f"WebSocket Second Auth/Subscription Response: {second_auth_msg_raw}")
                second_auth_msg = json.loads(second_auth_msg_raw)
                if not (isinstance(second_auth_msg, list) and second_auth_msg[0].get("T") == "success" and second_auth_msg[0].get("msg") == "authenticated"):
                     print(f"❌ WebSocket post-auth step failed: {second_auth_msg}")
                     # Continue to try subscribing anyway, or return

                subscription_data = {"action": "subscribe", "trades": tickers, "quotes": tickers} # Subscribe to trades and quotes
                await ws.send(json.dumps(subscription_data))
                
                # Handle multiple subscription confirmation messages
                subscribed_all = False
                subscription_confirmations = []
                for _ in range(len(tickers) * 2 + 1): # Expect confirmations for trades & quotes per ticker + general success
                    try:
                        sub_response_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                        subscription_confirmations.append(sub_response_raw)
                        print(f"WebSocket Subscription Response Part: {sub_response_raw}")
                        # A more robust check would be to see if all tickers are confirmed
                        # For now, just print and proceed
                    except asyncio.TimeoutError:
                        print("WebSocket: Timeout waiting for full subscription confirmation.")
                        break # Break if timeout

                print(f"WebSocket: Proceeding after subscription attempts. Confirmations: {subscription_confirmations}")

                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=65) # Increased timeout
                        updates = json.loads(message)
                        for update in updates:
                            if update.get("T") == "t": # Trade update
                                self._store_trade(update)
                            # elif update.get("T") == "q": # Quote update
                            #    self._store_quote(update) # Implement if needed
                    except asyncio.TimeoutError:
                        print("WebSocket: No message received in 65s. Connection might be stale.")
                        # Consider a ping mechanism if Alpaca's WS supports/requires it
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"❌ WebSocket connection closed: {e}")
                        break
                    except Exception as e:
                        print(f"❌ Error receiving/processing WebSocket message: {e}")
        except Exception as e:
            print(f"❌ WebSocket connection failed or error during setup: {e}")

    def _store_trade(self, trade):
        symbol = trade.get("S")
        timestamp_str = trade.get("t")
        price = trade.get("p")
        volume = trade.get("s")

        if not all([symbol, timestamp_str, price is not None, volume is not None]):
            return
        try:
            timestamp = pd.to_datetime(timestamp_str) # Already UTC from Alpaca
        except Exception:
            return

        if symbol not in self.data_cache:
            self.data_cache[symbol] = {"trades": [], "quotes": []} # Initialize for trades and quotes
        
        self.data_cache[symbol]["trades"].append({"timestamp": timestamp, "price": price, "volume": volume})
        if len(self.data_cache[symbol]["trades"]) > 5000: # Limit cache size
            self.data_cache[symbol]["trades"].pop(0)

    # def _store_quote(self, quote): # Example
    #     symbol = quote.get("S")
    #     # ... parse and store quote data ...
    #     if symbol not in self.data_cache:
    #         self.data_cache[symbol] = {"trades": [], "quotes": []}
    #     self.data_cache[symbol]["quotes"].append(...)


    def get_latest_trade(self, symbol):
        cached_symbol_data = self.data_cache.get(symbol)
        if cached_symbol_data and cached_symbol_data["trades"]:
            latest_trade_data = cached_symbol_data["trades"][-1]
            class MockTrade: # Mimic an SDK-like object for app.py compatibility
                def __init__(self, price, timestamp):
                    self.price = price
                    self.timestamp = timestamp # pd.Timestamp
            return MockTrade(price=latest_trade_data['price'], timestamp=latest_trade_data['timestamp'])
        # print(f"Debug: No cached trade data for {symbol} in get_latest_trade.")
        return None

    def _map_timeframe_to_alpaca(self, timeframe_str):
        try:
            if "Min" in timeframe_str:
                minutes = int(timeframe_str.replace("Min", ""))
                return TimeFrame(minutes, TimeFrameUnit.Minute)
            elif "H" in timeframe_str:
                hours = int(timeframe_str.replace("H", ""))
                return TimeFrame(hours, TimeFrameUnit.Hour)
            elif "D" in timeframe_str:
                return TimeFrame.Day # Or TimeFrame(1, TimeFrameUnit.Day)
            else:
                print(f"Unsupported timeframe string: {timeframe_str}, defaulting to 5 Minute.")
                return TimeFrame(5, TimeFrameUnit.Minute)
        except ValueError:
            print(f"Could not parse timeframe_str: {timeframe_str}, defaulting to 5 Minute.")
            return TimeFrame(5, TimeFrameUnit.Minute)

    def get_data(self, tickers, timeframe_str="5Min", limit_per_symbol=200):
        """
        Fetches historical bar data.
        Primary: Alpaca's REST API.
        Fallback: Yahoo Finance (with retries).
        """
        result = {}
        alpaca_tf = self._map_timeframe_to_alpaca(timeframe_str)

        for symbol in tickers:
            df = pd.DataFrame() # Initialize as empty
            # --- Try Alpaca REST API First ---
            try:
                bars_df = self.rest_api.get_bars(
                    symbol,
                    alpaca_tf,
                    limit=limit_per_symbol,
                    adjustment='split' # common adjustment
                ).df
                
                if not bars_df.empty:
                    bars_df.columns = bars_df.columns.str.lower() # Ensure lowercase
                    # Index is already DatetimeIndex (UTC)
                    df = bars_df
                    print(f"SUCCESS: Fetched {len(df)} '{timeframe_str}' bars for {symbol} from Alpaca REST API.")
                else:
                    print(f"INFO: Alpaca REST API returned no data for {symbol} (Timeframe: {timeframe_str}).")
            
            except Exception as e: # Specific Alpaca API errors
                if e.code == 40410000: # Symbol not found
                     print(f"INFO: Symbol {symbol} not found on Alpaca (Code: {e.code}). Will try yfinance.")
                elif e.code == 40010001 and "is not a tradable asset" in str(e): # Not tradable
                     print(f"INFO: Symbol {symbol} is not tradable on Alpaca. Will try yfinance.")
                elif e.code == 42920000: # Rate limit
                    print(f"WARNING: Alpaca API rate limit hit for {symbol}. Try again later or reduce request frequency.")
                    # Not attempting yfinance here as it indicates broader issue.
                else:
                    print(f"WARNING: Alpaca REST API error for {symbol} (Code: {e.code}): {e}. Will try yfinance.")
            except Exception as e: # Other unexpected errors with Alpaca REST
                print(f"WARNING: Unexpected error fetching {symbol} from Alpaca REST API: {e}. Will try yfinance.")

            # --- Fallback to yfinance if Alpaca REST API failed or returned empty ---
            if df.empty: 
                print(f"FALLBACK: Attempting to fetch {symbol} from yfinance (Timeframe: {timeframe_str})...")
                yf_interval = timeframe_str.lower().replace("min", "m").replace("h","h").replace("d","d")
                if yf_interval.endswith('m') and not yf_interval[:-1].isdigit(): yf_interval = f"1{yf_interval}" # Ensure "m" -> "1m"
                
                # Determine yfinance period (heuristic)
                yf_period = "1mo" # Default
                try:
                    if 'm' in yf_interval:
                        minutes_per_bar = int(yf_interval.replace('m',''))
                        total_minutes_needed = minutes_per_bar * limit_per_symbol
                        days_needed = max(2, (total_minutes_needed // (6.5 * 60)) + 2) # min 2 days, add buffer
                        yf_period = f"{min(int(days_needed), 59)}d" # yf 1m/5m interval max ~59-60 days
                    elif 'h' in yf_interval:
                        yf_period = f"{min(limit_per_symbol * 2 // 24 + 2, 729)}d" # Heuristic for hourly
                    elif 'd' in yf_interval:
                        yf_period = f"{limit_per_symbol + 20}d" # Daily, add buffer for non-trading days
                except ValueError: pass # Keep default yf_period

                for attempt in range(3): # Retry yfinance up to 3 times
                    try:
                        temp_df = yf.download(symbol, period=yf_period, interval=yf_interval, progress=False, timeout=10)
                        if not temp_df.empty:
                            temp_df = temp_df.rename(columns=str.lower)
                            temp_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True) # Ensure core cols
                            df = temp_df.tail(limit_per_symbol)
                            print(f"SUCCESS (yfinance attempt {attempt+1}): Fetched {len(df)} bars for {symbol}.")
                            break # Success, exit retry loop
                        else:
                            if attempt < 2:
                                print(f"WARNING (yfinance attempt {attempt+1}): yfinance returned empty for {symbol}. Retrying...")
                                time.sleep(1.5 ** attempt) # Exponential backoff
                            else:
                                print(f"FAIL (yfinance attempt {attempt+1}): yfinance returned empty for {symbol} after retries.")
                    except Exception as yf_e:
                        if attempt < 2:
                            print(f"WARNING (yfinance attempt {attempt+1}): Error fetching {symbol} - {yf_e}. Retrying...")
                            time.sleep(1.5 ** attempt)
                        else:
                            print(f"FAIL (yfinance attempt {attempt+1}): Error fetching {symbol} - {yf_e} after retries.")
            
            result[symbol] = df # Store whatever df was obtained (could be empty)
        return result
