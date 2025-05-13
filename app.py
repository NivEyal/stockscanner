# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from alpaca_data import AlpacaData # Assuming this class handles Alpaca API calls
from strategy import run_strategies # Assuming this module contains strategy logic
import asyncio
import threading
import requests
import datetime
import os

# --- Load API Keys from Streamlit Secrets ---
ALPACA_API_KEY_ID = st.secrets.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY")
FMP_API_KEY = st.secrets.get("FMP_API_KEY")

# --- Set Alpaca credentials as environment variables ---
if ALPACA_API_KEY_ID:
    os.environ["APCA_API_KEY_ID"] = ALPACA_API_KEY_ID
if ALPACA_SECRET_KEY:
    os.environ["APCA_API_SECRET_KEY"] = ALPACA_SECRET_KEY
APCA_BASE_URL_FROM_SECRET = st.secrets.get("APCA_API_BASE_URL")
if APCA_BASE_URL_FROM_SECRET:
    os.environ["APCA_API_BASE_URL"] = APCA_BASE_URL_FROM_SECRET
    print(f"DEBUG app.py: Set env APCA_API_BASE_URL to: {os.environ.get('APCA_API_BASE_URL')}")
else:
    print(f"DEBUG app.py: APCA_API_BASE_URL not found in secrets. AlpacaData will use its default.")

# --------------------------------------
# Config
# --------------------------------------
STRATEGY_CATEGORIES = {
    "Pattern Recognition": [
        "Fractal Breakout RSI", "Ross Hook Momentum", 
        "Hammer Volume", "RSI Bullish Divergence Candlestick"
    ],
    "Momentum": [
        "Momentum Trading", "MACD Bullish ADX", "ADX Rising MFI Surge", "TRIX OBV", "Vortex ADX"
    ],
    "Mean Reversion": [
        "Mean Reversion (RSI)", "Scalping (Bollinger Bands)", "MACD RSI Oversold", "CCI Reversion",
         "Bollinger Bounce Volume", "MFI Bollinger"
    ],
    "Trend Following": [
        "Trend Following (EMA/ADX)", "Golden Cross RSI", "ADX Heikin Ashi", "SuperTrend RSI Pullback",
        "Ichimoku Basic Combo", "Ichimoku Multi-Line", "EMA SAR"
    ],
    "Breakout": [
        "Breakout Trading", "Pivot Point (Intraday S/R)"
    ],
    "Volatility": [
        "Bollinger Upper Break Volume", "EMA Ribbon Expansion CMF"
    ],
    "Volume-Based": [
        "News Trading (Volatility Spike)", "TEMA Cross Volume", "Bollinger Bounce Volume",
         "Hammer Volume", "ADX Rising MFI Surge",
        "MFI Bollinger", "TRIX OBV", "VWAP RSI", "VWAP Aroon", "EMA Ribbon Expansion CMF",
    ],
    "Oscillator-Based": [
        "PSAR RSI", "RSI EMA Crossover",  "CCI Bollinger",
         "Awesome Oscillator Divergence MACD", "Heikin Ashi CMO",
         "MFI Bollinger",
    ],
    "News/Event-Driven": [
        "News Trading (Volatility Spike)"
    ],
    "Hybrid/Other": [
        "Reversal (RSI/MACD)", "Pullback Trading (EMA)", "End-of-Day (Intraday Consolidation)",
        "VWAP RSI",  "Chandelier Exit MACD", "Heikin Ashi CMO", 
        "Double MA Pullback", "RSI Range Breakout BB", "VWAP Aroon", "EMA Ribbon Expansion CMF",
         "MACD Bullish ADX", "ADX Rising MFI Surge", "Fractal Breakout RSI",
        "Bollinger Upper Break Volume",
        "RSI EMA Crossover",  "Vortex ADX", "Ross Hook Momentum",
        "RSI Bullish Divergence Candlestick", "Ichimoku Basic Combo",
        "Ichimoku Multi-Line", "EMA SAR", "MFI Bollinger", "Hammer Volume",
    ]
}
# Get a flat list of all unique strategy names from the configuration
ALL_UNIQUE_STRATEGY_NAMES = sorted(list(set(s_name for cat_strats in STRATEGY_CATEGORIES.values() for s_name in cat_strats)))


# --- Helper Functions ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_top_10_volume_fmp():
    if not FMP_API_KEY:
        st.sidebar.warning("FMP API Key not found. Using fallback.")
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "JPM", "V", "JNJ"]

    try:
        url = "https://financialmodelingprep.com/api/v3/stock_market/actives"
        params = {"apikey": FMP_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        filtered_data = [
            {
                "symbol": item["symbol"],
                "price": item.get("price", None),
                "change_pct": item.get("changesPercentage", None)
            }
            for item in data
            if isinstance(item.get("symbol"), str)
               and '.' not in item["symbol"]
               and item["symbol"].isalnum()
               and len(item["symbol"]) <= 5
               and item.get("price", 0) >= 1.7
        ]

        # Store full metadata in session for dashboard use
        st.session_state.fmp_metadata = {d["symbol"]: d for d in filtered_data}
        return [d["symbol"] for d in filtered_data[:10]]
    except requests.exceptions.RequestException as e:
        st.error(f"FMP API request error: {e}")
    except Exception as e:
        st.error(f"Error fetching top volume from FMP: {e}")
    return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "JPM", "V", "JNJ"]


def show_top_volume_dashboard(tickers_list, alpaca_client: AlpacaData):
    st.subheader("📊 Market Dashboard (Data from Alpaca)")
    data_for_df = []

    if not tickers_list:
        st.warning("No tickers provided for dashboard.")
        return

    try:
        dashboard_data_frames = alpaca_client.get_data(
            tickers_list,
            timeframe_str="1Min",
            limit_per_symbol=400
        )

        for ticker in tickers_list:
            df = dashboard_data_frames.get(ticker)

            if df is None or df.empty:
                latest_trade = alpaca_client.get_latest_trade(ticker)
                if latest_trade and hasattr(latest_trade, 'price'):
                    data_for_df.append({
                        "Ticker": ticker,
                        "Price": f"${last_price:.2f}",
                        "Change %": f"{change_pct:.2f}%" if isinstance(change_pct, (float, int)) else "N/A"
                    })
                else:
                    st.warning(f"No 1Min data or latest trade for {ticker} from Alpaca for dashboard.")
                continue
            
            df.columns = df.columns.str.lower()
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'], utc=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                 df.index = pd.to_datetime(df.index, utc=True)
            else:
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                else:
                    df.index = df.index.tz_convert('UTC')
            
            latest_day_in_data = df.index.normalize().max()
            df_today = df[df.index.date == latest_day_in_data.date()].copy()

            if df_today.empty:
                if not df.empty:
                    last_price = df["close"].iloc[-1]
                    data_for_df.append({"Ticker": ticker, "Price": f"${last_price:.2f}", "Change %": "N/A (Prev. Close)"})
                continue

            last_price = df_today["close"].iloc[-1]
            if "open" in df_today.columns and not df_today.empty:
                open_price = df_today["open"].iloc[0]
                change_pct = ((last_price - open_price) / open_price) * 100 if open_price != 0 else 0
                data_for_df.append({"Ticker": ticker, "Price": f"${last_price:.2f}", "Change %": f"{change_pct:.2f}%"})
            else:
                 data_for_df.append({"Ticker": ticker, "Price": f"${last_price:.2f}", "Change %": "N/A (No Open)"})

    except Exception as e:
        st.error(f"Error fetching/processing dashboard data from Alpaca: {e}")
        for ticker in tickers_list:
            data_for_df.append({"Ticker": ticker, "Price": "N/A", "Change %": "N/A (Error)"})

    if data_for_df:
        df_out = pd.DataFrame(data_for_df)
        if "Change %" in df_out.columns:
            df_out['Change % num'] = pd.to_numeric(df_out['Change %'].astype(str).str.rstrip('%'), errors='coerce')
            df_out = df_out.sort_values(by="Change % num", ascending=False, na_position='last').drop(columns=["Change % num"])
        st.dataframe(df_out, use_container_width=True)
    else:
        st.warning("⚠️ No data available for dashboard from Alpaca.")

def start_websocket_thread_loop(loop, alpaca_client, ws_tickers_list):
    asyncio.set_event_loop(loop)
    try:
        if not ws_tickers_list:
            print("WebSocket Thread: No tickers for WebSocket stream.")
            return
        print(f"WebSocket Thread: Starting WebSocket for: {', '.join(ws_tickers_list)}")
        loop.run_until_complete(alpaca_client.start_websocket(ws_tickers_list))
    except RuntimeError as e:
        if "There is no current event loop in thread" in str(e) or "cannot be used to create new tasks" in str(e):
            print(f"WebSocket Thread Error: Event loop issue in thread: {e}. Loop was: {loop}")
        else:
            print(f"WebSocket Thread runtime error: {e}")
    except Exception as e:
        print(f"WebSocket Thread encountered an error: {e}")
    finally:
        print("WebSocket Thread: Event loop finished.")

# --- UI Setup ---
st.set_page_config(page_title="🚀 Alpaca Strategy Scanner", layout="wide")
st.title("🚀 Alpaca Strategy Scanner")

# --- Initialize Alpaca Client ---
if not ALPACA_API_KEY_ID or not ALPACA_SECRET_KEY:
    st.error("Alpaca API Key ID or Secret Key is not configured. Please set them in Streamlit secrets (secrets.toml).")
    st.code("""
    # Example secrets.toml:
    ALPACA_API_KEY = "YOUR_KEY_ID"
    ALPACA_SECRET_KEY = "YOUR_SECRET_KEY"
    FMP_API_KEY = "YOUR_FMP_KEY"
    """)
    st.stop()

try:
    alpaca = AlpacaData()
    st.sidebar.success("Alpaca client initialized.")
except Exception as e:
    st.error(f"Failed to initialize AlpacaData client: {e}")
    st.stop()

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Scanner Settings")

# --- NEW Strategy Selection UI using Expanders and Checkboxes ---
st.sidebar.markdown("#### 1. Select Strategies")

# Initialize session state for individual strategy selections
# This dictionary will hold the selection state (True/False) for each unique strategy name.
if 'individual_strategy_selections' not in st.session_state:
    st.session_state.individual_strategy_selections = {}
    # Default selection logic: Select all strategies in "Momentum" category on first load
    momentum_strategies_list = STRATEGY_CATEGORIES.get("Momentum", []) # Get the list once
    for s_name in ALL_UNIQUE_STRATEGY_NAMES:
        # Check if the current strategy name is in the Momentum category's list
        is_in_momentum = s_name in momentum_strategies_list # CORRECTED LINE
        st.session_state.individual_strategy_selections[s_name] = is_in_momentum
else:
    # On subsequent runs, ensure all defined strategies exist in session state.
    # Add new ones as False, remove obsolete ones.
    # This handles cases where STRATEGY_CATEGORIES is updated in the code.
    current_session_keys = set(st.session_state.individual_strategy_selections.keys())
    all_defined_keys = set(ALL_UNIQUE_STRATEGY_NAMES)

    # Add new strategies (defined in STRATEGY_CATEGORIES but not in session_state) as False
    for s_name in all_defined_keys:
        if s_name not in st.session_state.individual_strategy_selections:
            st.session_state.individual_strategy_selections[s_name] = False # Default new to False
            
    # Remove any strategies from session_state that are no longer in STRATEGY_CATEGORIES
    # (This part might be optional if you prefer to keep old selections even if a strategy is removed from config)
    keys_to_remove = current_session_keys - all_defined_keys
    for s_key_to_remove in keys_to_remove:
        del st.session_state.individual_strategy_selections[s_key_to_remove]


# Global "Select All" / "Clear All" buttons for all strategies
col_main_all, col_main_none = st.sidebar.columns(2)
if col_main_all.button("✅ Select ALL Strategies", key="select_all_strategies_overall_btn", use_container_width=True):
    for s_name in st.session_state.individual_strategy_selections.keys():
        st.session_state.individual_strategy_selections[s_name] = True
    st.rerun()

if col_main_none.button("❌ Clear ALL Selections", key="clear_all_strategies_overall_btn", use_container_width=True):
    for s_name in st.session_state.individual_strategy_selections.keys():
        st.session_state.individual_strategy_selections[s_name] = False
    st.rerun()
st.sidebar.markdown("---")


# Iterate through categories and create an expander for each
for category_name, strategies_in_category_list in STRATEGY_CATEGORIES.items():
    # Calculate how many strategies are selected within this specific category
    num_selected_in_category = sum(
        1 for s_name in strategies_in_category_list 
        if st.session_state.individual_strategy_selections.get(s_name, False)
    )
    
    # Make "Momentum" category expanded by default, others collapsed
    is_expanded_default = (category_name == "Momentum")

    with st.sidebar.expander(f"{category_name} ({num_selected_in_category} selected)", expanded=is_expanded_default):
        
        # "Select All / None" buttons for strategies *within this category*
        col_cat_all, col_cat_none = st.columns(2)
        if col_cat_all.button(f"Select All in {category_name}", key=f"select_all_cat_{category_name.replace(' ', '_')}", use_container_width=True):
            for strat_name in strategies_in_category_list:
                if strat_name in st.session_state.individual_strategy_selections: 
                    st.session_state.individual_strategy_selections[strat_name] = True
            st.rerun()

        if col_cat_none.button(f"Deselect All in {category_name}", key=f"deselect_all_cat_{category_name.replace(' ', '_')}", use_container_width=True):
            for strat_name in strategies_in_category_list:
                if strat_name in st.session_state.individual_strategy_selections:
                    st.session_state.individual_strategy_selections[strat_name] = False
            st.rerun()
        
        st.markdown("<div style='margin-top: 5px; margin-bottom: 5px; border-bottom: 1px solid #444;'></div>", unsafe_allow_html=True)

        # List each strategy in the category with a checkbox
        for strategy_name in strategies_in_category_list:
            # Ensure the strategy exists in the central selection dict (should be by initialization)
            if strategy_name not in st.session_state.individual_strategy_selections:
                st.session_state.individual_strategy_selections[strategy_name] = False 

            # Create a unique key for the checkbox widget itself, but link its value to the global strategy name
            # This ensures that if a strategy appears in multiple categories, its checkbox reflects the single source of truth
            checkbox_key = f"checkbox_{category_name.replace(' ', '_')}_{strategy_name.replace(' ', '_').replace('/', '_')}"
            
            current_selection_state = st.session_state.individual_strategy_selections.get(strategy_name, False)
            
            new_selection_state = st.checkbox(
                strategy_name,
                value=current_selection_state,
                key=checkbox_key 
            )
            
            if new_selection_state != current_selection_state:
                st.session_state.individual_strategy_selections[strategy_name] = new_selection_state
                st.rerun()


# Collect all selected strategies for the scan
selected_strategies_for_run = sorted([
    s_name for s_name, is_sel in st.session_state.individual_strategy_selections.items() if is_sel
])

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Strategies Selected for Scan:")
if selected_strategies_for_run:
    st.sidebar.caption(f"Total: {len(selected_strategies_for_run)} strategies selected.")
    for strat_name in selected_strategies_for_run:
        st.sidebar.markdown(f"• _{strat_name}_")
else:
    st.sidebar.caption("No strategies selected yet.")
st.sidebar.markdown("---")
# --- End of New Strategy Selection UI ---


# Ticker Selection
st.sidebar.markdown("#### 2. Select Tickers") 
use_fmp_source = st.sidebar.checkbox("Use Top Volume from FMP (updates hourly)", value=True, key="use_fmp")
ticker_input_placeholder = "e.g.,\nAAPL\nMSFT\nGOOG"
manual_ticker_input = st.sidebar.text_area("📥 Enter tickers manually (1 per line):",
                                         placeholder=ticker_input_placeholder, height=100, key="manual_tickers")

current_tickers = []
if use_fmp_source:
    if 'fmp_tickers' not in st.session_state or st.sidebar.button("🔄 Refresh FMP List", key="refresh_fmp"):
        with st.spinner("Fetching top volume tickers from FMP..."):
            st.session_state.fmp_tickers = get_top_10_volume_fmp()
    current_tickers = st.session_state.get('fmp_tickers', [])
    if not current_tickers and FMP_API_KEY:
        st.sidebar.warning("FMP ticker list is empty. Check FMP API or try refreshing.")
else:
    current_tickers = [t.strip().upper() for t in manual_ticker_input.split("\n") if t.strip()]

if not current_tickers:
    if not use_fmp_source and not manual_ticker_input:
        st.sidebar.info("Enter tickers manually, or select FMP option.")
    elif use_fmp_source:
        pass

# Scan Button
st.sidebar.markdown("---")
scan_button = st.sidebar.button("🚀 Scan Selected Strategies",
                                disabled=not current_tickers or not selected_strategies_for_run,
                                key="scan_button", use_container_width=True)

# --- WebSocket Management ---
if current_tickers:
    ws_thread_key = "ws_thread"
    ws_loop_key = "ws_loop"
    ws_tickers_key = "ws_tickers_streaming"

    if ws_thread_key not in st.session_state: st.session_state[ws_thread_key] = None
    if ws_loop_key not in st.session_state: st.session_state[ws_loop_key] = None
    if ws_tickers_key not in st.session_state: st.session_state[ws_tickers_key] = []

    start_new_ws = False
    if st.session_state[ws_thread_key] is None or not st.session_state[ws_thread_key].is_alive():
        start_new_ws = True
    elif set(st.session_state.get(ws_tickers_key, [])) != set(current_tickers):
        start_new_ws = True
        st.sidebar.warning("Ticker list changed. Re-initializing WebSocket.")

    if start_new_ws:
        if st.session_state[ws_loop_key] is None or st.session_state[ws_loop_key].is_closed():
            st.session_state[ws_loop_key] = asyncio.new_event_loop()
            print(f"DEBUG: Created NEW event loop for WebSocket: {id(st.session_state[ws_loop_key])}")
        else:
            print(f"DEBUG: REUSING existing event loop: {id(st.session_state[ws_loop_key])}, closed={st.session_state[ws_loop_key].is_closed()}")

        st.session_state[ws_thread_key] = threading.Thread(
            target=start_websocket_thread_loop,
            args=(st.session_state[ws_loop_key], alpaca, current_tickers),
            daemon=True
        )
        st.session_state[ws_thread_key].start()
        st.session_state[ws_tickers_key] = current_tickers
        st.sidebar.caption(f"WebSocket initiated for {len(current_tickers)} tickers.")
else:
    pass


# --- Main Area Display ---
if not current_tickers:
    st.warning("⚠️ No tickers selected. Please select/enter tickers and strategies using the sidebar.")
    st.stop()

if alpaca:
    show_top_volume_dashboard(current_tickers, alpaca)

# --- Strategy Scanning Logic ---
if scan_button:
    if not selected_strategies_for_run:
        st.warning("No strategies selected. Please choose strategies in the sidebar.")
    elif not current_tickers:
        st.warning("No tickers selected for scanning.")
    else:
        st.markdown("---")
        st.subheader(f"📡 Strategy Scan Results")
        st.caption(f"Scanning {len(selected_strategies_for_run)} strategies: {', '.join(selected_strategies_for_run)}")

        progress_bar = st.progress(0)
        total_tickers_to_scan = len(current_tickers)
        
        scan_timeframe = "5Min"
        scan_limit = 300
        
        st.write(f"Fetching historical data ({scan_timeframe} bars, {scan_limit} periods) for {total_tickers_to_scan} tickers from Alpaca...")
        
        with st.spinner("Downloading historical data for scanning..."):
            all_historical_data = alpaca.get_data(
                current_tickers,
                timeframe_str=scan_timeframe,
                limit_per_symbol=scan_limit
            )

        results_container = st.container()

        for i, ticker in enumerate(current_tickers):
            df_ticker = all_historical_data.get(ticker)
            
            progress_text = f"Scanning {ticker} ({i+1}/{total_tickers_to_scan})"
            progress_bar.progress((i + 1) / total_tickers_to_scan, text=progress_text)

            if df_ticker is None or df_ticker.empty:
                with results_container:
                    st.warning(f"⚠️ No historical data from Alpaca for {ticker}. Skipping.")
                continue

            df_ticker.columns = df_ticker.columns.str.lower()
            if 'timestamp' in df_ticker.columns:
                df_ticker.index = pd.to_datetime(df_ticker['timestamp'], utc=True)
            elif not isinstance(df_ticker.index, pd.DatetimeIndex):
                df_ticker.index = pd.to_datetime(df_ticker.index, utc=True)
            else:
                if df_ticker.index.tz is None: df_ticker.index = df_ticker.index.tz_localize('UTC')
                else: df_ticker.index = df_ticker.index.tz_convert('UTC')
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df_ticker.columns]
            if missing_cols:
                with results_container:
                    st.warning(f"⚠️ Data for {ticker} missing: {', '.join(missing_cols)}. Skipping.")
                continue
            
            try:
                for col in required_cols:
                    df_ticker[col] = pd.to_numeric(df_ticker[col], errors='coerce')
                df_ticker.dropna(subset=required_cols, inplace=True)
                if df_ticker.empty or len(df_ticker) < 20:
                    raise ValueError(f"Insufficient data for {ticker} after cleaning ({len(df_ticker)} rows).")
            except Exception as e:
                with results_container:
                    st.warning(f"⚠️ Data for {ticker} had issues: {e}. Skipping.")
                continue

            try:
                signals = run_strategies(df_ticker.copy(), selected_strategies_for_run)
                latest_bar = df_ticker.iloc[-1]
                signal_texts = [f"{s_name}: {s_val}" for s_name, s_val in signals.items() if s_val != "NONE" and s_val is not None]

                with results_container:
                    st.markdown(f"--- \n ### {ticker.upper()}")
                    col1, col2, col3 = st.columns([1.5, 1.5, 5])
                    col1.metric("Last Close", f"${latest_bar['close']:.2f}",
                                help=f"Timestamp: {latest_bar.name.strftime('%Y-%m-%d %H:%M')}")
                    col2.metric("Last Volume", f"{int(latest_bar['volume']):,}")

                    fig = go.Figure(data=[go.Candlestick(x=df_ticker.index,
                                    open=df_ticker['open'], high=df_ticker['high'],
                                    low=df_ticker['low'], close=df_ticker['close'])])
                    fig.update_layout(
                        height=200, margin=dict(l=10, r=10, t=5, b=5),
                        xaxis_rangeslider_visible=False,
                    )
                    col3.plotly_chart(fig, use_container_width=True)

                    if signal_texts:
                        for sig_text in signal_texts:
                            st.success(f"Signal: {sig_text}")
                    else:
                        st.info("No specific strategy signals detected based on current selections.")
            except Exception as e:
                 with results_container:
                    st.error(f"Error running strategies for {ticker}: {e}")
            
        progress_bar.empty()
        st.success("Scan complete!")
        
# --- Always Show Donation Link ---
st.markdown("---")
st.markdown(
    '<p style="font-size:20px; font-weight: bold; text-align: center;">'
    'Like this tool? Consider supporting its development: '
    '<a href="https://paypal.me/niveyal" target="_blank">☕ Buy me a coffee (PayPal.me)</a>'
    '</p>',
    unsafe_allow_html=True
)

if st.sidebar.button("🔄 Refresh Page & Clear Cache", key="refresh_page_clear_cache", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.rerun()