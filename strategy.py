# strategy.py
import pandas as pd
import numpy as np
import strategy_functions as sf # Imports the consolidated logic
from typing import Dict, List

# This STRATEGY_MAP is crucial.
# Keys: User-friendly display names (MUST MATCH app.py's STRATEGY_CATEGORIES items)
# Values: Actual function objects from strategy_functions.py (sf.actual_function_name)
STRATEGY_MAP = {
    # Pattern Recognition
    "Fractal Breakout RSI": sf.strategy_fractal_breakout_rsi,
    "Ross Hook Momentum": sf.strategy_ross_hook_momentum,
    "Hammer on Keltner Volume": sf.strategy_hammer_on_keltner_volume,
    "Hammer Volume": sf.strategy_hammer_volume,
    "RSI Bullish Divergence Candlestick": sf.strategy_rsi_bullish_divergence_candlestick,
    # Momentum
    "Momentum Trading": sf.strategy_momentum,
    "MACD Bullish ADX": sf.strategy_macd_bullish_adx,
    "ADX Rising MFI Surge": sf.strategy_adx_rising_mfi_surge,
    "TRIX OBV": sf.strategy_trix_obv,
    "Vortex ADX": sf.strategy_vortex_adx,
    # Mean Reversion
    "Mean Reversion (RSI)": sf.strategy_mean_reversion,
    "Scalping (Bollinger Bands)": sf.strategy_scalping,
    "MACD RSI Oversold": sf.strategy_macd_rsi_oversold,
    "CCI Reversion": sf.strategy_cci_reversion,
    "Keltner RSI Oversold": sf.strategy_keltner_rsi_oversold,
    "Keltner MFI Oversold": sf.strategy_keltner_mfi_oversold,
    "Bollinger Bounce Volume": sf.strategy_bollinger_bounce_volume,
    "MFI Bollinger": sf.strategy_mfi_bollinger,
    # Trend Following
    "Trend Following (EMA/ADX)": sf.strategy_trend_following,
    "Golden Cross RSI": sf.strategy_golden_cross_rsi,
    "ADX Heikin Ashi": sf.strategy_adx_heikin_ashi,
    "SuperTrend RSI Pullback": sf.strategy_supertrend_rsi_pullback,
    "Ichimoku Basic Combo": sf.strategy_ichimoku_basic_combo,
    "Ichimoku Multi-Line": sf.strategy_ichimoku_multi_line,
    "EMA SAR": sf.strategy_ema_sar,
    # Breakout
    "Breakout Trading": sf.strategy_breakout,
    "Pivot Point (Intraday S/R)": sf.strategy_pivot_point,
    # Volatility
    "Bollinger Upper Break Volume": sf.strategy_bollinger_upper_break_volume,
    "Keltner Middle RSI Divergence": sf.strategy_keltner_middle_rsi_divergence,
    "EMA Ribbon Expansion CMF": sf.strategy_ema_ribbon_expansion_cmf,
    # Volume-Based
    "News Trading (Volatility Spike)": sf.strategy_news,
    "TEMA Cross Volume": sf.strategy_tema_cross_volume,
    "VWAP RSI": sf.strategy_vwap_rsi,
    "VWAP Aroon": sf.strategy_vwap_aroon,
    # Oscillator-Based
    "PSAR RSI": sf.strategy_psar_rsi,
    "RSI EMA Crossover": sf.strategy_rsi_ema_crossover,
    "CCI Bollinger": sf.strategy_cci_bollinger,
    "TSI Resistance Break": sf.strategy_tsi_resistance_break,
    "Awesome Oscillator Divergence MACD": sf.strategy_ao_divergence_macd,
    "Heikin Ashi CMO": sf.strategy_heikin_ashi_cmo,
    # Hybrid/Other
    "Reversal (RSI/MACD)": sf.strategy_reversal,
    "Pullback Trading (EMA)": sf.strategy_pullback,
    "End-of-Day (Intraday Consolidation)": sf.strategy_end_of_day_intraday,
    "EMA Ribbon MACD": sf.strategy_ema_ribbon_macd,
    "Chandelier Exit MACD": sf.strategy_chandelier_exit_macd,
    "Double MA Pullback": sf.strategy_double_ma_pullback,
    "RSI Range Breakout BB": sf.strategy_rsi_range_breakout_bb,
    # Add any unique strategies from Hybrid/Other that aren't already covered above
    # Ensure that every display name used in app.py's STRATEGY_CATEGORIES has a mapping here
    # to a real function in strategy_functions.py
}

# This dictionary should contain default parameters for strategies that need them.
# Keys are the *display names*. Values are dicts of params for the corresponding strategy function.
# (Copied from your original strategy.py)
STRATEGY_DEFAULT_PARAMS = {
    "Momentum Trading": {"rsi_period": 14, "volume_multiplier": 2.0, "rsi_level": 70},
    "Scalping (Bollinger Bands)": {"bb_period": 20, "bb_std": 2.0},
    "Breakout Trading": {"ema_period": 20, "volume_multiplier": 1.5},
    "Mean Reversion (RSI)": {"rsi_period": 14, "rsi_upper": 70, "rsi_lower": 30},
    "News Trading (Volatility Spike)": {"volume_multiplier": 2.5, "price_change_threshold": 0.02},
    "Trend Following (EMA/ADX)": {"ema_short": 9, "ema_long": 21, "adx_period": 14, "adx_threshold": 25},
    "Pivot Point (Intraday S/R)": {"pivot_lookback": 20, "exit_ema_period": 20}, # Adjusted to match your version
    "Reversal (RSI/MACD)": {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
    "Pullback Trading (EMA)": {"ema_short": 9, "ema_long": 21},
    "End-of-Day (Intraday Consolidation)": {"ema_period": 20, "volume_multiplier": 1.5, "price_stability_pct": 0.002},
    "Golden Cross RSI": {"short_ema": 50, "long_ema": 200, "rsi_period": 14, "rsi_level": 50},
    "MACD Bullish ADX": {"fast": 12, "slow": 26, "signal": 9, "adx_period": 14, "adx_level": 25},
    "MACD RSI Oversold": {"fast": 12, "slow": 26, "signal": 9, "rsi_period": 14, "rsi_oversold": 30},
    "ADX Heikin Ashi": {"adx_period": 14, "adx_level": 25},
    "PSAR RSI": {"initial_af": 0.02, "max_af": 0.2, "rsi_period": 14, "rsi_level": 50},
    "VWAP RSI": {"rsi_period": 14, "rsi_level": 50},
    "EMA Ribbon MACD": {"ema_lengths": [8, 13, 21, 34, 55], "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
    "ADX Rising MFI Surge": {"adx_period": 14, "adx_level": 25, "mfi_period": 14, "mfi_surge_level": 80},
    "Fractal Breakout RSI": {"fractal_lookback": 2, "rsi_period": 14, "rsi_level": 50},
    "Chandelier Exit MACD": {"atr_period": 22, "atr_mult": 3.0, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
    "SuperTrend RSI Pullback": {"atr_length": 10, "factor": 3.0, "rsi_period": 14, "rsi_pullback_level": 50},
    "TEMA Cross Volume": {"short_tema_len": 10, "long_tema_len": 30, "vol_ma_period": 20, "vol_factor": 1.5},
    "TSI Resistance Break": {"fast": 13, "slow": 25, "signal": 13, "resistance_period": 20},
    "TRIX OBV": {"trix_period": 15, "trix_signal": 9},
    "Awesome Oscillator Divergence MACD": {"ao_fast": 5, "ao_slow": 34, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "div_lookback": 14},
    "Heikin Ashi CMO": {"cmo_period": 14, "cmo_level": 0},
    "CCI Bollinger": {"cci_period": 20, "cci_extreme": -100, "bb_period": 20, "bb_std": 2.0},
    "CCI Reversion": {"cci_period": 20, "cci_extreme_low": -150, "cci_revert_level": -100},
    "Keltner RSI Oversold": {"kc_length": 20, "kc_atr_length": 10, "kc_mult": 2.0, "rsi_period": 14, "rsi_oversold": 30},
    "Keltner MFI Oversold": {"kc_length": 20, "kc_atr_length": 10, "kc_mult": 2.0, "mfi_period": 14, "mfi_oversold": 20},
    "Double MA Pullback": {"short_ma_len": 20, "long_ma_len": 50, "ma_type": "ema"},
    "Bollinger Bounce Volume": {"bb_period": 20, "bb_std": 2.0, "vol_ma_period": 20, "vol_factor": 1.5},
    "RSI Range Breakout BB": {"rsi_period": 14, "rsi_low": 40, "rsi_high": 60, "bb_period": 20, "bb_std": 2.0},
    "Keltner Middle RSI Divergence": {"kc_length": 20, "kc_atr_length": 10, "kc_mult": 2.0, "rsi_period": 14, "div_lookback": 14},
    "Hammer on Keltner Volume": {"kc_length": 20, "kc_atr_length": 10, "kc_mult": 2.0, "vol_ma_period": 20, "vol_factor": 1.2},
    "Bollinger Upper Break Volume": {"bb_period": 20, "bb_std": 2.0, "vol_ma_period": 20, "vol_factor": 1.5},
    "RSI EMA Crossover": {"rsi_period": 14, "rsi_ma_period": 10, "ema_period": 50},
    "VWAP Aroon": {"aroon_period": 14, "aroon_level": 70},
    "Vortex ADX": {"vortex_period": 14, "adx_period": 14, "adx_trend_level": 25},
    "EMA Ribbon Expansion CMF": {"ema_lengths": [8, 13, 21, 34, 55], "expansion_threshold": 0.005, "cmf_period": 20, "cmf_level": 0.0},
    "Ross Hook Momentum": {"ross_lookback": 10, "momentum_period": 10, "momentum_level": 0},
    "RSI Bullish Divergence Candlestick": {"rsi_period": 14, "div_lookback": 14},
    "Stochastic Oversold EMA Filter": {"stoch_k": 14, "stoch_d": 3, "stoch_smooth_k": 3, "ema_period": 200, "stoch_oversold": 20},
    "Ichimoku Basic Combo": {"tenkan_period": 9, "kijun_period": 26, "senkou_period": 52},
    "Ichimoku Multi-Line": {"tenkan_period": 9, "kijun_period": 26, "senkou_period": 52},
    "EMA SAR": {"ema_period": 50, "initial_af": 0.02, "max_af": 0.2},
    "MFI Bollinger": {"mfi_period": 14, "bb_period": 20, "bb_std": 2.0, "mfi_buy_level": 20},
    "Hammer Volume": {"vol_ma_period": 20, "vol_factor": 1.5},
    "CMF EMA Ribbon": {"cmf_period": 20, "ema_lengths": [20, 50], "cmf_level": 0.0},
}

def run_strategies(df: pd.DataFrame, strategy_names: List[str]) -> Dict[str, str]:
    results = {}
    if df.empty:
        for name in strategy_names:
            results[name] = "ERROR: Input DataFrame is empty"
        return results

    # Ensure DataFrame has necessary columns (lowercase)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        for name in strategy_names:
            results[name] = f"ERROR: DataFrame missing columns: {', '.join(missing_cols)}"
        return results
    
    # Ensure columns are numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[required_cols].isnull().any().any(): # Check for NaNs after coercion
        for name in strategy_names:
             results[name] = "ERROR: Non-numeric data or NaNs in OHLCV columns"
        return results


    for name in strategy_names:
        if name in STRATEGY_MAP:
            strategy_function_to_call = STRATEGY_MAP[name]
            params_for_function = STRATEGY_DEFAULT_PARAMS.get(name, {})
            
            try:
                # It's crucial that functions in sf return a DataFrame with boolean signal columns
                df_with_signals = strategy_function_to_call(df.copy(), **params_for_function)
                
                signal = "NONE"
                # Prioritize specific "Entry_Buy" or "Entry_Sell" if they exist
                latest_buy_signal = False
                if f"{name.replace(' ', '').replace('(', '').replace(')','').replace('/','')}_Entry_Buy" in df_with_signals.columns and \
                   not df_with_signals[f"{name.replace(' ', '').replace('(', '').replace(')','').replace('/','')}_Entry_Buy"].empty:
                    latest_buy_signal = bool(df_with_signals[f"{name.replace(' ', '').replace('(', '').replace(')','').replace('/','')}_Entry_Buy"].iloc[-1])
                
                latest_sell_signal = False
                if f"{name.replace(' ', '').replace('(', '').replace(')','').replace('/','')}_Entry_Sell" in df_with_signals.columns and \
                   not df_with_signals[f"{name.replace(' ', '').replace('(', '').replace(')','').replace('/','')}_Entry_Sell"].empty:
                    latest_sell_signal = bool(df_with_signals[f"{name.replace(' ', '').replace('(', '').replace(')','').replace('/','')}_Entry_Sell"].iloc[-1])

                # Generic "Entry" column check if specific buy/sell aren't primary
                if not latest_buy_signal and not latest_sell_signal:
                    generic_entry_cols = [col for col in df_with_signals.columns if "Entry" in col and col not in df.columns]
                    for col in generic_entry_cols:
                        if not df_with_signals[col].empty and isinstance(df_with_signals[col].iloc[-1], (bool, np.bool_)) and df_with_signals[col].iloc[-1]:
                            # Assuming generic "Entry" implies a BUY for this scanner's purpose
                            latest_buy_signal = True 
                            break
                
                if latest_buy_signal and latest_sell_signal:
                    signal = "BUY/SELL CONFLICT"
                elif latest_buy_signal:
                    signal = "BUY"
                elif latest_sell_signal:
                    signal = "SELL"
                
                results[name] = signal

            except Exception as e:
                import traceback
                print(f"--- ERROR RUNNING STRATEGY: {name} ---")
                print(f"Params: {params_for_function}")
                traceback.print_exc()
                results[name] = f"ERROR: {str(e)}"
        else:
            results[name] = f"ERROR: Strategy '{name}' not in STRATEGY_MAP"
    return results