# strategy_functions.py
import pandas as pd
import pandas_ta as ta
import numpy as np

# --- Helper Functions (Consolidated Here) ---
def crossed_above_level(series, level):
    if len(series) < 2 or series.isnull().all():
        return pd.Series([False] * len(series), index=series.index)
    shifted_series = series.shift(1)
    # Handle NaN at the beginning carefully
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is not None and shifted_series.loc[first_valid_idx] is pd.NA:
        if series.loc[first_valid_idx] >= level: # Already above or at level
             shifted_series.loc[first_valid_idx] = level -1 # Treat as if it was below
        else:
             shifted_series.loc[first_valid_idx] = level + 1 # Treat as if it was above
    # Fill remaining NaNs if any (e.g. if series starts with NaNs)
    shifted_series = shifted_series.fillna(method='bfill').fillna(level + 1) # Ensure no NaN comparison

    return (shifted_series < level) & (series >= level)

def crossed_below_level(series, level):
    if len(series) < 2 or series.isnull().all():
        return pd.Series([False] * len(series), index=series.index)
    shifted_series = series.shift(1)
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is not None and shifted_series.loc[first_valid_idx] is pd.NA:
        if series.loc[first_valid_idx] <= level:
            shifted_series.loc[first_valid_idx] = level + 1
        else:
            shifted_series.loc[first_valid_idx] = level - 1
    shifted_series = shifted_series.fillna(method='bfill').fillna(level - 1)
    return (shifted_series > level) & (series <= level)

def crossed_above_series(series1, series2):
    if len(series1) < 2 or len(series2) < 2 or series1.isnull().all() or series2.isnull().all():
        return pd.Series([False] * len(series1), index=series1.index)
    s1, s2 = series1.align(series2, join='inner') # Ensure alignment
    if len(s1) < 2 : return pd.Series([False] * len(series1), index=series1.index).reindex(series1.index).fillna(False)

    s1_shifted = s1.shift(1)
    s2_shifted = s2.shift(1)
    # Where both current and previous values are valid
    valid_mask = s1.notna() & s2.notna() & s1_shifted.notna() & s2_shifted.notna()
    cross_condition = pd.Series(False, index=s1.index)
    cross_condition[valid_mask] = (s1_shifted[valid_mask] <= s2_shifted[valid_mask]) & (s1[valid_mask] > s2[valid_mask])
    return cross_condition.reindex(series1.index).fillna(False)

def crossed_below_series(series1, series2):
    if len(series1) < 2 or len(series2) < 2 or series1.isnull().all() or series2.isnull().all():
        return pd.Series([False] * len(series1), index=series1.index)
    s1, s2 = series1.align(series2, join='inner')
    if len(s1) < 2 : return pd.Series([False] * len(series1), index=series1.index).reindex(series1.index).fillna(False)
    
    s1_shifted = s1.shift(1)
    s2_shifted = s2.shift(1)
    valid_mask = s1.notna() & s2.notna() & s1_shifted.notna() & s2_shifted.notna()
    cross_condition = pd.Series(False, index=s1.index)
    cross_condition[valid_mask] = (s1_shifted[valid_mask] >= s2_shifted[valid_mask]) & (s1[valid_mask] < s2[valid_mask])
    return cross_condition.reindex(series1.index).fillna(False)

def detect_divergence(price_series, indicator_series, lookback=14, type='bullish'):
    min_data_needed = lookback + 1
    if len(price_series) < min_data_needed or len(indicator_series) < min_data_needed or price_series.isnull().all() or indicator_series.isnull().all():
         return pd.Series([False] * len(price_series), index=price_series.index)

    price_rolling_min = price_series.rolling(window=lookback, min_periods=lookback).min()
    price_rolling_max = price_series.rolling(window=lookback, min_periods=lookback).max()
    indicator_rolling_min = indicator_series.rolling(window=lookback, min_periods=lookback).min()
    indicator_rolling_max = indicator_series.rolling(window=lookback, min_periods=lookback).max()

    is_divergence = pd.Series(False, index=price_series.index)
    valid_indices = price_rolling_min.dropna().index.intersection(indicator_rolling_min.dropna().index)
    valid_indices = valid_indices.intersection(price_rolling_min.shift(1).dropna().index).intersection(indicator_rolling_min.shift(1).dropna().index)
    if valid_indices.empty: return is_divergence.reindex(price_series.index).fillna(False)

    if type == 'bullish':
        price_ll = price_rolling_min.loc[valid_indices] < price_rolling_min.shift(1).loc[valid_indices]
        indicator_hl = indicator_rolling_min.loc[valid_indices] > indicator_rolling_min.shift(1).loc[valid_indices]
        is_divergence.loc[valid_indices] = price_ll & indicator_hl
    elif type == 'bearish':
        price_hh = price_rolling_max.loc[valid_indices] > price_rolling_max.shift(1).loc[valid_indices]
        indicator_lh = indicator_rolling_max.loc[valid_indices] < indicator_rolling_max.shift(1).loc[valid_indices]
        is_divergence.loc[valid_indices] = price_hh & indicator_lh
    return is_divergence.reindex(price_series.index).fillna(False)

def detect_fractal_high(high_series, lookback=2):
    min_data_needed = 2 * lookback + 1
    if len(high_series) < min_data_needed: return pd.Series([False] * len(high_series), index=high_series.index)
    is_fractal = pd.Series(True, index=high_series.index)
    for i in range(1, lookback + 1):
        is_fractal &= (high_series > high_series.shift(i).fillna(-np.inf)) & (high_series > high_series.shift(-i).fillna(-np.inf))
    return is_fractal.fillna(False)

def detect_fractal_low(low_series, lookback=2):
    min_data_needed = 2 * lookback + 1
    if len(low_series) < min_data_needed: return pd.Series([False] * len(low_series), index=low_series.index)
    is_fractal = pd.Series(True, index=low_series.index)
    for i in range(1, lookback + 1):
        is_fractal &= (low_series < low_series.shift(i).fillna(np.inf)) & (low_series < low_series.shift(-i).fillna(np.inf))
    return is_fractal.fillna(False)

def detect_ross_hook(df_ohlc, lookback=10): # Renamed df to df_ohlc to avoid clash
    df = df_ohlc.copy() # Work on a copy
    min_data_needed = lookback * 2 + 2
    if len(df) < min_data_needed: return pd.Series([False] * len(df), index=df.index)
    recent_high_col = f"Recent_High_RH_{lookback}"
    df[recent_high_col] = df["high"].rolling(window=lookback).max().shift(1)
    longer_lookback = lookback * 2
    prev_min_low_col = f"Prev_Min_Low_RH_{longer_lookback}"
    df[prev_min_low_col] = df["low"].rolling(window=longer_lookback).min().shift(1)
    breakout_cond = crossed_above_series(df["close"], df[recent_high_col].fillna(-np.inf))
    higher_low_cond = df["low"] > df[prev_min_low_col].fillna(np.inf)
    return (breakout_cond & higher_low_cond).fillna(False)

def _add_empty_signals(df, base_name, buy=True, sell=False): # DEFINED HERE
    df_ = df.copy()
    if buy:
        df_[f"{base_name}_Entry_Buy"] = False
        df_[f"{base_name}_Exit_Buy"] = False
    if sell:
        df_[f"{base_name}_Entry_Sell"] = False
        df_[f"{base_name}_Exit_Sell"] = False
    if not buy and not sell:
        df_[f"{base_name}_Entry_Generic"] = False
        df_[f"{base_name}_Exit_Generic"] = False
    return df_
# --- Strategy Implementations (from your strategy.py) ---
# Each function should return a DataFrame with new signal columns (e.g., StrategyName_Entry_Buy)

def crossed_above_level(series, level):
    """Returns True when series crosses above a fixed level."""
    # Needs at least 2 data points
    if len(series) < 2 or series.isnull().all():
        return pd.Series([False] * len(series), index=series.index)
    # Use .fillna with a value guaranteed *not* to trigger a cross on the first bar
    # If the first valid value is already >= level, series.shift(1) is NaN.
    # Filling with level - 1 prevents a false cross signal on the first bar.
    shifted_series = series.shift(1).fillna(level - 1) if series.first_valid_index() is not None and series.iloc[0] >= level else series.shift(1).fillna(level + 1)

    return (shifted_series < level) & (series >= level)

def crossed_below_level(series, level):
    """Returns True when series crosses below a fixed level."""
    # Needs at least 2 data points
    if len(series) < 2 or series.isnull().all():
        return pd.Series([False] * len(series), index=series.index)
    # Use .fillna with a value guaranteed *not* to trigger a cross on the first bar
    # If the first valid value is already <= level, series.shift(1) is NaN.
    # Filling with level + 1 prevents a false cross signal on the first bar.
    shifted_series = series.shift(1).fillna(level + 1) if series.first_valid_index() is not None and series.iloc[0] <= level else series.shift(1).fillna(level - 1)
    return (shifted_series > level) & (series <= level)

def crossed_above_series(series1, series2):
    """Returns True when series1 crosses above series2."""
    # Needs at least 2 data points for shifted values
    if len(series1) < 2 or len(series2) < 2 or series1.isnull().all() or series2.isnull().all():
        return pd.Series([False] * len(series1), index=series1.index)

    # Align indices and get shifted values
    s1 = series1.align(series2, join='inner')[0]
    s2 = series1.align(series2, join='inner')[1]
    s1_shifted = s1.shift(1)
    s2_shifted = s2.shift(1)

    # Ensure we only evaluate where both current and shifted series have valid data
    # Or handle NaNs appropriately at the start of the series
    # For a robust cross, typically require valid data for current and previous bars.
    # A simple check for NaNs in both shifted series on the current bar is sufficient.
    valid_condition_indices = s1.dropna().index.intersection(s2.dropna().index)
    # Need to also ensure shifted values were not NaN for a 'cross' to be meaningful
    valid_condition_indices = valid_condition_indices.intersection(s1_shifted.dropna().index).intersection(s2_shifted.dropna().index)


    if valid_condition_indices.empty:
        return pd.Series([False] * len(series1), index=series1.index).reindex(series1.index).fillna(False)


    cross_condition = pd.Series(False, index=s1.index)
    # Condition: s1 was <= s2 in the previous period AND s1 is > s2 in the current period
    cross_condition.loc[valid_condition_indices] = (s1_shifted.loc[valid_condition_indices] <= s2_shifted.loc[valid_condition_indices]) & \
                                                   (s1.loc[valid_condition_indices] > s2.loc[valid_condition_indices])

    return cross_condition.reindex(series1.index).fillna(False) # Reindex back to original index length

def crossed_below_series(series1, series2):
    """Returns True when series1 crosses below series2."""
     # Needs at least 2 data points for shifted values
    if len(series1) < 2 or len(series2) < 2 or series1.isnull().all() or series2.isnull().all():
        return pd.Series([False] * len(series1), index=series1.index)

    s1 = series1.align(series2, join='inner')[0]
    s2 = series1.align(series2, join='inner')[1]
    s1_shifted = s1.shift(1)
    s2_shifted = s2.shift(1)

    valid_condition_indices = s1.dropna().index.intersection(s2.dropna().index)
    valid_condition_indices = valid_condition_indices.intersection(s1_shifted.dropna().index).intersection(s2_shifted.dropna().index)

    if valid_condition_indices.empty:
        return pd.Series([False] * len(series1), index=series1.index).reindex(series1.index).fillna(False)

    cross_condition = pd.Series(False, index=s1.index)
    # Condition: s1 was >= s2 in the previous period AND s1 is < s2 in the current period
    cross_condition.loc[valid_condition_indices] = (s1_shifted.loc[valid_condition_indices] >= s2_shifted.loc[valid_condition_indices]) & \
                                                   (s1.loc[valid_condition_indices] < s2.loc[valid_condition_indices])

    return cross_condition.reindex(series1.index).fillna(False) # Reindex back to original index length


def detect_divergence(price_series, indicator_series, lookback=14, type='bullish'):
    """
    Detects bullish or bearish divergence. Simplified.
    Bullish: Price makes lower lows, indicator makes higher lows.
    Bearish: Price makes higher highs, indicator makes lower highs.

    Note: This is a simplified detection focused on rolling min/max comparison.
    True divergence detection is more complex, involving finding specific swing points.
    This version might give many false positives or miss true divergences.
    It checks if the *current* rolling min/max price is lower/higher than the *previous*
    rolling min/max, and the *current* rolling min/max indicator is higher/lower.
    """
    if price_series.isnull().all() or indicator_series.isnull().all():
         return pd.Series([False] * len(price_series), index=price_series.index)

    # Ensure minimum data for rolling window + shift
    min_data_needed = lookback + 1 # For rolling window and shift(1)
    if len(price_series) < min_data_needed or len(indicator_series) < min_data_needed:
         # print(f"  Not enough data ({len(price_series)}) for divergence detection (lookback={lookback}), need at least {min_data_needed}")
         return pd.Series([False] * len(price_series), index=price_series.index)


    # Using min_periods=lookback requires a full window for the rolling min/max.
    # This is crucial for divergence which looks for *established* highs/lows.
    price_rolling_min = price_series.rolling(window=lookback, min_periods=lookback).min()
    price_rolling_max = price_series.rolling(window=lookback, min_periods=lookback).max()
    indicator_rolling_min = indicator_series.rolling(window=lookback, min_periods=lookback).min()
    indicator_rolling_max = indicator_series.rolling(window=lookback, min_periods=lookback).max()

    is_divergence = pd.Series(False, index=price_series.index)

    # Check only where current bar is valid and rolling data + shifted rolling data is available
    valid_indices = price_rolling_min.dropna().index.intersection(indicator_rolling_min.dropna().index)
    valid_indices = valid_indices.intersection(price_rolling_min.shift(1).dropna().index) # Ensure previous rolling data exists
    valid_indices = valid_indices.intersection(indicator_rolling_min.shift(1).dropna().index)


    if valid_indices.empty:
        return pd.Series([False] * len(price_series), index=price_series.index)


    if type == 'bullish':
        # Bullish divergence: Price makes Lower Low, Indicator makes Higher Low
        # Simplified check: Is the current rolling min of price lower than the previous rolling min?
        # And is the current rolling min of the indicator higher than the previous rolling min?
        price_ll = price_rolling_min.loc[valid_indices] < price_rolling_min.shift(1).loc[valid_indices]
        indicator_hl = indicator_rolling_min.loc[valid_indices] > indicator_rolling_min.shift(1).loc[valid_indices]
        is_divergence.loc[valid_indices] = price_ll & indicator_hl
    elif type == 'bearish':
        # Bearish divergence: Price makes Higher High, Indicator makes Lower High
        # Simplified check: Is the current rolling max of price higher than the previous rolling max?
        # And is the current rolling max of the indicator lower than the previous rolling max?
        price_hh = price_rolling_max.loc[valid_indices] > price_rolling_max.shift(1).loc[valid_indices]
        indicator_lh = indicator_rolling_max.loc[valid_indices] < indicator_rolling_max.shift(1).loc[valid_indices]
        is_divergence.loc[valid_indices] = price_hh & indicator_lh

    # Handle cases where calculation window isn't met at the start
    return is_divergence.reindex(price_series.index).fillna(False)


def detect_fractal_high(high_series, lookback=2):
    """Detects a fractal high: high is higher than `lookback` periods on both sides."""
    # Requires at least (2 * lookback + 1) data points for the central bar to be compared
    min_data_needed = 2 * lookback + 1
    if len(high_series) < min_data_needed:
         return pd.Series([False] * len(high_series), index=high_series.index)

    # Ensure index supports shifting or handle explicitly
    if not isinstance(high_series.index, pd.DatetimeIndex):
         # If not datetime index, simple shift might be unreliable at ends.
         # For robustness, manual indexing or ensure DatetimeIndex is best.
         # Assuming DatetimeIndex for Alpaca data.
         pass # Proceed with shift


    is_fractal = pd.Series(True, index=high_series.index)
    for i in range(1, lookback + 1):
        # Use .fillna(-np.inf) when checking if current high is GREATER than shifted values
        # This correctly treats NaNs (at start/end of series due to shift) as not being higher.
        is_fractal &= (high_series > high_series.shift(i).fillna(-np.inf)) & (high_series > high_series.shift(-i).fillna(-np.inf))
    return is_fractal.fillna(False)

def detect_fractal_low(low_series, lookback=2):
    """Detects a fractal low: low is lower than `lookback` periods on both sides."""
    # Requires at least (2 * lookback + 1) data points
    min_data_needed = 2 * lookback + 1
    if len(low_series) < min_data_needed:
         return pd.Series([False] * len(low_series), index=low_series.index)

    if not isinstance(low_series.index, pd.DatetimeIndex):
         pass # Proceed with shift

    is_fractal = pd.Series(True, index=low_series.index)
    for i in range(1, lookback + 1):
        # Use .fillna(np.inf) when checking if current low is LESS than shifted values
        # This correctly treats NaNs (at start/end of series due to shift) as not being lower.
        is_fractal &= (low_series < low_series.shift(i).fillna(np.inf)) & (low_series < low_series.shift(-i).fillna(np.inf))
    return is_fractal.fillna(False)

def detect_ross_hook(df, lookback=10):
    """Detects a Ross Hook: A breakout after a 1-2-3 formation and a shallow pullback. Simplified."""
    # This is a very simplified interpretation.
    # Needs enough data for rolling windows and shifts
    # Rolling(lookback) max needs lookback. Shift(1) needs +1. Rolling(lookback*2) min needs lookback*2. Shift(1) needs +1.
    # Max needed: lookback*2 + 1 + 1 = 2*lookback + 2
    min_data_needed = lookback * 2 + 2
    if len(df) < min_data_needed:
         # print(f"  Not enough data ({len(df)}) for detect_ross_hook, need at least {min_data_needed}")
         return pd.Series([False] * len(df), index=df.index)

    # Simplified: Breakout above a recent high after establishing a higher low structure.
    # Recent high (proxy for point 2 of 1-2-3) - shifted by 1 to represent the level *before* the potential breakout bar
    recent_high_col = f"Recent_High_RH_{lookback}"
    df[recent_high_col] = df["high"].rolling(window=lookback).max().shift(1)

    # Higher low approximation (proxy for point 3 being higher than point 1)
    # Check if current low is higher than a minimum low over a longer past period - shifted by 1
    longer_lookback = lookback * 2
    prev_min_low_col = f"Prev_Min_Low_RH_{longer_lookback}"
    df[prev_min_low_col] = df["low"].rolling(window=longer_lookback).min().shift(1) # min low of previous 2*lookback bars

    # Condition check applies to the *current* bar
    # Breakout above the recent high level
    breakout_cond = crossed_above_series(df["close"], df[recent_high_col].fillna(-np.inf)) # Treat NaN recent_high as infinitely low

    # Higher low confirmation: the current low is higher than the previous significant low (prev_min_low)
    # This condition is checked on the current bar, comparing its low to the calculated previous low level
    higher_low_cond = df["low"] > df[prev_min_low_col].fillna(np.inf) # Treat NaN prev_min_low as infinitely high

    # A very simplified Ross Hook signal: Bullish breakout above recent high accompanied by a 'higher low' structure
    # This still lacks the precise 1-2-3 point identification and shallow pullback check.
    df["Ross_Hook_Signal"] = breakout_cond & higher_low_cond
    # Clean up intermediate columns if desired, but often left for inspection
    # df = df.drop(columns=[recent_high_col, prev_min_low_col], errors='ignore') # errors='ignore' if cols weren't created due to lack of data

    return df["Ross_Hook_Signal"].fillna(False)


# --- Strategy Implementations ---
# Each function takes df (OHLCV+Volume) and optional params, returns df with signal columns added.
# Signal columns should be named like "[StrategyName]_Entry_Buy", "[StrategyName]_Exit_Buy", etc.
# They should contain boolean values (True where signal occurs).

# 1. Momentum Trading Strategy
def strategy_momentum(df, rsi_period=14, volume_multiplier=2.0, rsi_level=70):
    df_ = df.copy()
    # Ensure enough data for indicators
    min_data_needed = max(rsi_period, 20) + 1 # RSI needs period, Vol Avg needs 20, crossed_above needs +1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_momentum, need at least {min_data_needed}")
         df_["Momentum_Entry"] = False
         df_["Momentum_Exit"] = False
         return df_

    rsi_col = f"RSI_{rsi_period}"
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)
    df_["Volume_Avg"] = df_["volume"].rolling(window=20).mean()

    if rsi_col not in df_.columns or "Volume_Avg" not in df_.columns or df_[rsi_col].isnull().all() or df_["Volume_Avg"].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_momentum.")
        df_["Momentum_Entry"] = False
        df_["Momentum_Exit"] = False
        return df_


    entry_cond1 = crossed_above_level(df_[rsi_col], rsi_level)
    entry_cond2 = df_["volume"] > volume_multiplier * df_["Volume_Avg"]
    df_["Momentum_Entry"] = entry_cond1 & entry_cond2

    df_["Momentum_Exit"] = crossed_below_level(df_[rsi_col], rsi_level - 10) # Exit e.g. RSI drops below 60
    return df_

# 2. Scalping Strategy (Bollinger Bands)
def strategy_scalping(df, bb_period=20, bb_std=2.0):
    df_ = df.copy()
     # Ensure enough data for indicators
    min_data_needed = bb_period + 1 # BB needs period, crossed_above needs +1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_scalping, need at least {min_data_needed}")
         df_["Scalping_Entry_Buy"] = False
         df_["Scalping_Entry_Sell"] = False
         df_["Scalping_Exit_Buy"] = False
         df_["Scalping_Exit_Sell"] = False
         return df_

    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)
    if bbands is None or bbands.empty:
        # print(f"  BBands calculation failed for strategy_scalping.")
        df_["Scalping_Entry_Buy"] = df_["Scalping_Entry_Sell"] = df_["Scalping_Exit_Buy"] = df_["Scalping_Exit_Sell"] = False
        return df_
    df_ = df_.join(bbands) # Joins BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0

    # Ensure BB columns exist before accessing
    bbl_col = f"BBL_{bb_period}_{bb_std}"
    bbu_col = f"BBU_{bb_period}_{bb_std}"
    bbm_col = f"BBM_{bb_period}_{bb_std}"

    if not all(col in df_.columns for col in [bbl_col, bbu_col, bbm_col]) or df_[bbl_col].isnull().all():
        # print(f"  Missing expected BB columns or they contain NaNs for strategy_scalping. Found: {df_.columns.tolist()}")
        df_["Scalping_Entry_Buy"] = False
        df_["Scalping_Entry_Sell"] = False
        df_["Scalping_Exit_Buy"] = False
        df_["Scalping_Exit_Sell"] = False
        return df_


    entry_cond_buy = crossed_above_series(df_["close"], df_[bbl_col])
    entry_cond_sell = crossed_below_series(df_["close"], df_[bbu_col])

    df_["Scalping_Entry_Buy"] = entry_cond_buy
    df_["Scalping_Entry_Sell"] = entry_cond_sell

    df_["Scalping_Exit_Buy"] = crossed_above_series(df_["close"], df_[bbm_col])
    df_["Scalping_Exit_Sell"] = crossed_below_series(df_["close"], df_[bbm_col])
    return df_

# 3. Breakout Trading Strategy
def strategy_breakout(df, ema_period=20, volume_multiplier=1.5):
    df_ = df.copy()
     # Ensure enough data for indicators
    min_data_needed = ema_period + 1 # EMA needs period, crossed_above needs +1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_breakout, need at least {min_data_needed}")
         df_["Breakout_Entry"] = False
         df_["Breakout_Exit"] = False
         return df_

    ema_col = f"EMA_{ema_period}"
    df_[ema_col] = ta.ema(df_["close"], length=ema_period)
    df_["Volume_Avg_Short"] = df_["volume"].rolling(window=ema_period).mean()

    if ema_col not in df_.columns or df_[ema_col].isnull().all() or df_["Volume_Avg_Short"].isnull().all():
         # print(f"  Indicator calculation failed or produced NaNs for strategy_breakout.")
         df_["Breakout_Entry"] = False
         df_["Breakout_Exit"] = False
         return df_


    entry_cond1 = crossed_above_series(df_["close"], df_[ema_col])
    entry_cond2 = df_["volume"] > volume_multiplier * df_["Volume_Avg_Short"]
    df_["Breakout_Entry"] = entry_cond1 & entry_cond2

    df_["Breakout_Exit"] = crossed_below_series(df_["close"], df_[ema_col])
    return df_

# 4. Mean Reversion Strategy
def strategy_mean_reversion(df, rsi_period=14, rsi_upper=70, rsi_lower=30):
    df_ = df.copy()
     # Ensure enough data for indicators
    min_data_needed = rsi_period + 1 # RSI needs period, crossed_above/below needs +1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_mean_reversion, need at least {min_data_needed}")
         df_["MeanReversion_Entry_Buy"] = False
         df_["MeanReversion_Entry_Sell"] = False
         df_["MeanReversion_Exit_Buy"] = False
         df_["MeanReversion_Exit_Sell"] = False
         return df_

    rsi_col = f"RSI_{rsi_period}"
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)

    if rsi_col not in df_.columns or df_[rsi_col].isnull().all():
         # print(f"  RSI calculation failed or produced NaNs for strategy_mean_reversion.")
         df_["MeanReversion_Entry_Buy"] = False
         df_["MeanReversion_Entry_Sell"] = False
         df_["MeanReversion_Exit_Buy"] = False
         df_["MeanReversion_Exit_Sell"] = False
         return df_


    entry_cond_buy = crossed_above_level(df_[rsi_col], rsi_lower)
    entry_cond_sell = crossed_below_level(df_[rsi_col], rsi_upper)

    df_["MeanReversion_Entry_Buy"] = entry_cond_buy
    df_["MeanReversion_Entry_Sell"] = entry_cond_sell

    df_["MeanReversion_Exit_Buy"] = crossed_above_level(df_[rsi_col], 50)
    df_["MeanReversion_Exit_Sell"] = crossed_below_level(df_[rsi_col], 50)
    return df_

# 5. News Trading Strategy
def strategy_news(df, volume_multiplier=2.5, price_change_threshold=0.02):
    df_ = df.copy()
     # Ensure enough data for indicators
    min_data_needed = 5 + 1 # Vol Avg needs 5, pct_change needs +1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_news, need at least {min_data_needed}")
         df_["News_Entry"] = False
         df_["News_Exit"] = False
         return df_

    df_["Volume_Avg_VShort"] = df_["volume"].rolling(window=5).mean()
    df_["Price_Change_1m"] = df_["close"].pct_change() # Uses previous close

    if df_["Volume_Avg_VShort"].isnull().all() or df_["Price_Change_1m"].isnull().all():
         # print(f"  Indicator calculation failed or produced NaNs for strategy_news.")
         df_["News_Entry"] = False
         df_["News_Exit"] = False
         return df_


    entry_cond1 = df_["volume"] > volume_multiplier * df_["Volume_Avg_VShort"]
    entry_cond2 = df_["Price_Change_1m"].abs() > price_change_threshold
    df_["News_Entry"] = entry_cond1 & entry_cond2 # This is a generic entry, could be buy or sell

    df_["News_Exit"] = (df_["Price_Change_1m"].abs() < (price_change_threshold / 4)) # Price change subsides
    # Ensure Exit signal is only true *after* an Entry signal could have occurred
    # This simple strategy doesn't track state, the exit condition is just a general volatility decrease.
    # A proper backtest would manage position state. For a scanner, this exit just flags
    # when the volatility spike is potentially over.
    return df_

# 6. Trend Following Strategy
def strategy_trend_following(df, ema_short=9, ema_long=21, adx_period=14, adx_threshold=25):
    df_ = df.copy()
     # Ensure enough data for indicators
    min_data_needed = max(ema_long, adx_period) + 1 # Max of periods + 1 for crosses
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_trend_following, need at least {min_data_needed}")
         df_["TrendFollowing_Entry_Buy"] = False
         df_["TrendFollowing_Exit_Buy"] = False
         return df_


    ema_short_col = f"EMA_{ema_short}"
    ema_long_col = f"EMA_{ema_long}"
    adx_col = f"ADX_{adx_period}"

    df_[ema_short_col] = ta.ema(df_["close"], length=ema_short)
    df_[ema_long_col] = ta.ema(df_["close"], length=ema_long)
    adx_data = ta.adx(df_["high"], df_["low"], df_["close"], length=adx_period)

    if adx_data is None or adx_data.empty or ema_short_col not in df_.columns or ema_long_col not in df_.columns or df_[ema_short_col].isnull().all() or df_[ema_long_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_trend_following (EMA/ADX data check before join).")
        df_["TrendFollowing_Entry_Buy"] = False
        df_["TrendFollowing_Exit_Buy"] = False
        return df_

    df_ = df_.join(adx_data[[adx_col]]) # Join only ADX column

    if adx_col not in df_.columns or df_[adx_col].isnull().all(): # Final check after join
        # print(f"  ADX column '{adx_col}' not found or produced NaNs after join.")
        df_["TrendFollowing_Entry_Buy"] = False
        df_["TrendFollowing_Exit_Buy"] = False
        return df_


    entry_cond_buy1 = df_["close"] > df_[ema_short_col]
    entry_cond_buy2 = df_[ema_short_col] > df_[ema_long_col]
    entry_cond_buy3 = df_[adx_col] > adx_threshold
    df_["TrendFollowing_Entry_Buy"] = entry_cond_buy1 & entry_cond_buy2 & entry_cond_buy3

    df_["TrendFollowing_Exit_Buy"] = crossed_below_series(df_["close"], df_[ema_short_col])
    return df_

# 7. Pivot Point Trading Strategy (Intraday S/R)
def strategy_pivot_point(df, pivot_lookback=20, exit_ema_period=20):
    df_ = df.copy()
     # Ensure enough data for indicators
    min_data_needed = max(pivot_lookback, exit_ema_period) + 1 # Rolling max/min + EMA + crossed
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_pivot_point, need at least {min_data_needed}")
         df_["PivotPoint_Entry_Buy"] = False
         df_["PivotPoint_Entry_Sell"] = False
         df_["PivotPoint_Exit_Buy"] = False
         df_["PivotPoint_Exit_Sell"] = False
         return df_


    recent_high_col = f"Recent_High_{pivot_lookback}"
    recent_low_col = f"Recent_Low_{pivot_lookback}"
    exit_ema_col = f"EMA_{exit_ema_period}_Exit"

    df_[recent_high_col] = df_["high"].rolling(window=pivot_lookback).max().shift(1)
    df_[recent_low_col] = df_["low"].rolling(window=pivot_lookback).min().shift(1)
    df_[exit_ema_col] = ta.ema(df_["close"], length=exit_ema_period)

    if recent_high_col not in df_.columns or recent_low_col not in df_.columns or exit_ema_col not in df_.columns or df_[recent_high_col].isnull().all() or df_[recent_low_col].isnull().all() or df_[exit_ema_col].isnull().all():
         # print(f"  Indicator calculation failed or produced NaNs for strategy_pivot_point.")
         df_["PivotPoint_Entry_Buy"] = False
         df_["PivotPoint_Entry_Sell"] = False
         df_["PivotPoint_Exit_Buy"] = False
         df_["PivotPoint_Exit_Sell"] = False
         return df_


    df_["PivotPoint_Entry_Buy"] = crossed_above_series(df_["close"], df_[recent_high_col].fillna(-np.inf)) # Treat NaN as infinitely low for cross up
    df_["PivotPoint_Entry_Sell"] = crossed_below_series(df_["close"], df_[recent_low_col].fillna(np.inf)) # Treat NaN as infinitely high for cross down

    df_["PivotPoint_Exit_Buy"] = crossed_below_series(df_["close"], df_[exit_ema_col])
    df_["PivotPoint_Exit_Sell"] = crossed_above_series(df_["close"], df_[exit_ema_col])
    return df_

# 8. Reversal Trading Strategy
def strategy_reversal(df, rsi_period=14, rsi_oversold=30, rsi_overbought=70, macd_fast=12, macd_slow=26, macd_signal=9):
    df_ = df.copy()
     # Ensure enough data for indicators
    min_data_needed = max(rsi_period, macd_slow + macd_signal) + 1 # Max of periods + 1 for crosses
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_reversal, need at least {min_data_needed}")
         df_["Reversal_Entry_Buy"] = False
         df_["Reversal_Entry_Sell"] = False
         df_["Reversal_Exit_Buy"] = False
         df_["Reversal_Exit_Sell"] = False
         return df_

    rsi_col = f"RSI_{rsi_period}"
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)
    macd_data = ta.macd(df_["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)

    macd_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
    macds_col = f"MACDs_{fast}_{slow}_{signal}"

    if macd_data is None or macd_data.empty or rsi_col not in df_.columns or df_[rsi_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_reversal (RSI/MACD data check before join).")
        df_["Reversal_Entry_Buy"] = False
        df_["Reversal_Entry_Sell"] = False
        df_["Reversal_Exit_Buy"] = False
        df_["Reversal_Exit_Sell"] = False
        return df_
    df_ = df_.join(macd_data)

    if macd_col not in df_.columns or macds_col not in df_.columns or df_[macd_col].isnull().all() or df_[macds_col].isnull().all(): # Final check after join
        # print(f"  MACD columns '{macd_col}', '{macds_col}' not found or produced NaNs after join.")
        df_["Reversal_Entry_Buy"] = False
        df_["Reversal_Entry_Sell"] = False
        df_["Reversal_Exit_Buy"] = False
        df_["Reversal_Exit_Sell"] = False
        return df_


    entry_cond_buy1 = df_[rsi_col] < rsi_oversold
    entry_cond_buy2 = crossed_above_series(df_[macd_col], df_[macds_col])
    df_["Reversal_Entry_Buy"] = entry_cond_buy1 & entry_cond_buy2

    entry_cond_sell1 = df_[rsi_col] > rsi_overbought
    entry_cond_sell2 = crossed_below_series(df_[macd_col], df_[macds_col])
    df_["Reversal_Entry_Sell"] = entry_cond_sell1 & entry_cond_sell2

    df_["Reversal_Exit_Buy"] = crossed_below_series(df_[macd_col], df_[macds_col]) | (df_[rsi_col] > 50)
    df_["Reversal_Exit_Sell"] = crossed_above_series(df_[macd_col], df_[macds_col]) | (df_[rsi_col] < 50)
    return df_

# 9. Pullback Trading Strategy
def strategy_pullback(df, ema_short=9, ema_long=21):
    df_ = df.copy()
     # Ensure enough data for indicators
    min_data_needed = ema_long + 2 # EMA needs period, shifted low needs +1, crossed needs +1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_pullback, need at least {min_data_needed}")
         df_["Pullback_Entry_Buy"] = False
         df_["Pullback_Exit_Buy"] = False
         return df_


    ema_short_col = f"EMA_{ema_short}"
    ema_long_col = f"EMA_{ema_long}"
    df_[ema_short_col] = ta.ema(df_["close"], length=ema_short)
    df_[ema_long_col] = ta.ema(df_["close"], length=ema_long)

    if ema_short_col not in df_.columns or ema_long_col not in df_.columns or df_[ema_short_col].isnull().all() or df_[ema_long_col].isnull().all():
         # print(f"  EMA calculation failed or produced NaNs for strategy_pullback.")
         df_["Pullback_Entry_Buy"] = False
         df_["Pullback_Exit_Buy"] = False
         return df_


    is_uptrend = df_[ema_short_col] > df_[ema_long_col]
    # Pulled back to EMA short: low of *previous* bar was <= EMA, and close of *current* bar is > EMA
    # Ensure shifted low and shifted EMA are not NaN
    shifted_low = df_["low"].shift(1).fillna(np.inf) # Treat NaN low as infinitely high for <= check
    shifted_ema = df_[ema_short_col].shift(1).fillna(-np.inf) # Treat NaN EMA as infinitely low for <= check

    pulled_back_to_ema_short = (shifted_low <= shifted_ema) & \
                               (df_["close"] > df_[ema_short_col])

    df_["Pullback_Entry_Buy"] = is_uptrend & pulled_back_to_ema_short

    df_["Pullback_Exit_Buy"] = crossed_below_series(df_["close"], df_[ema_short_col])
    # Note: The provided exit also includes EMA cross, but the function name implies price cross exit.
    # Leaving as is, but this might exit *before* EMA cross if price crosses EMA first.
    return df_

# 10. End-of-Day Trading Strategy (Intraday Consolidation)
def strategy_end_of_day_intraday(df, ema_period=20, volume_multiplier=1.5, price_stability_pct=0.002):
    df_ = df.copy()
     # Ensure enough data for indicators
    min_data_needed = ema_period + 2 # EMA needs period, pct_change needs +1, crossed needs +1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_end_of_day_intraday, need at least {min_data_needed}")
         df_["EndOfDay_Intraday_Entry"] = False
         df_["EndOfDay_Intraday_Exit"] = False
         return df_

    ema_col = f"EMA_{ema_period}"
    df_[ema_col] = ta.ema(df_["close"], length=ema_period)
    df_["Volume_Avg_Short"] = df_["volume"].rolling(window=ema_period).mean()
    df_["Price_Change_1m"] = df_["close"].pct_change() # Uses previous close


    if ema_col not in df_.columns or df_[ema_col].isnull().all() or df_["Volume_Avg_Short"].isnull().all() or df_["Price_Change_1m"].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_end_of_day_intraday.")
        df_["EndOfDay_Intraday_Entry"] = False
        df_["EndOfDay_Intraday_Exit"] = False
        return df_


    price_is_stable = (df_["close"] / df_[ema_col]).between(1 - price_stability_pct, 1 + price_stability_pct)
    volume_is_decent = df_["volume"] > volume_multiplier * df_["Volume_Avg_Short"]

    # Entry: Price stable and above EMA, with decent volume.
    df_["EndOfDay_Intraday_Entry"] = price_is_stable & volume_is_decent & (df_["close"] > df_[ema_col])

    # Exit: Price volatility increases OR price crosses below EMA
    df_["EndOfDay_Intraday_Exit"] = (df_["Price_Change_1m"].abs() > (price_stability_pct * 2)) | \
                                   crossed_below_series(df_["close"], df_[ema_col])
    return df_

# --- Additional Strategies (11-50) ---

# 11. Golden Cross RSI
def strategy_golden_cross_rsi(df, short_ema=50, long_ema=200, rsi_period=14, rsi_level=50):
    df_ = df.copy()
    min_data_needed = max(long_ema, rsi_period) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_golden_cross_rsi, need at least {min_data_needed}")
         df_["GC_RSI_Entry"] = False
         df_["GC_RSI_Exit"] = False
         return df_

    short_ema_col = f"EMA_{short_ema}"
    long_ema_col = f"EMA_{long_ema}"
    rsi_col = f"RSI_{rsi_period}"

    df_[short_ema_col] = ta.ema(df_["close"], length=short_ema)
    df_[long_ema_col] = ta.ema(df_["close"], length=long_ema)
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)

    if short_ema_col not in df_.columns or long_ema_col not in df_.columns or rsi_col not in df_.columns or df_[short_ema_col].isnull().all() or df_[long_ema_col].isnull().all() or df_[rsi_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_golden_cross_rsi.")
        df_["GC_RSI_Entry"] = False
        df_["GC_RSI_Exit"] = False
        return df_

    entry_cond1 = crossed_above_series(df_[short_ema_col], df_[long_ema_col])
    entry_cond2 = df_[rsi_col] > rsi_level
    df_["GC_RSI_Entry"] = entry_cond1 & entry_cond2

    df_["GC_RSI_Exit"] = crossed_below_series(df_[short_ema_col], df_[long_ema_col])
    return df_

# 12. MACD Bullish ADX
def strategy_macd_bullish_adx(df, fast=12, slow=26, signal=9, adx_period=14, adx_level=25):
    df_ = df.copy()
    min_data_needed = max(slow + signal, adx_period) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_macd_bullish_adx, need at least {min_data_needed}")
         df_["MACD_ADX_Entry"] = False
         df_["MACD_ADX_Exit"] = False
         return df_


    mac_data = ta.macd(df_["close"], fast=fast, slow=slow, signal=signal)
    adx_data = ta.adx(df_["high"], df_["low"], df_["close"], length=adx_period)

    macd_col = f"MACD_{fast}_{slow}_{signal}"
    macds_col = f"MACDs_{fast}_{slow}_{signal}"
    adx_col = f"ADX_{adx_period}"

    if mac_data is None or adx_data is None or mac_data.empty or adx_data.empty:
        # print(f"  Indicator calculation failed for strategy_macd_bullish_adx (MACD/ADX data check before join).")
        df_["MACD_ADX_Entry"] = False
        df_["MACD_ADX_Exit"] = False
        return df_

    df_ = df_.join(mac_data)
    df_ = df_.join(adx_data[[adx_col]]) # Join only ADX column

    if macd_col not in df_.columns or macds_col not in df_.columns or adx_col not in df_.columns or df_[macd_col].isnull().all() or df_[macds_col].isnull().all() or df_[adx_col].isnull().all():
        # print(f"  Missing expected columns or they produced NaNs after join for strategy_macd_bullish_adx.")
        df_["MACD_ADX_Entry"] = False
        df_["MACD_ADX_Exit"] = False
        return df_


    entry_cond1 = crossed_above_series(df_[macd_col], df_[macds_col])
    entry_cond2 = df_[adx_col] > adx_level
    df_["MACD_ADX_Entry"] = entry_cond1 & entry_cond2

    df_["MACD_ADX_Exit"] = crossed_below_series(df_[macd_col], df_[macds_col])
    return df_

# 13. MACD RSI Oversold
def strategy_macd_rsi_oversold(df, fast=12, slow=26, signal=9, rsi_period=14, rsi_oversold=30):
    df_ = df.copy()
    min_data_needed = max(slow + signal, rsi_period) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_macd_rsi_oversold, need at least {min_data_needed}")
         df_["MACD_RSI_OS_Entry"] = False
         df_["MACD_RSI_OS_Exit"] = False
         return df_

    mac_data = ta.macd(df_["close"], fast=fast, slow=slow, signal=signal)
    rsi_col = f"RSI_{rsi_period}"
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)

    macd_col = f"MACD_{fast}_{slow}_{signal}"
    macds_col = f"MACDs_{fast}_{slow}_{signal}"

    if mac_data is None or mac_data.empty or rsi_col not in df_.columns or df_[rsi_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_macd_rsi_oversold (RSI/MACD data check before join).")
        df_["MACD_RSI_OS_Entry"] = False
        df_["MACD_RSI_OS_Exit"] = False
        return df_
    df_ = df_.join(mac_data)

    if macd_col not in df_.columns or macds_col not in df_.columns or df_[macd_col].isnull().all() or df_[macds_col].isnull().all():
        # print(f"  Missing expected columns or they produced NaNs after join for strategy_macd_rsi_oversold.")
        df_["MACD_RSI_OS_Entry"] = False
        df_["MACD_RSI_OS_Exit"] = False
        return df_


    entry_cond1 = crossed_above_level(df_[rsi_col], rsi_oversold)
    entry_cond2 = df_[macd_col] > df_[macds_col] # MACD line is currently above signal line
    df_["MACD_RSI_OS_Entry"] = entry_cond1 & entry_cond2

    df_["MACD_RSI_OS_Exit"] = crossed_below_series(df_[macd_col], df_[macds_col]) | \
                             (df_[rsi_col] > 70)
    return df_

# 14. ADX Heikin Ashi
def strategy_adx_heikin_ashi(df, adx_period=14, adx_level=25):
    df_ = df.copy()
    # HA needs high, low, open, close for each bar. ADX needs its period.
    # HA color change check needs +1 bar. Cross check needs +1 bar.
    min_data_needed = adx_period + 2
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_adx_heikin_ashi, need at least {min_data_needed}")
         df_["ADX_HA_Entry"] = False
         df_["ADX_HA_Exit"] = False
         return df_


    adx_data = ta.adx(df_["high"], df_["low"], df_["close"], length=adx_period)
    ha_data = ta.ha(df_["open"], df_["high"], df_["low"], df_["close"])

    adx_col = f"ADX_{adx_period}"

    if adx_data is None or ha_data is None or adx_data.empty or ha_data.empty:
        # print(f"  Indicator calculation failed for strategy_adx_heikin_ashi (ADX/HA data check before join).")
        df_["ADX_HA_Entry"] = False
        df_["ADX_HA_Exit"] = False
        return df_

    df_ = df_.join(adx_data[[adx_col]])
    df_["HA_Open"] = ha_data["HA_open"]
    df_["HA_Close"] = ha_data["HA_close"]

    if adx_col not in df_.columns or "HA_Open" not in df_.columns or "HA_Close" not in df_.columns or df_[adx_col].isnull().all() or df_["HA_Open"].isnull().all() or df_["HA_Close"].isnull().all():
        # print(f"  Missing expected columns or they produced NaNs after join for strategy_adx_heikin_ashi.")
        df_["ADX_HA_Entry"] = False
        df_["ADX_HA_Exit"] = False
        return df_


    # HA turns green: current HA close > HA open AND previous HA close <= HA open
    # Need to handle NaNs from shift(1)
    ha_turns_green = (df_["HA_Close"] > df_["HA_Open"]) & (df_["HA_Close"].shift(1).fillna(-np.inf) <= df_["HA_Open"].shift(1).fillna(-np.inf)) # Treat NaNs as very small numbers
    adx_trending = df_[adx_col] > adx_level

    df_["ADX_HA_Entry"] = adx_trending & ha_turns_green

    # Exit: HA candle turns red
    df_["ADX_HA_Exit"] = df_["HA_Close"] < df_["HA_Open"]
    return df_

# 15. PSAR RSI
def strategy_psar_rsi(df, initial_af=0.02, max_af=0.2, rsi_period=14, rsi_level=50):
    df_ = df.copy()
    # PSAR initialization can take many bars, especially if max_af is high. Estimate required data generously.
    # RSI needs its period. Cross checks need +1. PSAR flip can take > 2 * 1/af bars.
    # Let's use a conservative estimate like max(rsi_period, int(2/initial_af)) + 2
    min_data_needed = max(rsi_period, int(2/initial_af)) + 2
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_psar_rsi, need at least {min_data_needed}")
         df_["PSAR_RSI_Entry"] = False
         df_["PSAR_RSI_Exit"] = False
         return df_


    psar_data = ta.psar(df_["high"], df_["low"], df_["close"], af0=initial_af, af=initial_af, max_af=max_af)
    rsi_col = f"RSI_{rsi_period}"
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)

    if psar_data is None or psar_data.empty or rsi_col not in df_.columns or df_[rsi_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_psar_rsi (PSAR/RSI data check before join).")
        df_["PSAR_RSI_Entry"] = False
        df_["PSAR_RSI_Exit"] = False
        return df_

    df_ = df_.join(psar_data)

    # Determine the correct PSAR long/short/reversal column names based on pandas_ta output
    # Often, it's PSARl_af_maxaf, PSARs_af_maxaf, PSARr_af_maxaf
    psar_long_col = f"PSARl_{initial_af}_{max_af}"
    psar_short_col = f"PSARs_{initial_af}_{max_af}"
    psar_reversal_col = f"PSARr_{initial_af}_{max_af}"

    # Check which PSAR columns are available and use the most reliable flip detection
    entry_cond1 = pd.Series(False, index=df_.index)
    if psar_reversal_col in df_.columns and not df_[psar_reversal_col].isnull().all():
        entry_cond1 = df_[psar_reversal_col] == 1 # Bullish reversal signal (PSAR flips below price)
    elif psar_long_col in df_.columns and psar_short_col in df_.columns:
         # Fallback: PSAR long series becomes non-NaN where it was NaN, or PSAR short series becomes NaN where it was non-NaN
         entry_cond1 = (df_[psar_long_col].notna() & df_[psar_long_col].shift(1).isna()) | \
                       (df_[psar_short_col].isna() & df_[psar_short_col].shift(1).notna())
         # Ensure the flipped PSAR is actually below the current price for a long signal
         entry_cond1 = entry_cond1 & (df_[psar_long_col].fillna(np.inf) < df_["close"]) # Fill NaN PSARl with inf for comparison where PSARl is NaN (not uptrend)
    else:
        # print(f"  Could not determine PSAR flip condition from columns {df_.columns.tolist()} for strategy_psar_rsi.")
        df_["PSAR_RSI_Entry"] = False
        df_["PSAR_RSI_Exit"] = False
        return df_


    entry_cond2 = df_[rsi_col] > rsi_level # RSI confirming bullish sentiment
    df_["PSAR_RSI_Entry"] = entry_cond1 & entry_cond2

    # Exit is typically on a bearish PSAR reversal
    exit_cond1 = pd.Series(False, index=df_.index)
    if psar_reversal_col in df_.columns and not df_[psar_reversal_col].isnull().all():
        exit_cond1 = df_[psar_reversal_col] == -1 # Bearish reversal signal (PSAR flips above price)
    elif psar_long_col in df_.columns and psar_short_col in df_.columns:
        # Fallback: PSAR short series becomes non-NaN or PSAR long series becomes NaN
        exit_cond1 = (df_[psar_short_col].notna() & df_[psar_short_col].shift(1).isna()) | \
                     (df_[psar_long_col].isna() & df_[psar_long_col].shift(1).notna())
        # Ensure the flipped PSAR is actually above the current price for a short signal
        exit_cond1 = exit_cond1 & (df_[psar_short_col].fillna(-np.inf) > df_["close"]) # Fill NaN PSARs with -inf for comparison where PSARs is NaN (not downtrend)
    else:
        exit_cond1 = pd.Series([False] * len(df_), index=df_.index) # Cannot determine PSAR flip condition

    df_["PSAR_RSI_Exit"] = exit_cond1
    return df_

# 16. VWAP RSI
def strategy_vwap_rsi(df, rsi_period=14, rsi_level=50):
    df_ = df.copy()
    min_data_needed = max(rsi_period, 1) + 1 # RSI period + requires volume for VWAP + 1 for conditions
    # VWAP needs at least one bar with volume. RSI needs its period.
    if len(df_) < rsi_period + 1 or df_["volume"].isnull().all(): # Basic check
         # print(f"  Not enough data ({len(df_)}) or no volume for strategy_vwap_rsi, need at least {rsi_period + 1}")
         df_["VWAP_RSI_Entry"] = False
         df_["VWAP_RSI_Exit"] = False
         return df_

    rsi_col = f"RSI_{rsi_period}"
    df_["VWAP"] = ta.vwap(df_["high"], df_["low"], df_["close"], df_["volume"])
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)

    if "VWAP" not in df_.columns or rsi_col not in df_.columns or df_["VWAP"].isnull().all() or df_[rsi_col].isnull().all():
         # print(f"  Indicator calculation failed or produced NaNs for strategy_vwap_rsi.")
         df_["VWAP_RSI_Entry"] = False
         df_["VWAP_RSI_Exit"] = False
         return df_


    entry_cond = (df_["close"] > df_["VWAP"]) & (df_[rsi_col] > rsi_level)
    df_["VWAP_RSI_Entry"] = entry_cond

    # Exit: Price crosses below VWAP OR RSI drops below a certain level (e.g., 40)
    exit_cond = (df_["close"] < df_["VWAP"]) | (df_[rsi_col] < rsi_level - 10)
    df_["VWAP_RSI_Exit"] = exit_cond
    return df_

# 17. EMA Ribbon MACD
def strategy_ema_ribbon_macd(df, ema_lengths=[8, 13, 21, 34, 55], macd_fast=12, macd_slow=26, macd_signal=9):
    df_ = df.copy()
    if not ema_lengths or len(ema_lengths) < 2:
         # print("  ema_lengths must be a list of at least two integers.")
         df_["EMARibbon_MACD_Entry"] = False
         df_["EMARibbon_MACD_Exit"] = False
         return df_
    sorted_ema_lengths = sorted(ema_lengths)
    max_ema_len = sorted_ema_lengths[-1]

    min_data_needed = max(max_ema_len, macd_slow + macd_signal) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_ema_ribbon_macd, need at least {min_data_needed}")
         df_["EMARibbon_MACD_Entry"] = False
         df_["EMARibbon_MACD_Exit"] = False
         return df_


    ema_cols = []
    for length in sorted_ema_lengths:
        col_name = f"EMA_{length}"
        df_[col_name] = ta.ema(df_["close"], length=length)
        ema_cols.append(col_name)

    mac_data = ta.macd(df_["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)

    macd_col = f"MACD_{fast}_{slow}_{signal}"
    macds_col = f"MACDs_{fast}_{slow}_{signal}"

    # Check if all EMAs were calculated and have valid data, and if MACD data is available
    if not all(col in df_.columns and not df_[col].isnull().all() for col in ema_cols) or \
       mac_data is None or mac_data.empty:
        # print(f"  Indicator calculation failed or produced NaNs for strategy_ema_ribbon_macd (EMA/MACD data check before join).")
        df_["EMARibbon_MACD_Entry"] = False
        df_["EMARibbon_MACD_Exit"] = False
        return df_

    df_ = df_.join(mac_data)

    # Check if MACD columns exist and have valid data after join
    if macd_col not in df_.columns or macds_col not in df_.columns or df_[macd_col].isnull().all() or df_[macds_col].isnull().all():
        # print(f"  Missing expected MACD columns or they produced NaNs after join for strategy_ema_ribbon_macd.")
        df_["EMARibbon_MACD_Entry"] = False
        df_["EMARibbon_MACD_Exit"] = False
        return df_


    # Bullish ribbon: shortest EMA > ... > longest EMA
    # The condition can only be true when all EMAs are calculated
    is_bullish_ribbon = pd.Series(True, index=df_.index)
    for i in range(len(ema_cols) - 1):
        is_bullish_ribbon &= (df_[ema_cols[i]] > df_[ema_cols[i+1]])
    is_bullish_ribbon = is_bullish_ribbon.reindex(df_.index).fillna(False) # Align index after boolean ops

    macd_bullish_cross = crossed_above_series(df_[macd_col], df_[macds_col])

    df_["EMARibbon_MACD_Entry"] = is_bullish_ribbon & macd_bullish_cross
    df_["EMARibbon_MACD_Exit"] = crossed_below_series(df_[macd_col], df_[macds_col])
    return df_

# 18. ADX Rising MFI Surge
def strategy_adx_rising_mfi_surge(df, adx_period=14, adx_level=25, mfi_period=14, mfi_surge_level=80):
    df_ = df.copy()
    min_data_needed = max(adx_period, mfi_period) + 1
    # MFI also requires volume
    if len(df_) < min_data_needed or df_["volume"].isnull().all():
         # print(f"  Not enough data ({len(df_)}) or no volume for strategy_adx_rising_mfi_surge, need at least {min_data_needed}")
         df_["ADX_MFI_Entry"] = False
         df_["ADX_MFI_Exit"] = False
         return df_


    adx_data = ta.adx(df_["high"], df_["low"], df_["close"], length=adx_period)
    mfi_col = f"MFI_{mfi_period}"
    df_[mfi_col] = ta.mfi(df_["high"], df_["low"], df_["close"], df_["volume"], length=mfi_period)

    adx_col = f"ADX_{adx_period}"

    if adx_data is None or adx_data.empty or mfi_col not in df_.columns or df_[mfi_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_adx_rising_mfi_surge (MFI/ADX data check before join).")
        df_["ADX_MFI_Entry"] = False
        df_["ADX_MFI_Exit"] = False
        return df_

    df_ = df_.join(adx_data[[adx_col]])

    if adx_col not in df_.columns or df_[adx_col].isnull().all():
        # print(f"  Missing expected ADX column or it produced NaNs after join for strategy_adx_rising_mfi_surge.")
        df_["ADX_MFI_Entry"] = False
        df_["ADX_MFI_Exit"] = False
        return df_


    adx_trending = df_[adx_col] > adx_level
    # adx_rising = df_[adx_col] > df_[adx_col].shift(1) # Optional: ADX is also rising
    mfi_surge = df_[mfi_col] > mfi_surge_level

    df_["ADX_MFI_Entry"] = adx_trending & mfi_surge
    df_["ADX_MFI_Exit"] = df_[mfi_col] < 50 # Exit when MFI drops to neutral (50)
    return df_

# 19. Fractal Breakout RSI
def strategy_fractal_breakout_rsi(df, fractal_lookback=2, rsi_period=14, rsi_level=50):
    df_ = df.copy()
    # Fractal needs 2*lookback+1 for detection. Last Fractal High Val needs another shift (-1 or +1 depending on logic).
    # Breakout needs a shift(1) on the breakout level. RSI needs its period. Cross needs +1.
    # Minimum data needed: max(2*fractal_lookback + 1 + 1 (for value shift), rsi_period + 1) + 1 (for breakout cross)
    min_data_needed = max(2 * fractal_lookback + 2, rsi_period + 1) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_fractal_breakout_rsi, need at least {min_data_needed}")
         df_["Fractal_RSI_Entry"] = False
         df_["Fractal_RSI_Exit"] = False
         return df_

    # Using the helper function for fractal detection
    df_["Is_Fractal_High"] = detect_fractal_high(df_["high"], lookback=fractal_lookback)
    # Get the value of the last confirmed fractal high, forward filled.
    # A fractal high is confirmed *after* the bar itself and the `lookback` bars to its right.
    # So, df_["Is_Fractal_High"] will be True *on* the bar where the fractal peak occurs.
    # We want the *value* of that high, available *after* confirmation.
    # Let's use shift(lookback) on the high where Is_Fractal_High is True.
    # We need the level to be stable *before* the breakout bar. So, ffill and then shift(1) for breakout level.
    df_["Fractal_High_Value"] = df_["high"].where(df_["Is_Fractal_High"]).shift(fractal_lookback) # Value of the peak, aligned to the confirmation bar's right edge
    df_["Last_Confirmed_Fractal_High_Val"] = df_["Fractal_High_Value"].ffill() # Forward fill the last confirmed value

    # The breakout level is the *previous* bar's confirmed fractal high value
    df_["Breakout_Level_Fractal"] = df_["Last_Confirmed_Fractal_High_Val"].shift(1)


    rsi_col = f"RSI_{rsi_period}"
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)

    if "Breakout_Level_Fractal" not in df_.columns or rsi_col not in df_.columns or df_["Breakout_Level_Fractal"].isnull().all() or df_[rsi_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_fractal_breakout_rsi.")
        df_["Fractal_RSI_Entry"] = False
        df_["Fractal_RSI_Exit"] = False
        return df_


    # Breakout occurs when close crosses above the established breakout level
    breakout_cond = crossed_above_series(df_["close"], df_["Breakout_Level_Fractal"].fillna(-np.inf)) # Treat NaN level as infinitely low
    rsi_confirm = df_[rsi_col] > rsi_level

    df_["Fractal_RSI_Entry"] = breakout_cond & rsi_confirm
    df_["Fractal_RSI_Exit"] = df_[rsi_col] < rsi_level - 10 # e.g. RSI drops below 40
    return df_

# 20. Chandelier Exit MACD (Entry based on MACD, Exit using Chandelier)
def strategy_chandelier_exit_macd(df, atr_period=22, atr_mult=3.0, macd_fast=12, macd_slow=26, macd_signal=9):
    df_ = df.copy()
    min_data_needed = max(atr_period, macd_slow + macd_signal) + 1
    # Chandelier needs high, low, close. ATR needs the same.
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_chandelier_exit_macd, need at least {min_data_needed}")
         df_["CE_MACD_Entry"] = False
         df_["CE_MACD_Exit"] = False
         return df_


    chandelier_data = ta.chandelier(df_["high"], df_["low"], df_["close"], n=atr_period, mult=atr_mult)
    mac_data = ta.macd(df_["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)

    macd_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
    macds_col = f"MACDs_{fast}_{slow}_{signal}"
    cel_col = f"CEL_{atr_period}_{atr_mult}.0" # Chandelier Long Exit column name

    if chandelier_data is None or mac_data is None or chandelier_data.empty or mac_data.empty:
        # print(f"  Indicator calculation failed for strategy_chandelier_exit_macd (Chandelier/MACD data check before join).")
        df_["CE_MACD_Entry"] = False
        df_["CE_MACD_Exit"] = False
        return df_

    df_ = df_.join(chandelier_data) # Adds 'CES_22_3.0' and 'CEL_22_3.0'
    df_ = df_.join(mac_data)

    if macd_col not in df_.columns or macds_col not in df_.columns or cel_col not in df_.columns or df_[macd_col].isnull().all() or df_[macds_col].isnull().all() or df_[cel_col].isnull().all():
        # print(f"  Missing expected columns or they produced NaNs after join for strategy_chandelier_exit_macd.")
        df_["CE_MACD_Entry"] = False
        df_["CE_MACD_Exit"] = False
        return df_


    macd_bullish_cross = crossed_above_series(df_[macd_col], df_[macds_col])

    # Entry: MACD bullish cross AND price is currently above the Chandelier Long Exit level
    df_["CE_MACD_Entry"] = macd_bullish_cross & (df_["close"] > df_[cel_col].fillna(-np.inf)) # Treat NaN as infinitely low
    df_["CE_MACD_Exit"] = df_["close"] < df_[cel_col].fillna(np.inf) # Exit when price hits Chandelier Long Exit (treat NaN as infinitely high)
    return df_

# 21. SuperTrend RSI Pullback
def strategy_supertrend_rsi_pullback(df, atr_length=10, factor=3.0, rsi_period=14, rsi_pullback_level=50):
    df_ = df.copy()
    min_data_needed = max(atr_length, rsi_period) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_supertrend_rsi_pullback, need at least {min_data_needed}")
         df_["SuperTrend_RSI_Entry"] = False
         df_["SuperTrend_RSI_Exit"] = False
         return df_

    # pandas_ta supertrend returns a DataFrame with columns based on params
    # e.g., SUPERT_10_3.0, SUPERTd_10_3.0, SUPERTl_10_3.0, SUPERTs_10_3.0
    # We primarily need the direction column (SUPERTd...) and value column (SUPERT...)
    supertrend_data = ta.supertrend(df_["high"], df_["low"], df_["close"], length=atr_length, multiplier=factor)
    rsi_col = f"RSI_{rsi_period}"
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)


    if supertrend_data is None or supertrend_data.empty or rsi_col not in df_.columns or df_[rsi_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_supertrend_rsi_pullback (SuperTrend/RSI data check before join).")
        df_["SuperTrend_RSI_Entry"] = False
        df_["SuperTrend_RSI_Exit"] = False
        return df_

    # Determine the correct SuperTrend column names
    st_dir_col = f"SUPERTd_{atr_length}_{factor}.0" # Default name for direction
    st_val_col = f"SUPERT_{atr_length}_{factor}.0" # Default name for value

    # Check if the default names exist, if not, try to find them heuristically
    if st_dir_col not in supertrend_data.columns or st_val_col not in supertrend_data.columns:
         st_cols = [col for col in supertrend_data.columns if 'SUPERT' in col]
         if len(st_cols) >= 2:
              st_dir_col = [col for col in st_cols if col.endswith('d')][0] if [col for col in st_cols if col.endswith('d')] else None
              st_val_col = [col for col in st_cols if col.startswith('SUPERT_')][0] if [col for col in st_cols if col.startswith('SUPERT_')] else None
         # If still not found, fail
         if not st_dir_col or not st_val_col:
             # print(f"  Could not identify SuperTrend direction or value columns: {st_cols}")
             df_["SuperTrend_RSI_Entry"] = False
             df_["SuperTrend_RSI_Exit"] = False
             return df_

    df_ = df_.join(supertrend_data[[st_dir_col, st_val_col]]) # Join only needed columns

    if st_dir_col not in df_.columns or df_[st_dir_col].isnull().all():
        # print(f"  SuperTrend direction column '{st_dir_col}' not found or produced NaNs after join.")
        df_["SuperTrend_RSI_Entry"] = False
        df_["SuperTrend_RSI_Exit"] = False
        return df_


    is_uptrend_st = df_[st_dir_col] == 1 # SuperTrend is bullish (1) or bearish (-1)
    # RSI recovering from pullback: RSI was below pullback level, now crossed above it
    rsi_recovering = crossed_above_level(df_[rsi_col], rsi_pullback_level)

    # Entry: SuperTrend indicates uptrend AND RSI shows recovery (pullback might be ending)
    df_["SuperTrend_RSI_Entry"] = is_uptrend_st & rsi_recovering

    # Exit: SuperTrend flips to downtrend OR RSI becomes very overbought
    df_["SuperTrend_RSI_Exit"] = (df_[st_dir_col] == -1) | (df_[rsi_col] > 75)
    return df_

# 22. TEMA Cross Volume
def strategy_tema_cross_volume(df, short_tema_len=10, long_tema_len=30, vol_ma_period=20, vol_factor=1.5):
    df_ = df.copy()
    min_data_needed = max(long_tema_len, vol_ma_period) + 1 # Max period + 1 for cross
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_tema_cross_volume, need at least {min_data_needed}")
         df_["TEMA_Vol_Entry"] = False
         df_["TEMA_Vol_Exit"] = False
         return df_

    short_tema_col = f"TEMA_{short_tema_len}"
    long_tema_col = f"TEMA_{long_tema_len}"
    df_[short_tema_col] = ta.tema(df_["close"], length=short_tema_len)
    df_[long_tema_col] = ta.tema(df_["close"], length=long_tema_len)
    df_["Volume_MA"] = df_["volume"].rolling(window=vol_ma_period).mean()

    if short_tema_col not in df_.columns or long_tema_col not in df_.columns or "Volume_MA" not in df_.columns or df_[short_tema_col].isnull().all() or df_[long_tema_col].isnull().all() or df_["Volume_MA"].isnull().all():
         # print(f"  Indicator calculation failed or produced NaNs for strategy_tema_cross_volume.")
         df_["TEMA_Vol_Entry"] = False
         df_["TEMA_Vol_Exit"] = False
         return df_


    tema_cross_bullish = crossed_above_series(df_[short_tema_col], df_[long_tema_col])
    high_volume = df_["volume"] > vol_factor * df_["Volume_MA"]

    df_["TEMA_Vol_Entry"] = tema_cross_bullish & high_volume
    df_["TEMA_Vol_Exit"] = crossed_below_series(df_[short_tema_col], df_[long_tema_col])
    return df_

# 23. TSI Resistance Break
def strategy_tsi_resistance(df, tsi_short=13, tsi_long=25, resistance_level=0):
    df_ = df.copy()
    min_data_needed = tsi_long + 2
    if len(df_) < min_data_needed:
        df_["TSI_Break_Entry"] = False
        df_["TSI_Break_Exit"] = False
        return df_

    tsi = ta.tsi(df_["close"], short=tsi_short, long=tsi_long)
    df_ = safe_join(df_, tsi)

    tsi_col = f"TSI_{tsi_long}_{tsi_short}"  # Note: pandas-ta reverses order
    if tsi_col not in df_.columns:
        df_["TSI_Break_Entry"] = False
        df_["TSI_Break_Exit"] = False
        return df_

    df_["TSI_Break_Entry"] = crossed_above_level(df_[tsi_col], resistance_level)
    df_["TSI_Break_Exit"] = crossed_below_level(df_[tsi_col], resistance_level)
    return df_

# 24. TRIX OBV
def strategy_trix_obv(df, trix_period=15, trix_signal=9):
    df_ = df.copy()
    # TRIX involves triple EMA (length*3 roughly). OBV starts from 1 bar. Cross needs +1.
    min_data_needed = max(trix_period * 3, trix_signal) + 1
    # OBV needs volume
    if len(df_) < min_data_needed or df_["volume"].isnull().all():
         # print(f"  Not enough data ({len(df_)}) or no volume for strategy_trix_obv, need at least {min_data_needed}")
         df_["TRIX_OBV_Entry"] = False
         df_["TRIX_OBV_Exit"] = False
         return df_


    trix_data = ta.trix(df_["close"], length=trix_period, signal=trix_signal)
    df_["OBV"] = ta.obv(df_["close"], df_["volume"])

    trix_col = f"TRIX_{trix_period}_{trix_signal}"
    trixs_col = f"TRIXs_{trix_period}_{trix_signal}"

    if trix_data is None or trix_data.empty or "OBV" not in df_.columns or df_["OBV"].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_trix_obv (TRIX data check before join).")
        df_["TRIX_OBV_Entry"] = False
        df_["TRIX_OBV_Exit"] = False
        return df_
    df_ = df_.join(trix_data) # TRIX_15_9, TRIXs_15_9

    if trix_col not in df_.columns or trixs_col not in df_.columns or df_[trix_col].isnull().all() or df_[trixs_col].isnull().all():
        # print(f"  Missing expected TRIX columns or they produced NaNs after join for strategy_trix_obv.")
        df_["TRIX_OBV_Entry"] = False
        df_["TRIX_OBV_Exit"] = False
        return df_


    trix_bullish_cross = crossed_above_series(df_[trix_col], df_[trixs_col])
    # OBV rising check needs +1 bar for shift
    obv_rising = df_["OBV"] > df_["OBV"].shift(1).fillna(-np.inf) # Treat initial NaN as infinitely low

    df_["TRIX_OBV_Entry"] = trix_bullish_cross & obv_rising
    df_["TRIX_OBV_Exit"] = crossed_below_series(df_[trix_col], df_[trixs_col])
    return df_

# 25. Awesome Oscillator Divergence MACD
def strategy_ao_divergence_macd(df, ao_fast=5, ao_slow=34, macd_fast=12, macd_slow=26, macd_signal=9, div_lookback=14):
    df_ = df.copy()
    # AO needs slow period. MACD needs slow+signal. Divergence check needs lookback*2 + buffer. Cross needs +1.
    min_data_needed = max(ao_slow, macd_slow + signal, div_lookback) + 2 # Use max of indicator periods + some buffer + cross needs 1 shifted bar
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_ao_divergence_macd, need at least {min_data_needed}")
         df_["AO_MACD_Div_Entry"] = False
         df_["AO_MACD_Div_Exit"] = False
         return df_


    df_["AO"] = ta.ao(df_["high"], df_["low"], fast=ao_fast, slow=ao_slow)
    mac_data = ta.macd(df_["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)

    macd_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
    macds_col = f"MACDs_{fast}_{slow}_{signal}"

    if df_["AO"].isnull().all() or mac_data is None or mac_data.empty:
        # print(f"  Indicator calculation failed or produced NaNs for strategy_ao_divergence_macd (AO/MACD data check before join).")
        df_["AO_MACD_Div_Entry"] = False
        df_["AO_MACD_Div_Exit"] = False
        return df_
    df_ = df_.join(mac_data)

    if macd_col not in df_.columns or macds_col not in df_.columns or df_[macd_col].isnull().all() or df_[macds_col].isnull().all():
        # print(f"  Missing expected MACD columns or they produced NaNs after join for strategy_ao_divergence_macd.")
        df_["AO_MACD_Div_Entry"] = False
        df_["AO_MACD_Div_Exit"] = False
        return df_

    # Need enough data for divergence calculation after other indicators have settled
    # Pass the relevant series subset where AO is not null for divergence detection
    valid_ao_indices = df_["AO"].dropna().index
    if len(valid_ao_indices) >= div_lookback + 1: # Check if enough data exists AFTER AO is calculated
         divergence_series = detect_divergence(df_["close"].loc[valid_ao_indices], df_["AO"].loc[valid_ao_indices], lookback=div_lookback, type='bullish')
         # Reindex divergence series to match main df index
         df_["AO_Bullish_Divergence"] = divergence_series.reindex(df_.index).fillna(False)
    else:
         # print(f"  Not enough valid AO data ({len(valid_ao_indices)}) for divergence detection (lookback={div_lookback}).")
         df_["AO_Bullish_Divergence"] = False


    macd_bullish_cross = crossed_above_series(df_[macd_col], df_[macds_col])

    df_["AO_MACD_Div_Entry"] = df_["AO_Bullish_Divergence"] & macd_bullish_cross
    df_["AO_MACD_Div_Exit"] = crossed_below_series(df_[macd_col], df_[macds_col])
    return df_

# 26. Heikin Ashi CMO
def strategy_heikin_ashi_cmo(df, cmo_period=14, cmo_level=0):
    df_ = df.copy()
    # CMO needs its period. HA needs open, high, low, close. HA color change needs 2 bars minimum.
    # CMO cross level needs +1. HA check needs +1.
    min_data_needed = max(cmo_period, 2) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_heikin_ashi_cmo, need at least {min_data_needed}")
         df_["HA_CMO_Entry"] = False
         df_["HA_CMO_Exit"] = False
         return df_


    ha_data = ta.ha(df_["open"], df_["high"], df_["low"], df_["close"])
    cmo_col = f"CMO_{cmo_period}"
    df_[cmo_col] = ta.cmo(df_["close"], length=cmo_period)

    if ha_data is None or ha_data.empty or cmo_col not in df_.columns or df_[cmo_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_heikin_ashi_cmo (HA/CMO data check before join).")
        df_["HA_CMO_Entry"] = False
        df_["HA_CMO_Exit"] = False
        return df_

    df_["HA_Open"] = ha_data["HA_open"]
    df_["HA_Close"] = ha_data["HA_close"]

    if "HA_Open" not in df_.columns or "HA_Close" not in df_.columns or df_["HA_Open"].isnull().all() or df_["HA_Close"].isnull().all():
         # print(f"  Missing expected HA columns or they produced NaNs after join for strategy_heikin_ashi_cmo.")
         df_["HA_CMO_Entry"] = False
         df_["HA_CMO_Exit"] = False
         return df_


    # HA turns green: current HA close > HA open AND previous HA close <= HA open
    ha_turns_green = (df_["HA_Close"] > df_["HA_Open"]) & (df_["HA_Close"].shift(1).fillna(-np.inf) <= df_["HA_Open"].shift(1).fillna(-np.inf))
    cmo_bullish_cross = crossed_above_level(df_[cmo_col], cmo_level)

    df_["HA_CMO_Entry"] = ha_turns_green & cmo_bullish_cross

    # Exit: HA candle turns red OR CMO crosses below level - 10
    df_["HA_CMO_Exit"] = (df_["HA_Close"] < df_["HA_Open"]) | crossed_below_level(df_[cmo_col], cmo_level - 10)
    return df_

# 27. CCI Bollinger
def strategy_cci_bollinger(df, cci_period=20, cci_extreme=-100, bb_period=20, bb_std=2.0):
    df_ = df.copy()
    # CCI needs its period. BB needs its period. Cross/Level checks need +1.
    min_data_needed = max(cci_period, bb_period) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_cci_bollinger, need at least {min_data_needed}")
         df_["CCI_BB_Entry_Buy"] = False
         df_["CCI_BB_Exit_Buy"] = False # Only implementing Buy for now
         return df_


    cci_col = f"CCI_{cci_period}"
    df_[cci_col] = ta.cci(df_["high"], df_["low"], df_["close"], length=cci_period)
    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)

    bbl_col = f"BBL_{bb_period}_{bb_std}"
    bbm_col = f"BBM_{bb_period}_{bb_std}"

    if bbands is None or bbands.empty or cci_col not in df_.columns or df_[cci_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_cci_bollinger (CCI/BB data check before join).")
        df_["CCI_BB_Entry_Buy"] = False
        df_["CCI_BB_Exit_Buy"] = False
        return df_
    df_ = df_.join(bbands)

    if bbl_col not in df_.columns or bbm_col not in df_.columns or df_[bbl_col].isnull().all() or df_[bbm_col].isnull().all():
        # print(f"  Missing expected BB columns or they produced NaNs after join for strategy_cci_bollinger.")
        df_["CCI_BB_Entry_Buy"] = False
        df_["CCI_BB_Exit_Buy"] = False
        return df_


    cci_oversold = df_[cci_col] < cci_extreme
    # Bounce off lower BB: low was at/below lower band, close is above
    # Needs shift(1) on low and lower band.
    shifted_low = df_["low"].shift(1).fillna(np.inf) # Treat NaN low as infinitely high
    shifted_bbl = df_[bbl_col].shift(1).fillna(np.inf) # Treat NaN BBL as infinitely high

    bounce_lower_bb = (shifted_low <= shifted_bbl) & \
                      (df_["close"] > df_[bbl_col].fillna(-np.inf)) # Treat NaN BBL as infinitely low


    df_["CCI_BB_Entry_Buy"] = cci_oversold & bounce_lower_bb
    df_["CCI_BB_Exit_Buy"] = crossed_above_level(df_[cci_col], 0) | \
                            crossed_above_series(df_["close"], df_[bbm_col])
    return df_

# 28. CCI Reversion
def strategy_cci_reversion(df, cci_period=20, cci_extreme_low=-150, cci_revert_level=-100):
    df_ = df.copy()
    # CCI needs its period. Crossed above level needs +1. Previous check needs +1.
    min_data_needed = cci_period + 2
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_cci_reversion, need at least {min_data_needed}")
         df_["CCI_Revert_Entry_Buy"] = False
         df_["CCI_Revert_Exit_Buy"] = False
         return df_

    cci_col = f"CCI_{cci_period}"
    df_[cci_col] = ta.cci(df_["high"], df_["low"], df_["close"], length=cci_period)

    if cci_col not in df_.columns or df_[cci_col].isnull().all():
         # print(f"  Indicator calculation failed or produced NaNs for strategy_cci_reversion.")
         df_["CCI_Revert_Entry_Buy"] = False
         df_["CCI_Revert_Exit_Buy"] = False
         return df_


    # CCI was below extreme_low, now crossed above revert_level
    # Needs shift(1) on CCI for previous bar value
    shifted_cci = df_[cci_col].shift(1).fillna(df_[cci_col].iloc[0] if not df_[cci_col].empty else cci_extreme_low - 1) # Fill with a value below extreme_low if NaN

    entry_buy = (shifted_cci < cci_extreme_low) & \
                crossed_above_level(df_[cci_col], cci_revert_level)

    df_["CCI_Revert_Entry_Buy"] = entry_buy
    df_["CCI_Revert_Exit_Buy"] = crossed_above_level(df_[cci_col], 0) # Exit when CCI crosses 0
    return df_

# 29. Keltner RSI Oversold
def strategy_keltner_rsi_oversold(df, kc_length=20, kc_scalar=2.0, rsi_period=14, rsi_oversold=30):
    df_ = df.copy()
    min_data_needed = max(kc_length, rsi_period) + 1
    if len(df_) < min_data_needed:
        df_["KeltnerRSI_Entry"] = False
        df_["KeltnerRSI_Exit"] = False
        return df_

    kc = ta.kc(df_["high"], df_["low"], df_["close"], length=kc_length, scalar=kc_scalar)
    rsi_col = f"RSI_{rsi_period}"
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)
    df_ = safe_join(df_, kc)

    kcl_col = f"KCL_{kc_length}_{int(kc_scalar)}"
    if kcl_col not in df_.columns or df_[rsi_col].isnull().all():
        df_["KeltnerRSI_Entry"] = False
        df_["KeltnerRSI_Exit"] = False
        return df_

    df_["KeltnerRSI_Entry"] = (df_["close"] < df_[kcl_col]) & (df_[rsi_col] < rsi_oversold)
    df_["KeltnerRSI_Exit"] = crossed_above_level(df_[rsi_col], rsi_oversold + 10)
    return df_

    # RSI recovering from oversold: RSI was below rsi_oversold, now crossed above it
    rsi_recovering_from_oversold = crossed_above_level(df_[rsi_col], rsi_oversold)
    # Price above lower KC band (after potentially dipping below)
    price_above_lower_kc = df_["close"] > df_[kcl_col].fillna(-np.inf) # Treat NaN as infinitely low

    df_["KC_RSI_OS_Entry_Buy"] = rsi_recovering_from_oversold & price_above_lower_kc
    df_["KC_RSI_OS_Exit_Buy"] = crossed_above_level(df_[rsi_col], 50) | \
                               crossed_above_series(df_["close"], df_[kcm_col])
    return df_

# 30. Keltner MFI Oversold
def strategy_keltner_mfi_oversold(df, kc_length=20, kc_atr_length=10, kc_mult=2.0, mfi_period=14, mfi_oversold=20):
    df_ = df.copy()
    # KC needs kc_length + kc_atr_length. MFI needs its period + volume. Cross needs +1.
    min_data_needed = max(kc_length, kc_atr_length, mfi_period) + 1
    if len(df_) < min_data_needed or df_["volume"].isnull().all():
         # print(f"  Not enough data ({len(df_)}) or no volume for strategy_keltner_mfi_oversold, need at least {min_data_needed}")
         df_["KC_MFI_OS_Entry_Buy"] = False
         df_["KC_MFI_OS_Exit_Buy"] = False
         return df_

    kc_data = ta.kc(df_["high"], df_["low"], df_["close"], length=kc_length, atr_length=kc_atr_length, scalar=kc_mult)
    mfi_col = f"MFI_{mfi_period}"
    df_[mfi_col] = ta.mfi(df_["high"], df_["low"], df_["close"], df_["volume"], length=mfi_period)

     # Corrected based on typical pandas-ta output
    kcl_col = f"KCL_{kc_length}_{int(kc_scalar)}" # Ensure mult is float in name
    kcm_col = f"KCM_{kc_length}_{int(kc_scalar)}"
    kcu_col = f"KCU_{kc_length}_{kc_atr_length}_{float(kc_mult)}" # Also for KCU if used

    if kc_data is None or kc_data.empty or mfi_col not in df_.columns or df_[mfi_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_keltner_mfi_oversold (KC/MFI data check before join).")
        df_["KC_MFI_OS_Entry_Buy"] = False
        df_["KC_MFI_OS_Exit_Buy"] = False
        return df_
    df_ = df_.join(kc_data)

    if kcl_col not in df_.columns or kcm_col not in df_.columns or df_[kcl_col].isnull().all() or df_[kcm_col].isnull().all():
        # print(f"  Missing expected KC columns or they produced NaNs after join for strategy_keltner_mfi_oversold.")
        df_["KC_MFI_OS_Entry_Buy"] = False
        df_["KC_MFI_OS_Exit_Buy"] = False
        return df_


    # MFI recovering from oversold: MFI was below mfi_oversold, now crossed above it
    mfi_recovering_from_oversold = crossed_above_level(df_[mfi_col], mfi_oversold)
    # Price above lower KC band (after potentially dipping below)
    price_above_lower_kc = df_["close"] > df_[kcl_col].fillna(-np.inf) # Treat NaN as infinitely low


    df_["KC_MFI_OS_Entry_Buy"] = mfi_recovering_from_oversold & price_above_lower_kc
    df_["KC_MFI_OS_Exit_Buy"] = crossed_above_level(df_[mfi_col], 50) | \
                               crossed_above_series(df_["close"], df_[kcm_col])
    return df_

# 31. Double MA Pullback
def strategy_double_ma_pullback(df, short_ma_len=20, long_ma_len=50, ma_type="ema"):
    df_ = df.copy()
    # MAs need their length. Trend check needs max length. Pullback check needs +2 (shifted low, shifted MA).
    min_data_needed = max(short_ma_len, long_ma_len) + 2
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_double_ma_pullback, need at least {min_data_needed}")
         df_["MA_Pullback_Entry_Buy"] = False
         df_["MA_Pullback_Exit_Buy"] = False
         return df_

    short_ma_col = "Short_MA"
    long_ma_col = "Long_MA"

    if ma_type.lower() == "sma":
        df_[short_ma_col] = ta.sma(df_["close"], length=short_ma_len)
        df_[long_ma_col] = ta.sma(df_["close"], length=long_ma_len)
    else: # Default to EMA
        df_[short_ma_col] = ta.ema(df_["close"], length=short_ma_len)
        df_[long_ma_col] = ta.ema(df_["close"], length=long_ma_len)

    if short_ma_col not in df_.columns or long_ma_col not in df_.columns or df_[short_ma_col].isnull().all() or df_[long_ma_col].isnull().all():
         # print(f"  MA calculation failed or produced NaNs for strategy_double_ma_pullback.")
         df_["MA_Pullback_Entry_Buy"] = False
         df_["MA_Pullback_Exit_Buy"] = False
         return df_

    is_uptrend = df_[short_ma_col] > df_[long_ma_col]

    # Pullback to short MA: low of *previous* bar was <= short MA, and close of *current* bar is > short MA
    # Ensure shifted low and shifted MA are not NaN
    shifted_low = df_["low"].shift(1).fillna(np.inf) # Treat NaN low as infinitely high for <= check
    shifted_short_ma = df_[short_ma_col].shift(1).fillna(-np.inf) # Treat NaN MA as infinitely low for <= check

    pulled_back_to_short_ma = (shifted_low <= shifted_short_ma) & \
                              (df_["close"] > df_[short_ma_col].fillna(-np.inf)) # Treat NaN MA as infinitely low

    df_["MA_Pullback_Entry_Buy"] = is_uptrend & pulled_back_to_short_ma
    df_["MA_Pullback_Exit_Buy"] = crossed_below_series(df_["close"], df_[short_ma_col]) | \
                                  crossed_below_series(df_[short_ma_col], df_[long_ma_col])
    return df_

# 32. Bollinger Bounce Volume
def strategy_bollinger_bounce_volume(df, bb_period=20, bb_std=2.0, vol_ma_period=20, vol_factor=1.5):
    df_ = df.copy()
    # BB needs its period. Vol MA needs its period. Bounce condition needs +2 (shifted low, shifted BB).
    min_data_needed = max(bb_period, vol_ma_period) + 2
    if len(df_) < min_data_needed or df_["volume"].isnull().all():
         # print(f"  Not enough data ({len(df_)}) or no volume for strategy_bollinger_bounce_volume, need at least {min_data_needed}")
         df_["BB_Bounce_Vol_Entry_Buy"] = False
         df_["BB_Bounce_Vol_Exit_Buy"] = False
         return df_

    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)
    df_["Volume_MA"] = df_["volume"].rolling(window=vol_ma_period).mean()

    bbl_col = f"BBL_{bb_period}_{bb_std}"
    bbm_col = f"BBM_{bb_period}_{bb_std}"

    if bbands is None or bbands.empty or "Volume_MA" not in df_.columns or df_["Volume_MA"].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_bollinger_bounce_volume (BB data check before join).")
        df_["BB_Bounce_Vol_Entry_Buy"] = False
        df_["BB_Bounce_Vol_Exit_Buy"] = False
        return df_
    df_ = df_.join(bbands)

    if bbl_col not in df_.columns or bbm_col not in df_.columns or df_[bbl_col].isnull().all() or df_[bbm_col].isnull().all():
        # print(f"  Missing expected BB columns or they produced NaNs after join for strategy_bollinger_bounce_volume.")
        df_["BB_Bounce_Vol_Entry_Buy"] = False
        df_["BB_Bounce_Vol_Exit_Buy"] = False
        return df_


    # Bounce off lower BB: low of *previous* bar was <= lower band, and close of *current* bar is > lower band
    # Needs shift(1) on low and lower band.
    shifted_low = df_["low"].shift(1).fillna(np.inf) # Treat NaN low as infinitely high for <= check
    shifted_bbl = df_[bbl_col].shift(1).fillna(np.inf) # Treat NaN BBL as infinitely high

    bounce_lower_bb = (shifted_low <= shifted_bbl) & \
                      (df_["close"] > df_[bbl_col].fillna(-np.inf)) # Treat NaN BBL as infinitely low

    high_volume = df_["volume"] > vol_factor * df_["Volume_MA"]

    df_["BB_Bounce_Vol_Entry_Buy"] = bounce_lower_bb & high_volume
    df_["BB_Bounce_Vol_Exit_Buy"] = crossed_above_series(df_["close"], df_[bbm_col])
    return df_

# 33. RSI Range Breakout BB
def strategy_rsi_range_breakout_bb(df, rsi_period=14, rsi_low=40, rsi_high=60, bb_period=20, bb_std=2.0):
    df_ = df.copy()
    # RSI needs its period. BB needs its period. Level/Series Crosses need +1.
    min_data_needed = max(rsi_period, bb_period) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_rsi_range_breakout_bb, need at least {min_data_needed}")
         df_["RSI_Range_BB_Entry_Buy"] = False
         df_["RSI_Range_BB_Exit_Buy"] = False
         return df_

    rsi_col = f"RSI_{rsi_period}"
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)
    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)

    bbu_col = f"BBU_{bb_period}_{bb_std}"
    bbm_col = f"BBM_{bb_period}_{bb_std}"


    if bbands is None or bbands.empty or rsi_col not in df_.columns or df_[rsi_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_rsi_range_breakout_bb (BB/RSI data check before join).")
        df_["RSI_Range_BB_Entry_Buy"] = False
        df_["RSI_Range_BB_Exit_Buy"] = False
        return df_
    df_ = df_.join(bbands)

    if bbu_col not in df_.columns or bbm_col not in df_.columns or df_[bbu_col].isnull().all() or df_[bbm_col].isnull().all():
        # print(f"  Missing expected BB columns or they produced NaNs after join for strategy_rsi_range_breakout_bb.")
        df_["RSI_Range_BB_Entry_Buy"] = False
        df_["RSI_Range_BB_Exit_Buy"] = False
        return df_


    # RSI breaks above its range high (e.g., 60)
    rsi_breakout_bullish = crossed_above_level(df_[rsi_col], rsi_high)
    # Price breaks above the Upper Bollinger Band
    bb_breakout_bullish = crossed_above_series(df_["close"], df_[bbu_col].fillna(-np.inf)) # Treat NaN as infinitely low

    df_["RSI_Range_BB_Entry_Buy"] = rsi_breakout_bullish & bb_breakout_bullish
    df_["RSI_Range_BB_Exit_Buy"] = crossed_below_level(df_[rsi_col], rsi_high - 10) | \
                                   crossed_below_series(df_["close"], df_[bbm_col])
    return df_

# 34. Keltner Middle RSI Divergence
def strategy_keltner_middle_rsi_divergence(df, kc_length=20, kc_scalar=2.0, rsi_period=14, lookback=14):
    df_ = df.copy()
    min_data_needed = max(kc_length, rsi_period, lookback) + 1
    if len(df_) < min_data_needed:
        df_["KCM_RSI_Div_Entry"] = False
        return df_

    kc = ta.kc(df_["high"], df_["low"], df_["close"], length=kc_length, scalar=kc_scalar)
    rsi_col = f"RSI_{rsi_period}"
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)
    df_ = safe_join(df_, kc)

    kcm_col = f"KCM_{kc_length}_{kc_scalar}"
    if kcm_col not in df_.columns or rsi_col not in df_.columns:
        df_["KCM_RSI_Div_Entry"] = False
        return df_

    df_["KCM_RSI_Div_Entry"] = detect_divergence(df_[kcm_col], df_[rsi_col], lookback=lookback, type='bullish')
    return df_

    # Price is near middle band
    channel_width = df_[kcu_col] - df_[kcl_col]
    # Use .abs() and fillna for robustness
    price_near_kc_mid_strict = (df_["close"] - df_[kcm_col]).abs() < (channel_width * 0.1).fillna(np.inf) # Treat NaN width as infinitely large diff

    # Need enough data for divergence calculation after other indicators have settled
    valid_rsi_indices = df_[rsi_col].dropna().index
    if len(valid_rsi_indices) >= div_lookback + 1: # Check if enough data exists AFTER RSI is calculated
         bullish_divergence = detect_divergence(df_["close"].loc[valid_rsi_indices], df_[rsi_col].loc[valid_rsi_indices], lookback=div_lookback, type='bullish')
         df_["RSI_Bullish_Divergence"] = bullish_divergence.reindex(df_.index).fillna(False)
    else:
         # print(f"  Not enough valid RSI data ({len(valid_rsi_indices)}) for divergence detection (lookback={div_lookback}).")
         df_["RSI_Bullish_Divergence"] = False


    df_["KC_Mid_RSI_Div_Entry_Buy"] = price_near_kc_mid_strict & df_["RSI_Bullish_Divergence"]
    df_["KC_Mid_RSI_Div_Exit_Buy"] = df_[rsi_col] > 70 # Exit if RSI gets overbought
    return df_

# 35. Hammer on Keltner Volume
def strategy_hammer_keltner_volume(df, kc_length=20, kc_scalar=2.0, volume_ratio=2.0):
    df_ = df.copy()
    min_data_needed = kc_length + 1
    if len(df_) < min_data_needed:
        df_["HammerKeltner_Entry"] = False
        return df_

    kc = ta.kc(df_["high"], df_["low"], df_["close"], length=kc_length, scalar=kc_scalar)
    df_["Volume_Avg"] = df_["volume"].rolling(window=20).mean()
    df_ = safe_join(df_, kc)

    kcl_col = f"KCL_{kc_length}_{int(kc_scalar)}"
    if kcl_col not in df_.columns:
        df_["HammerKeltner_Entry"] = False
        return df_

    hammer_cond = (df_["close"] > df_["open"]) & \
                  ((df_["low"] < df_[kcl_col]) | (df_["close"] < df_[kcl_col])) & \
                  (df_["volume"] > volume_ratio * df_["Volume_Avg"])

    df_["HammerKeltner_Entry"] = hammer_cond
    return df_


# 36. Bollinger Upper Break Volume
def strategy_bollinger_upper_break_volume(df, bb_period=20, bb_std=2.0, vol_ma_period=20, vol_factor=1.5):
    df_ = df.copy()
    # BB needs its period. Vol MA needs its period. Cross needs +1.
    min_data_needed = max(bb_period, vol_ma_period) + 1
    if len(df_) < min_data_needed or df_["volume"].isnull().all():
         # print(f"  Not enough data ({len(df_)}) or no volume for strategy_bollinger_upper_break_volume, need at least {min_data_needed}")
         df_["BB_Break_Vol_Entry"] = False
         df_["BB_Break_Vol_Exit"] = False
         return df_

    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)
    df_["Volume_MA"] = df_["volume"].rolling(window=vol_ma_period).mean()

    bb_upper_col = f"BBU_{bb_period}_{bb_std}"
    bb_mid_col = f"BBM_{bb_period}_{bb_std}"


    if bbands is None or bbands.empty or "Volume_MA" not in df_.columns or df_["Volume_MA"].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_bollinger_upper_break_volume (BB/VolMA data check before join).")
        df_["BB_Break_Vol_Entry"] = False
        df_["BB_Break_Vol_Exit"] = False
        return df_
    df_ = df_.join(bbands)

    if bb_upper_col not in df_.columns or bb_mid_col not in df_.columns or df_[bb_upper_col].isnull().all() or df_[bb_mid_col].isnull().all():
        # print(f"  Missing expected BB columns or they produced NaNs after join for strategy_bollinger_upper_break_volume.")
        df_["BB_Break_Vol_Entry"] = False
        df_["BB_Break_Vol_Exit"] = False
        return df_


    breakout_upper_bb = crossed_above_series(df_["close"], df_[bb_upper_col].fillna(-np.inf)) # Treat NaN as infinitely low
    high_volume = df_["volume"] > vol_factor * df_["Volume_MA"].fillna(0) # Treat NaN Volume_MA as 0

    df_["BB_Break_Vol_Entry"] = breakout_upper_bb & high_volume
    df_["BB_Break_Vol_Exit"] = crossed_below_series(df_["close"], df_[bb_mid_col])
    return df_

# 37. RSI EMA Crossover
def strategy_rsi_ema_crossover(df, rsi_period=14, rsi_ma_period=10, ema_period=50):
    df_ = df.copy()
    # RSI needs its period. RSI EMA needs rsi_period + rsi_ma_period. Price EMA needs ema_period. Cross needs +1.
    min_data_needed = max(rsi_period + rsi_ma_period, ema_period) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_rsi_ema_crossover, need at least {min_data_needed}")
         df_["RSI_EMA_Cross_Entry"] = False
         df_["RSI_EMA_Cross_Exit"] = False
         return df_


    rsi_col = f"RSI_{rsi_period}"
    rsi_ema_col = f"RSI_EMA_{rsi_ma_period}"
    price_ema_col = f"Price_EMA_{ema_period}"

    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)

    if rsi_col not in df_.columns or df_[rsi_col].isnull().all():
        # print(f"  RSI calculation failed or produced NaNs for strategy_rsi_ema_crossover.")
        df_["RSI_EMA_Cross_Entry"] = False
        df_["RSI_EMA_Cross_Exit"] = False
        return df_

    df_[rsi_ema_col] = ta.ema(df_[rsi_col], length=rsi_ma_period)
    df_[price_ema_col] = ta.ema(df_["close"], length=ema_period)

    if rsi_ema_col not in df_.columns or price_ema_col not in df_.columns or df_[rsi_ema_col].isnull().all() or df_[price_ema_col].isnull().all():
         # print(f"  EMA calculation failed or produced NaNs for strategy_rsi_ema_crossover after first step.")
         df_["RSI_EMA_Cross_Entry"] = False
         df_["RSI_EMA_Cross_Exit"] = False
         return df_


    rsi_cross_ema_bullish = crossed_above_series(df_[rsi_col], df_[rsi_ema_col])
    price_above_ema = df_["close"] > df_[price_ema_col].fillna(-np.inf) # Treat NaN as infinitely low

    df_["RSI_EMA_Cross_Entry"] = rsi_cross_ema_bullish & price_above_ema
    df_["RSI_EMA_Cross_Exit"] = crossed_below_series(df_[rsi_col], df_[rsi_ema_col])
    return df_

# 39. VWAP Aroon
def strategy_vwap_aroon(df, aroon_period=14, aroon_level=70):
    df_ = df.copy()
    # VWAP needs volume. Aroon needs its period. Conditions need +1.
    min_data_needed = aroon_period + 1
    if len(df_) < min_data_needed or df_["volume"].isnull().all():
         # print(f"  Not enough data ({len(df_)}) or no volume for strategy_vwap_aroon, need at least {min_data_needed}")
         df_["VWAP_Aroon_Entry"] = False
         df_["VWAP_Aroon_Exit"] = False
         return df_

    df_["VWAP"] = ta.vwap(df_["high"], df_["low"], df_["close"], df_["volume"])
    aroon_data = ta.aroon(df_["high"], df_["low"], length=aroon_period)

    aroon_up_col = f"AROONU_{aroon_period}"
    aroon_down_col = f"AROOND_{aroon_period}"

    if aroon_data is None or aroon_data.empty or "VWAP" not in df_.columns or df_["VWAP"].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_vwap_aroon (VWAP/Aroon data check before join).")
        df_["VWAP_Aroon_Entry"] = False
        df_["VWAP_Aroon_Exit"] = False
        return df_
    df_ = df_.join(aroon_data) # AROOND_14, AROONU_14, AROONOSC_14

    if aroon_up_col not in df_.columns or aroon_down_col not in df_.columns or df_[aroon_up_col].isnull().all() or df_[aroon_down_col].isnull().all():
        # print(f"  Missing expected Aroon columns or they produced NaNs after join for strategy_vwap_aroon.")
        df_["VWAP_Aroon_Entry"] = False
        df_["VWAP_Aroon_Exit"] = False
        return df_


    price_above_vwap = df_["close"] > df_["VWAP"]
    # Aroon bullish: AroonUp > AroonDown and AroonUp is high
    aroon_bullish = (df_[aroon_up_col] > df_[aroon_down_col]) & (df_[aroon_up_col] > aroon_level)

    df_["VWAP_Aroon_Entry"] = price_above_vwap & aroon_bullish
    df_["VWAP_Aroon_Exit"] = (df_["close"] < df_["VWAP"]) | (df_[aroon_up_col] < df_[aroon_down_col]) # Exit if price drops below VWAP or Aroon flips
    return df_

# 40. Vortex ADX
def strategy_vortex_adx(df, vortex_period=14, adx_period=14, adx_trend_level=25):
    df_ = df.copy()
    # Vortex needs its period. ADX needs its period. Crosses/Levels need +1.
    min_data_needed = max(vortex_period, adx_period) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_vortex_adx, need at least {min_data_needed}")
         df_["Vortex_ADX_Entry"] = False
         df_["Vortex_ADX_Exit"] = False
         return df_

    vortex_data = ta.vortex(df_["high"], df_["low"], df_["close"], length=vortex_period)
    adx_data = ta.adx(df_["high"], df_["low"], df_["close"], length=adx_period)

    vip_col = f"VIP_{vortex_period}" # Vortex Positive Indicator
    vim_col = f"VIM_{vortex_period}" # Vortex Negative Indicator
    adx_col = f"ADX_{adx_period}"

    if vortex_data is None or adx_data is None or vortex_data.empty or adx_data.empty:
        # print(f"  Indicator calculation failed for strategy_vortex_adx (Vortex/ADX data check before join).")
        df_["Vortex_ADX_Entry"] = False
        df_["Vortex_ADX_Exit"] = False
        return df_
    df_ = df_.join(vortex_data)
    df_ = df_.join(adx_data[[adx_col]])

    if vip_col not in df_.columns or vim_col not in df_.columns or adx_col not in df_.columns or df_[vip_col].isnull().all() or df_[vim_col].isnull().all() or df_[adx_col].isnull().all():
        # print(f"  Missing expected Vortex/ADX columns or they produced NaNs after join for strategy_vortex_adx.")
        df_["Vortex_ADX_Entry"] = False
        df_["Vortex_ADX_Exit"] = False
        return df_


    vortex_bullish_cross = crossed_above_series(df_[vip_col], df_[vim_col])
    adx_trending = df_[adx_col] > adx_trend_level

    df_["Vortex_ADX_Entry"] = vortex_bullish_cross & adx_trending
    df_["Vortex_ADX_Exit"] = crossed_below_series(df_[vip_col], df_[vim_col]) | (df_[adx_col] < adx_trend_level - 5) # Exit if Vortex flips or ADX weakens
    return df_

# 41. EMA Ribbon Expansion CMF
def strategy_ema_ribbon_expansion_cmf(df, ema_lengths=[8, 13, 21, 34, 55], expansion_threshold=0.005, cmf_period=20, cmf_level=0): # expansion_threshold is relative
    df_ = df.copy()
    if not ema_lengths or len(ema_lengths) < 2:
         # print("  ema_lengths must be a list of at least two integers for strategy_ema_ribbon_expansion_cmf.")
         df_["EMARibbonExp_CMF_Entry"] = False
         df_["EMARibbonExp_CMF_Exit"] = False
         return df_
    sorted_ema_lengths = sorted(ema_lengths)
    max_ema_len = sorted_ema_lengths[-1]

    # EMAs need max length. CMF needs period + volume. Ribbon expansion check needs +2 (ribbon width + shifted width).
    min_data_needed = max(max_ema_len, cmf_period) + 2
    if len(df_) < min_data_needed or df_["volume"].isnull().all():
         # print(f"  Not enough data ({len(df_)}) or no volume for strategy_ema_ribbon_expansion_cmf, need at least {min_data_needed}")
         df_["EMARibbonExp_CMF_Entry"] = False
         df_["EMARibbonExp_CMF_Exit"] = False
         return df_


    ema_cols = []
    for length in sorted_ema_lengths:
        col_name = f"EMA_{length}"
        df_[col_name] = ta.ema(df_["close"], length=length)
        ema_cols.append(col_name)

    cmf_col = f"CMF_{cmf_period}"
    df_[cmf_col] = ta.cmf(df_["high"], df_["low"], df_["close"], df_["volume"], length=cmf_period)

    if not all(col in df_.columns and not df_[col].isnull().all() for col in ema_cols) or \
       cmf_col not in df_.columns or df_[cmf_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_ema_ribbon_expansion_cmf.")
        df_["EMARibbonExp_CMF_Entry"] = False
        df_["EMARibbonExp_CMF_Exit"] = False
        return df_


    # Bullish ribbon: shortest EMA > ... > longest EMA (or simple shortest > longest)
    is_bullish_ribbon = df_[ema_cols[0]] > df_[ema_cols[-1]]

    # Ribbon expansion: width (shortest - longest) is increasing by a threshold
    ribbon_width = df_[ema_cols[0]] - df_[ema_cols[-1]]
    # Ensure shifted width is not NaN
    shifted_ribbon_width = ribbon_width.shift(1).fillna(0) # Treat initial NaN as 0 width

    ribbon_expanding = ribbon_width > shifted_ribbon_width * (1 + expansion_threshold) # Relative expansion

    cmf_positive = df_[cmf_col] > cmf_level

    df_["EMARibbonExp_CMF_Entry"] = is_bullish_ribbon & ribbon_expanding & cmf_positive
    # Exit: CMF turns negative or ribbon starts contracting
    df_["EMARibbonExp_CMF_Exit"] = (df_[cmf_col] < cmf_level) | (ribbon_width < ribbon_width.shift(1).fillna(np.inf)) # Treat NaN shifted width as inf for contraction check
    return df_

# 42. Ross Hook Momentum
def strategy_ross_hook_momentum(df, ross_lookback=10, momentum_period=10, momentum_level=0):
    df_ = df.copy()
    # Ross Hook detection needs 2*lookback+2. Momentum needs its period. Conditions need +1.
    min_data_needed = max(ross_lookback * 2 + 2, momentum_period) + 1
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_ross_hook_momentum, need at least {min_data_needed}")
         df_["RossHook_Mom_Entry"] = False
         df_["RossHook_Mom_Exit"] = False
         return df_

    # detect_ross_hook creates 'Ross_Hook_Signal' and potentially intermediate columns
    df_ = df.copy() # Work on a copy before calling detect_ross_hook
    df_["Ross_Hook_Signal"] = detect_ross_hook(df_[["high", "low", "close"]].copy(), lookback=ross_lookback) # Pass only needed columns

    momentum_col = f"Momentum_{momentum_period}"
    df_[momentum_col] = ta.mom(df_["close"], length=momentum_period)

    if momentum_col not in df_.columns or df_[momentum_col].isnull().all() or "Ross_Hook_Signal" not in df_.columns or df_["Ross_Hook_Signal"].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_ross_hook_momentum.")
        df_["RossHook_Mom_Entry"] = False
        df_["RossHook_Mom_Exit"] = False
        return df_


    momentum_positive = df_[momentum_col] > momentum_level

    df_["RossHook_Mom_Entry"] = df_["Ross_Hook_Signal"] & momentum_positive
    df_["RossHook_Mom_Exit"] = df_[momentum_col] < momentum_level
    return df_

# 43. RSI Bullish Divergence Candlestick
def strategy_rsi_bullish_divergence_candlestick(df, rsi_period=14, div_lookback=14):
    df_ = df.copy()
    # RSI needs its period. Divergence needs lookback*2 + buffer. Candle pattern needs 1 or 2 bars. Condition needs +1.
    min_data_needed = max(rsi_period, div_lookback * 2) + 2 # Max of periods + div buffer + check on current/prev bar
    # Hammer/Engulfing need open, high, low, close.
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_rsi_bullish_divergence_candlestick, need at least {min_data_needed}")
         df_["RSI_Div_Candle_Entry"] = False
         df_["RSI_Div_Candle_Exit"] = False
         return df_

    rsi_col = f"RSI_{rsi_period}"
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)

    # Candlestick patterns (example: Hammer or Bullish Engulfing)
    # Fillna(0) so the boolean operations work correctly on periods without the pattern
    hammer = (ta.cdl_hammer(df_["open"], df_["high"], df_["low"], df_["close"]).fillna(0) == 100)
    bullish_engulfing = (ta.cdl_engulfing(df_["open"], df_["high"], df_["low"], df_["close"]).fillna(0) == 100)
    is_bullish_candle = hammer | bullish_engulfing


    if rsi_col not in df_.columns or df_[rsi_col].isnull().all():
        # print(f"  RSI calculation failed or produced NaNs for strategy_rsi_bullish_divergence_candlestick.")
        df_["RSI_Div_Candle_Entry"] = False
        df_["RSI_Div_Candle_Exit"] = False
        return df_

    # Divergence on previous bar, bullish candle on current bar
    # Need enough valid RSI data for divergence calculation
    valid_rsi_indices = df_[rsi_col].dropna().index
    if len(valid_rsi_indices) >= div_lookback + 1:
         bullish_divergence = detect_divergence(df_["close"].loc[valid_rsi_indices], df_[rsi_col].loc[valid_rsi_indices], lookback=div_lookback, type='bullish')
         # Shift the divergence signal by 1 bar to act as a confirmation on the *next* bar
         bullish_divergence_prev = bullish_divergence.reindex(df_.index).shift(1).fillna(False)
    else:
         # print(f"  Not enough valid RSI data ({len(valid_rsi_indices)}) for divergence detection (lookback={div_lookback}).")
         bullish_divergence_prev = pd.Series([False] * len(df_), index=df_.index) # Default to False


    df_["RSI_Div_Candle_Entry"] = bullish_divergence_prev & is_bullish_candle
    df_["RSI_Div_Candle_Exit"] = df_[rsi_col] > 70 # Exit if RSI overbought
    return df_


# 45. Ichimoku Basic Combo
def strategy_ichimoku_basic_combo(df, tenkan_period=9, kijun_period=26, senkou_period=52): # senkou_b is usually 52, senkou leading span by kijun_period
    df_ = df.copy()
    # Ichimoku needs max(tenkan, kijun, senkou_b) + kijun_period bars lookahead for Senkou A/B (but pandas_ta calcs it on current bar relative to future)
    # Chikou span needs kijun_period bars *backwards* to check against price kijun_period bars ago.
    # Need max period for calculation (senkou_b is usually longest) + kijun for chikou + some buffer.
    min_data_needed = max(tenkan_period, kijun_period, senkou_period) + kijun_period + 2 # Max calc period + kijun period for chikou shift + buffer

    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_ichimoku_basic_combo, need at least {min_data_needed}")
         df_["Ichimoku_Basic_Entry"] = False
         df_["Ichimoku_Basic_Exit"] = False
         return df_

    # pandas_ta ichimoku returns two dataframes in a tuple: (ichi_df, kumo_df)
    # ichi_df: TENKAN, KIJUN, CHIKOU, SPANA, SPANB on the current time index
    # kumo_df: SPANA, SPANB shifted kijun_period into the future
    ichimoku_data_curr, ichimoku_data_future = ta.ichimoku(df_["high"], df_["low"], df_["close"], tenkan=tenkan_period, kijun=kijun_period, senkou=senkou_period)

    # Column names based on periods
    span_a_col_curr = f"ISA_{tenkan_period}_{kijun_period}_{senkou_period}" # Note: pandas_ta names for current spans might differ
    span_b_col_curr = f"ISB_{tenkan_period}_{kijun_period}_{senkou_period}"
    tenkan_col = f"ITC_{tenkan_period}"
    kijun_col = f"IKC_{kijun_period}"
    chikou_col = f"ISC_{kijun_period}" # Lagging Span

    # Let's check the actual column names returned by pandas_ta
    expected_cols_curr = [tenkan_col, kijun_col, chikou_col]
    expected_cols_kumo = [f"ISA_{tenkan_period}_{kijun_period}", f"ISB_{tenkan_period}_{senkou_period}"] # Kumo uses different naming sometimes

    if ichimoku_data_curr is None or ichimoku_data_curr.empty or \
       ichimoku_data_future is None or ichimoku_data_future.empty:
        # print(f"  Indicator calculation failed for strategy_ichimoku_basic_combo (Ichimoku data check).")
        df_["Ichimoku_Basic_Entry"] = False
        df_["Ichimoku_Basic_Exit"] = False
        return df_

    # Join data back to main dataframe. Use suffixes if necessary, but pandas_ta usually has distinct names
    df_ = df_.join(ichimoku_data_curr)
    # Join future Kumo spans, they are aligned to the *current* index, despite representing future
    df_ = df_.join(ichimoku_data_future.rename(columns={
        f"ISA_{tenkan_period}_{kijun_period}": f"ISA_Future_{tenkan_period}_{kijun_period}",
        f"ISB_{tenkan_period}_{senkou_period}": f"ISB_Future_{tenkan_period}_{senkou_period}"
    }))

    # Check if the needed columns are actually in the DataFrame and are not all NaN
    required_cols = [tenkan_col, kijun_col, chikou_col, f"ISA_Future_{tenkan_period}_{kijun_period}", f"ISB_Future_{tenkan_period}_{senkou_period}"]
    if not all(col in df_.columns and not df_[col].isnull().all() for col in required_cols):
         # print(f"  Missing expected Ichimoku columns or they produced NaNs after join for strategy_ichimoku_basic_combo. Required: {required_cols}. Found: {df_.columns.tolist()}")
         df_["Ichimoku_Basic_Entry"] = False
         df_["Ichimoku_Basic_Exit"] = False
         return df_

    # Use the joined future spans for Kumo checks
    span_a_col_future = f"ISA_Future_{tenkan_period}_{kijun_period}"
    span_b_col_future = f"ISB_Future_{tenkan_period}_{senkou_period}"

    # Price above Kumo: Close > max(Senkou A, Senkou B) (using future spans)
    price_above_kumo = df_["close"] > df_[[span_a_col_future, span_b_col_future]].max(axis=1).fillna(-np.inf) # Treat NaN as infinitely low

    # Tenkan Kijun Cross: Tenkan crosses above Kijun (using current lines)
    tk_cross_bullish = crossed_above_series(df_[tenkan_col], df_[kijun_col])

    # Chikou Span above price: Chikou Span is above the price *kijun_period* bars ago
    # Chikou span is already shifted back by pandas_ta
    # Need to shift price back by kijun_period to compare with Chikou
    price_kijun_period_ago = df_["close"].shift(kijun_period).fillna(-np.inf) # Treat NaN as infinitely low

    chikou_above_price_ago = df_[chikou_col].fillna(-np.inf) > price_kijun_period_ago


    # Basic buy signal: Price is above the future Kumo, Tenkan crosses above Kijun, Chikou Span is above price
    # A common signal might be when Price and TK cross *above* the Kumo, AND Chikou is free above price.
    # This combines the conditions. The TK cross is a trigger, the others are confirmation.
    # Let's use the TK cross as the primary trigger, confirmed by the state of Price and Chikou relative to Kumo/past price.
    entry_cond = tk_cross_bullish & price_above_kumo.shift(1).fillna(False) & chikou_above_price_ago # Price should be above Kumo *before* or *at* the TK cross

    # Refined entry: TK Cross *above* the Kumo
    tk_cross_above_kumo = crossed_above_series(df_[tenkan_col], df_[kijun_col]) & (df_[tenkan_col] > df_[[span_a_col_future, span_b_col_future]].max(axis=1).fillna(-np.inf))

    # Even stronger entry: TK cross + price above kumo + chikou above price ago + future kumo is bullish (ISA > ISB)
    future_kumo_bullish = df_[span_a_col_future].fillna(np.inf) > df_[span_b_col_future].fillna(-np.inf) # Treat NaN appropriately

    df_["Ichimoku_Basic_Entry"] = tk_cross_above_kumo & chikou_above_price_ago & future_kumo_bullish

    # Exit: Tenkan crosses below Kijun OR Price crosses below Kijun OR Price enters Kumo
    df_["Ichimoku_Basic_Exit"] = crossed_below_series(df_[tenkan_col], df_[kijun_col]) | \
                                 crossed_below_series(df_["close"], df_[kijun_col]) | \
                                 (df_["close"] < df_[[span_a_col_future, span_b_col_future]].max(axis=1).fillna(np.inf)) # Treat NaN as infinitely high for below check
    return df_

# 46. Ichimoku Multi-Line (Strong Signal)
def strategy_ichimoku_multi_line(df, tenkan_period=9, kijun_period=26, senkou_period=52):
    df_ = df.copy()
    # Same data requirements as Basic Combo
    min_data_needed = max(tenkan_period, kijun_period, senkou_period) + kijun_period + 2
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_ichimoku_multi_line, need at least {min_data_needed}")
         df_["Ichimoku_Multi_Entry"] = False
         df_["Ichimoku_Multi_Exit"] = False
         return df_

    ichimoku_data_curr, ichimoku_data_future = ta.ichimoku(df_["high"], df_["low"], df_["close"], tenkan=tenkan_period, kijun=kijun_period, senkou=senkou_period)

    tenkan_col = f"ITC_{tenkan_period}"
    kijun_col = f"IKC_{kijun_period}"
    chikou_col = f"ISC_{kijun_period}"
    span_a_col_future = f"ISA_{tenkan_period}_{kijun_period}" # Corrected pandas_ta future naming convention
    span_b_col_future = f"ISB_{tenkan_period}_{senkou_period}"


    if ichimoku_data_curr is None or ichimoku_data_curr.empty or \
       ichimoku_data_future is None or ichimoku_data_future.empty:
        # print(f"  Indicator calculation failed for strategy_ichimoku_multi_line (Ichimoku data check).")
        df_["Ichimoku_Multi_Entry"] = False
        df_["Ichimoku_Multi_Exit"] = False
        return df_

    # Join data back to main dataframe
    df_ = df_.join(ichimoku_data_curr)
    df_ = df_.join(ichimoku_data_future.rename(columns={
        f"ISA_{tenkan_period}_{kijun_period}": span_a_col_future + "_Future", # Append "_Future" to distinguish
        f"ISB_{tenkan_period}_{senkou_period}": span_b_col_future + "_Future"
    }))

    span_a_future_joined_col = span_a_col_future + "_Future"
    span_b_future_joined_col = span_b_col_future + "_Future"


    # Check if the needed columns are actually in the DataFrame and are not all NaN
    required_cols = [tenkan_col, kijun_col, chikou_col, span_a_future_joined_col, span_b_future_joined_col]
    if not all(col in df_.columns and not df_[col].isnull().all() for col in required_cols):
         # print(f"  Missing expected Ichimoku columns or they produced NaNs after join for strategy_ichimoku_multi_line. Required: {required_cols}. Found: {df_.columns.tolist()}")
         df_["Ichimoku_Multi_Entry"] = False
         df_["Ichimoku_Multi_Exit"] = False
         return df_


    # Strong Buy Signal Conditions:
    # 1. Tenkan crosses above Kijun
    tk_cross_bullish = crossed_above_series(df_[tenkan_col], df_[kijun_col])

    # 2. Price is above the Kumo (current or future depends on interpretation, let's use future Kumo for strength)
    price_above_kumo_future = df_["close"] > df_[[span_a_future_joined_col, span_b_future_joined_col]].max(axis=1).fillna(-np.inf)

    # 3. Chikou Span is above the price kijun_period bars ago
    price_kijun_period_ago = df_["close"].shift(kijun_period).fillna(-np.inf)
    chikou_above_price_ago = df_[chikou_col].fillna(-np.inf) > price_kijun_period_ago

    # 4. Future Kumo is bullish (Senkou Span A > Senkou Span B)
    future_kumo_bullish = df_[span_a_future_joined_col].fillna(np.inf) > df_[span_b_future_joined_col].fillna(-np.inf)


    # Strongest signal combines these. The TK cross is the trigger, others are confirmation.
    # Trigger: TK cross happens
    # Confirmation: Price is above Kumo, Chikou is free, Future Kumo is bullish *around the time of the cross*
    # Simplification: Check all conditions are true on the current bar where TK cross occurred
    # This is a very strict signal.

    # Alternative strong signal: Price breaks OUT of Kumo, confirmed by TK cross ABOVE kumo, Chikou free.
    price_breakout_kumo_up = crossed_above_series(df_["close"], df_[[span_a_future_joined_col, span_b_future_joined_col]].max(axis=1).fillna(-np.inf))

    # Let's use a combination: TK cross bullish AND Price is above Kumo AND Chikou is above Price AND Future Kumo is bullish
    df_["Ichimoku_Multi_Entry"] = tk_cross_bullish & price_above_kumo_future & chikou_above_price_ago & future_kumo_bullish

    # Exit: Tenkan crosses below Kijun OR Price crosses below Kijun OR Price enters Kumo
    df_["Ichimoku_Multi_Exit"] = crossed_below_series(df_[tenkan_col], df_[kijun_col]) | \
                                 crossed_below_series(df_["close"], df_[kijun_col]) | \
                                 (df_["close"] < df_[[span_a_future_joined_col, span_b_future_joined_col]].max(axis=1).fillna(np.inf))
    return df_

# 47. EMA SAR
def strategy_ema_sar(df, ema_period=50, initial_af=0.02, max_af=0.2):
    df_ = df.copy()
    # EMA needs its period. PSAR needs data (estimate based on AF). Cross/Level check needs +1.
    min_data_needed = max(ema_period, int(2/initial_af)) + 2
    if len(df_) < min_data_needed:
         # print(f"  Not enough data ({len(df_)}) for strategy_ema_sar, need at least {min_data_needed}")
         df_["EMA_SAR_Entry"] = False
         df_["EMA_SAR_Exit"] = False
         return df_

    ema_col = f"EMA_{ema_period}"
    df_[ema_col] = ta.ema(df_["close"], length=ema_period)
    psar_data = ta.psar(df_["high"], df_["low"], df_["close"], af0=initial_af, af=initial_af, max_af=max_af)

    psar_long_col = f"PSARl_{initial_af}_{max_af}"
    psar_short_col = f"PSARs_{initial_af}_{max_af}"
    psar_reversal_col = f"PSARr_{initial_af}_{max_af}" # Check if this exists


    if ema_col not in df_.columns or df_[ema_col].isnull().all() or psar_data is None or psar_data.empty:
        # print(f"  Indicator calculation failed or produced NaNs for strategy_ema_sar (EMA/PSAR data check before join).")
        df_["EMA_SAR_Entry"] = False
        df_["EMA_SAR_Exit"] = False
        return df_
    df_ = df_.join(psar_data)

    # Check if the necessary PSAR columns exist and are not all NaN after join
    # We need at least one of the PSAR value columns or the reversal column
    if (psar_reversal_col not in df_.columns or df_[psar_reversal_col].isnull().all()) and \
       (psar_long_col not in df_.columns or df_[psar_long_col].isnull().all()) and \
       (psar_short_col not in df_.columns or df_[psar_short_col].isnull().all()):
         # print(f"  Missing expected PSAR columns or they produced NaNs after join for strategy_ema_sar. Found: {df_.columns.tolist()}")
         df_["EMA_SAR_Entry"] = False
         df_["EMA_SAR_Exit"] = False
         return df_


    price_above_ema = df_["close"] > df_[ema_col].fillna(-np.inf)

    # PSAR bullish flip detection logic (same as strategy 15)
    psar_bullish_flip = pd.Series(False, index=df_.index)
    if psar_reversal_col in df_.columns and not df_[psar_reversal_col].isnull().all():
        psar_bullish_flip = df_[psar_reversal_col] == 1
    elif psar_long_col in df_.columns and psar_short_col in df_.columns:
         psar_bullish_flip = (df_[psar_long_col].notna() & df_[psar_long_col].shift(1).isna()) | \
                             (df_[psar_short_col].isna() & df_[psar_short_col].shift(1).notna())
         psar_bullish_flip = psar_bullish_flip & (df_[psar_long_col].fillna(np.inf) < df_["close"])


    df_["EMA_SAR_Entry"] = price_above_ema & psar_bullish_flip

    # PSAR bearish flip detection logic (same as strategy 15)
    psar_bearish_flip = pd.Series(False, index=df_.index)
    if psar_reversal_col in df_.columns and not df_[psar_reversal_col].isnull().all():
        psar_bearish_flip = df_[psar_reversal_col] == -1
    elif psar_long_col in df_.columns and psar_short_col in df_.columns:
         psar_bearish_flip = (df_[psar_short_col].notna() & df_[psar_short_col].shift(1).isna()) | \
                             (df_[psar_long_col].isna() & df_[psar_long_col].shift(1).notna())
         psar_bearish_flip = psar_bearish_flip & (df_[psar_short_col].fillna(-np.inf) > df_["close"])


    df_["EMA_SAR_Exit"] = psar_bearish_flip | crossed_below_series(df_["close"], df_[ema_col])
    return df_

# 48. MFI Bollinger
def strategy_mfi_bollinger(df, mfi_period=14, bb_period=20, bb_std=2.0, mfi_buy_level=20):
    df_ = df.copy()
    # MFI needs its period + volume. BB needs its period. Bounce condition needs +2. Level check needs +1.
    min_data_needed = max(mfi_period, bb_period) + 2
    if len(df_) < min_data_needed or df_["volume"].isnull().all():
         # print(f"  Not enough data ({len(df_)}) or no volume for strategy_mfi_bollinger, need at least {min_data_needed}")
         df_["MFI_BB_Entry_Buy"] = False
         df_["MFI_BB_Exit_Buy"] = False
         return df_

    mfi_col = f"MFI_{mfi_period}"
    df_[mfi_col] = ta.mfi(df_["high"], df_["low"], df_["close"], df_["volume"], length=mfi_period)
    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)

    bbl_col = f"BBL_{bb_period}_{bb_std}"
    bbm_col = f"BBM_{bb_period}_{bb_std}"

    if bbands is None or bbands.empty or mfi_col not in df_.columns or df_[mfi_col].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_mfi_bollinger (BB/MFI data check before join).")
        df_["MFI_BB_Entry_Buy"] = False
        df_["MFI_BB_Exit_Buy"] = False
        return df_
    df_ = df_.join(bbands)

    if bbl_col not in df_.columns or bbm_col not in df_.columns or df_[bbl_col].isnull().all() or df_[bbm_col].isnull().all():
        # print(f"  Missing expected BB columns or they produced NaNs after join for strategy_mfi_bollinger.")
        df_["MFI_BB_Entry_Buy"] = False
        df_["MFI_BB_Exit_Buy"] = False
        return df_

    # MFI recovering from oversold: MFI was <= oversold, now crossed above oversold level
    mfi_recovering_from_oversold = crossed_above_level(df_[mfi_col], mfi_buy_level)

    # Price bouncing off lower BB: low of *previous* bar was <= lower band, and close of *current* bar is > lower band
    # Needs shift(1) on low and lower band.
    shifted_low = df_["low"].shift(1).fillna(np.inf) # Treat NaN low as infinitely high for <= check
    shifted_bbl = df_[bbl_col].shift(1).fillna(np.inf) # Treat NaN BBL as infinitely high

    price_bouncing_lower_bb = (shifted_low <= shifted_bbl) & \
                              (df_["close"] > df_[bbl_col].fillna(-np.inf)) # Treat NaN BBL as infinitely low


    df_["MFI_BB_Entry_Buy"] = mfi_recovering_from_oversold & price_bouncing_lower_bb
    df_["MFI_BB_Exit_Buy"] = crossed_above_level(df_[mfi_col], 50) | \
                             crossed_above_series(df_["close"], df_[bbm_col])
    return df_

# 49. Hammer Volume
def strategy_hammer_volume(df, vol_ma_period=20, vol_factor=1.5):
    df_ = df.copy()
    # Hammer needs open, high, low, close. Vol MA needs its period. Condition needs +1.
    min_data_needed = vol_ma_period + 1
    if len(df_) < min_data_needed or df_["volume"].isnull().all():
         # print(f"  Not enough data ({len(df_)}) or no volume for strategy_hammer_volume, need at least {min_data_needed}")
         df_["Hammer_Vol_Entry"] = False
         df_["Hammer_Vol_Exit"] = False
         return df_

    hammer = ta.cdl_hammer(df_["open"], df_["high"], df_["low"], df_["close"])
    df_["Volume_MA"] = df_["volume"].rolling(window=vol_ma_period).mean()

    if hammer is None or "Volume_MA" not in df_.columns or df_["Volume_MA"].isnull().all():
        # print(f"  Indicator calculation failed or produced NaNs for strategy_hammer_volume (Hammer/VolMA data check).")
        df_["Hammer_Vol_Entry"] = False
        df_["Hammer_Vol_Exit"] = False
        return df_
    df_["Hammer"] = hammer

    # Ensure hammer column is numeric and handle NaNs
    is_hammer = (df_["Hammer"].fillna(0) == 100)
    high_volume = df_["volume"] > vol_factor * df_["Volume_MA"].fillna(0) # Treat NaN Volume_MA as 0

    df_["Hammer_Vol_Entry"] = is_hammer & high_volume
    # Exit if price breaks below the low of the hammer candle
    # This requires storing hammer_low. For simplicity, using a time-based or MA cross exit.
    # Or, more simply: exit if the candle *after* the hammer goes below the hammer's low.
    # Let's store the low of the most recent hammer and exit if close goes below it.
    # Fill NaNs with inf so ffill only keeps actual hammer lows.
    df_["Last_Hammer_Low"] = df_["low"].where(is_hammer).ffill().shift(1).fillna(np.inf) # Low of the last hammer, available on the next bar

    df_["Hammer_Vol_Exit"] = df_["close"] < df_["Last_Hammer_Low"]
    return df_

## strategy_functions.py
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List

# --- Helper Functions ---
def crossed_above_level(series, level):
    """Returns True when series crosses above a fixed level."""
    return (series.shift(1) < level) & (series >= level)

def crossed_below_level(series, level):
    """Returns True when series crosses below a fixed level."""
    return (series.shift(1) > level) & (series <= level)

def crossed_above_series(series1, series2):
    """Returns True when series1 crosses above series2."""
    return (series1.shift(1) <= series2.shift(1)) & (series1 > series2)

def crossed_below_series(series1, series2):
    """Returns True when series1 crosses below series2."""
    return (series1.shift(1) >= series2.shift(1)) & (series1 < series2)

def detect_divergence(price_series, indicator_series, lookback=14, type='bullish'):
    """
    Detects bullish or bearish divergence.
    Bullish: Price makes lower lows, indicator makes higher lows.
    Bearish: Price makes higher highs, indicator makes lower highs.
    """
    if type == 'bullish':
        price_ll = price_series.rolling(window=lookback).min()
        indicator_hl = indicator_series.rolling(window=lookback).min()
        return (price_ll < price_ll.shift(1)) & (indicator_hl >= indicator_hl.shift(1))
    elif type == 'bearish':
        price_hh = price_series.rolling(window=lookback).max()
        indicator_lh = indicator_series.rolling(window=lookback).max()
        return (price_hh > price_hh.shift(1)) & (indicator_lh <= indicator_lh.shift(1))
    return pd.Series([False] * len(price_series), index=price_series.index)

def detect_fractal_high(df_high_series, lookback=2):
    """Detects a fractal high: high is higher than `lookback` periods on both sides."""
    is_fractal = True
    for i in range(1, lookback + 1):
        is_fractal &= (df_high_series > df_high_series.shift(i)) & (df_high_series > df_high_series.shift(-i))
    return is_fractal

def detect_fractal_low(df_low_series, lookback=2):
    """Detects a fractal low: low is lower than `lookback` periods on both sides."""
    is_fractal = True
    for i in range(1, lookback + 1):
        is_fractal &= (df_low_series < df_low_series.shift(i)) & (df_low_series < df_low_series.shift(-i))
    return is_fractal

def detect_ross_hook(df, lookback=10):
    """Detects a Ross Hook: Breakout after a 1-2-3 formation and a shallow pullback."""
    recent_high = df["high"].rolling(window=lookback).max().shift(1)
    recent_low_point1 = df["low"].rolling(window=lookback*2).min().shift(lookback)
    breakout_cond = df["close"] > recent_high
    higher_low_cond = df["low"] > recent_low_point1
    return breakout_cond & higher_low_cond

def detect_hammer(df):
    """Detects a hammer candlestick: small body, long lower wick, little/no upper wick."""
    body = abs(df["close"] - df["open"])
    lower_wick = df["open"].where(df["close"] >= df["open"], df["close"]) - df["low"]
    upper_wick = df["high"] - df["close"].where(df["close"] >= df["open"], df["open"])
    is_hammer = (body < lower_wick * 2) & (lower_wick > upper_wick * 2) & (df["close"] >= df["open"])
    return is_hammer

# --- Strategy Implementations ---

# 1. Momentum Trading
def strategy_momentum(df, rsi_period=14, volume_multiplier=2.0, rsi_level=70):
    df_ = df.copy()
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    df_["Volume_Avg"] = df_["volume"].rolling(window=20).mean()
    entry_cond1 = crossed_above_level(df_[f"RSI_{rsi_period}"], rsi_level)
    entry_cond2 = df_["volume"] > volume_multiplier * df_["Volume_Avg"]
    df_["Momentum_Entry"] = entry_cond1 & entry_cond2
    df_["Momentum_Exit"] = crossed_below_level(df_[f"RSI_{rsi_period}"], rsi_level - 10)
    return df_

# 2. Scalping (Bollinger Bands)
def strategy_scalping(df, bb_period=20, bb_std=2.0):
    df_ = df.copy()
    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)
    if bbands is None or bbands.empty: return df_
    df_[f"BB_Lower"] = bbands[f"BBL_{bb_period}_{bb_std}"]
    df_[f"BB_Upper"] = bbands[f"BBU_{bb_period}_{bb_std}"]
    df_[f"BB_Mid"] = bbands[f"BBM_{bb_period}_{bb_std}"]
    entry_cond_buy = crossed_above_series(df_["close"], df_[f"BB_Lower"])
    entry_cond_sell = crossed_below_series(df_["close"], df_[f"BB_Upper"])
    df_["Scalping_Entry_Buy"] = entry_cond_buy
    df_["Scalping_Entry_Sell"] = entry_cond_sell
    df_["Scalping_Exit_Buy"] = crossed_above_series(df_["close"], df_[f"BB_Mid"])
    df_["Scalping_Exit_Sell"] = crossed_below_series(df_["close"], df_[f"BB_Mid"])
    return df_

# 3. Breakout Trading
def strategy_breakout(df, ema_period=20, volume_multiplier=1.5):
    df_ = df.copy()
    df_[f"EMA_{ema_period}"] = ta.ema(df_["close"], length=ema_period)
    df_["Volume_Avg_Short"] = df_["volume"].rolling(window=ema_period).mean()
    entry_cond1 = crossed_above_series(df_["close"], df_[f"EMA_{ema_period}"])
    entry_cond2 = df_["volume"] > volume_multiplier * df_["Volume_Avg_Short"]
    df_["Breakout_Entry"] = entry_cond1 & entry_cond2
    df_["Breakout_Exit"] = crossed_below_series(df_["close"], df_[f"EMA_{ema_period}"])
    return df_

# 4. Mean Reversion (RSI)
def strategy_mean_reversion(df, rsi_period=14, rsi_upper=70, rsi_lower=30):
    df_ = df.copy()
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    entry_cond_buy = crossed_above_level(df_[f"RSI_{rsi_period}"], rsi_lower)
    entry_cond_sell = crossed_below_level(df_[f"RSI_{rsi_period}"], rsi_upper)
    df_["MeanReversion_Entry_Buy"] = entry_cond_buy
    df_["MeanReversion_Entry_Sell"] = entry_cond_sell
    df_["MeanReversion_Exit_Buy"] = crossed_above_level(df_[f"RSI_{rsi_period}"], 50)
    df_["MeanReversion_Exit_Sell"] = crossed_below_level(df_[f"RSI_{rsi_period}"], 50)
    return df_

# 5. News Trading (Volatility Spike)
def strategy_news(df, volume_multiplier=2.5, price_change_threshold=0.02):
    df_ = df.copy()
    df_["Volume_Avg_VShort"] = df_["volume"].rolling(window=5).mean()
    df_["Price_Change_1m"] = df_["close"].pct_change()
    entry_cond1 = df_["volume"] > volume_multiplier * df_["Volume_Avg_VShort"]
    entry_cond2 = df_["Price_Change_1m"].abs() > price_change_threshold
    df_["News_Entry"] = entry_cond1 & entry_cond2
    df_["News_Exit"] = df_["Price_Change_1m"].abs() < (price_change_threshold / 4)
    return df_

# 6. Trend Following (EMA/ADX)
def strategy_trend_following(df, ema_short=9, ema_long=21, adx_period=14, adx_threshold=25):
    df_ = df.copy()
    df_[f"EMA_{ema_short}"] = ta.ema(df_["close"], length=ema_short)
    df_[f"EMA_{ema_long}"] = ta.ema(df_["close"], length=ema_long)
    adx_data = ta.adx(df_["high"], df_["low"], df_["close"], length=adx_period)
    if adx_data is None or adx_data.empty: return df_
    df_[f"ADX_{adx_period}"] = adx_data[f"ADX_{adx_period}"]
    entry_cond_buy1 = df_["close"] > df_[f"EMA_{ema_short}"]
    entry_cond_buy2 = df_[f"EMA_{ema_short}"] > df_[f"EMA_{ema_long}"]
    entry_cond_buy3 = df_[f"ADX_{adx_period}"] > adx_threshold
    df_["TrendFollowing_Entry_Buy"] = entry_cond_buy1 & entry_cond_buy2 & entry_cond_buy3
    df_["TrendFollowing_Exit_Buy"] = crossed_below_series(df_["close"], df_[f"EMA_{ema_short}"])
    return df_

# 7. Pivot Point (Intraday S/R)
def strategy_pivot_point(df, pivot_lookback=20, exit_ema_period=20):
    df_ = df.copy()
    df_["Recent_High"] = df_["high"].rolling(window=pivot_lookback).max().shift(1)
    df_["Recent_Low"] = df_["low"].rolling(window=pivot_lookback).min().shift(1)
    df_["EMA_Exit"] = ta.ema(df_["close"], length=exit_ema_period)
    entry_cond_buy = crossed_above_series(df_["close"], df_["Recent_High"])
    entry_cond_sell = crossed_below_series(df_["close"], df_["Recent_Low"])
    df_["PivotPoint_Entry_Buy"] = entry_cond_buy
    df_["PivotPoint_Entry_Sell"] = entry_cond_sell
    df_["PivotPoint_Exit_Buy"] = crossed_below_series(df_["close"], df_["EMA_Exit"])
    df_["PivotPoint_Exit_Sell"] = crossed_above_series(df_["close"], df_["EMA_Exit"])
    return df_

# 8. Reversal (RSI/MACD)
def strategy_reversal(df, rsi_period=14, rsi_oversold=30, rsi_overbought=70, macd_fast=12, macd_slow=26, macd_signal=9):
    df_ = df.copy()
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    macd_data = ta.macd(df_["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if macd_data is None or macd_data.empty: return df_
    df_["MACD_Line"] = macd_data[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
    df_["MACD_Signal_Line"] = macd_data[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
    entry_cond_buy1 = df_[f"RSI_{rsi_period}"] < rsi_oversold
    entry_cond_buy2 = crossed_above_series(df_["MACD_Line"], df_["MACD_Signal_Line"])
    entry_cond_sell1 = df_[f"RSI_{rsi_period}"] > rsi_overbought
    entry_cond_sell2 = crossed_below_series(df_["MACD_Line"], df_["MACD_Signal_Line"])
    df_["Reversal_Entry_Buy"] = entry_cond_buy1 & entry_cond_buy2
    df_["Reversal_Entry_Sell"] = entry_cond_sell1 & entry_cond_sell2
    df_["Reversal_Exit_Buy"] = crossed_below_series(df_["MACD_Line"], df_["MACD_Signal_Line"]) | (df_[f"RSI_{rsi_period}"] > 50)
    df_["Reversal_Exit_Sell"] = crossed_above_series(df_["MACD_Line"], df_["MACD_Signal_Line"]) | (df_[f"RSI_{rsi_period}"] < 50)
    return df_

# 9. Pullback Trading (EMA)
def strategy_pullback(df, ema_short=9, ema_long=21):
    df_ = df.copy()
    df_[f"EMA_{ema_short}"] = ta.ema(df_["close"], length=ema_short)
    df_[f"EMA_{ema_long}"] = ta.ema(df_["close"], length=ema_long)
    is_uptrend = df_[f"EMA_{ema_short}"] > df_[f"EMA_{ema_long}"]
    pulled_back_to_ema_short = (df_["low"].shift(1) <= df_[f"EMA_{ema_short}"].shift(1)) & (df_["close"] > df_[f"EMA_{ema_short}"])
    df_["Pullback_Entry_Buy"] = is_uptrend & pulled_back_to_ema_short
    df_["Pullback_Exit_Buy"] = crossed_below_series(df_["close"], df_[f"EMA_{ema_short}"])
    return df_

# 10. End-of-Day (Intraday Consolidation)
def strategy_end_of_day_intraday(df, ema_period=20, volume_multiplier=1.5, price_stability_pct=0.002):
    df_ = df.copy()
    df_[f"EMA_{ema_period}"] = ta.ema(df_["close"], length=ema_period)
    df_["Volume_Avg_Short"] = df_["volume"].rolling(window=ema_period).mean()
    price_is_stable = (df_["close"] / df_[f"EMA_{ema_period}"]).between(1 - price_stability_pct, 1 + price_stability_pct)
    volume_is_decent = df_["volume"] > volume_multiplier * df_["Volume_Avg_Short"]
    df_["EndOfDay_Intraday_Entry"] = price_is_stable & volume_is_decent & (df_["close"] > df_[f"EMA_{ema_period}"])
    df_["EndOfDay_Intraday_Exit"] = (df_["close"].pct_change().abs() > (price_stability_pct * 2)) | crossed_below_series(df_["close"], df_[f"EMA_{ema_period}"])
    return df_

# 11. Golden Cross RSI
def strategy_golden_cross_rsi(df, short_ema=50, long_ema=200, rsi_period=14, rsi_level=50):
    df_ = df.copy(); base_name="GoldenCrossRSI" # Corrected base_name
    min_data_needed = max(long_ema, rsi_period) + 2 # +2 for shift and comparison
    if len(df_) < min_data_needed: return _add_empty_signals(df_, base_name)

    short_ema_col = f"EMA_{short_ema}"
    long_ema_col = f"EMA_{long_ema}"
    rsi_col = f"RSI_{rsi_period}"

    df_[short_ema_col] = ta.ema(df_["close"], length=short_ema)
    df_[long_ema_col] = ta.ema(df_["close"], length=long_ema)
    df_[rsi_col] = ta.rsi(df_["close"], length=rsi_period)

    # Crucial: Check for NaNs after calculation before passing to cross functions
    if df_[short_ema_col].isnull().all() or df_[long_ema_col].isnull().all() or df_[rsi_col].isnull().all():
        return _add_empty_signals(df_, base_name)

    entry_cond1 = crossed_above_series(df_[short_ema_col], df_[long_ema_col])
    entry_cond2 = df_[rsi_col] > rsi_level
    df_[f"{base_name}_Entry_Buy"] = entry_cond1 & entry_cond2
    df_[f"{base_name}_Exit_Buy"] = crossed_below_series(df_[short_ema_col], df_[long_ema_col])
    return df_

# 12. MACD Bullish ADX
def strategy_macd_bullish_adx(df, fast=12, slow=26, signal=9, adx_period=14, adx_level=25):
    df_ = df.copy()
    macd_data = ta.macd(df_["close"], fast=fast, slow=slow, signal=signal)
    adx_data = ta.adx(df_["high"], df_["low"], df_["close"], length=adx_period)
    if macd_data is None or adx_data is None: return df_
    df_["MACD_Line"] = macd_data[f"MACD_{fast}_{slow}_{signal}"]
    df_["MACD_Signal_Line"] = macd_data[f"MACDs_{fast}_{slow}_{signal}"]
    df_["ADX_Line"] = adx_data[f"ADX_{adx_period}"]
    entry_cond1 = crossed_above_series(df_["MACD_Line"], df_["MACD_Signal_Line"])
    entry_cond2 = df_["ADX_Line"] > adx_level
    df_["MACD_ADX_Entry"] = entry_cond1 & entry_cond2
    df_["MACD_ADX_Exit"] = crossed_below_series(df_["MACD_Line"], df_["MACD_Signal_Line"])
    return df_

# 13. MACD RSI Oversold
def strategy_macd_rsi_oversold(df, fast=12, slow=26, signal=9, rsi_period=14, rsi_oversold=30):
    df_ = df.copy()
    macd_data = ta.macd(df_["close"], fast=fast, slow=slow, signal=signal)
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    if macd_data is None: return df_
    df_["MACD_Line"] = macd_data[f"MACD_{fast}_{slow}_{signal}"]
    df_["MACD_Signal_Line"] = macd_data[f"MACDs_{fast}_{slow}_{signal}"]
    entry_cond1 = crossed_above_level(df_[f"RSI_{rsi_period}"], rsi_oversold)
    entry_cond2 = df_["MACD_Line"] > df_["MACD_Signal_Line"]
    df_["MACD_RSI_OS_Entry"] = entry_cond1 & entry_cond2
    df_["MACD_RSI_OS_Exit"] = crossed_below_series(df_["MACD_Line"], df_["MACD_Signal_Line"]) | (df_[f"RSI_{rsi_period}"] > 70)
    return df_

# 14. ADX Heikin Ashi
def strategy_adx_heikin_ashi(df, adx_period=14, adx_level=25):
    df_ = df.copy()
    adx_data = ta.adx(df_["high"], df_["low"], df_["close"], length=adx_period)
    ha_data = ta.ha(df_["open"], df_["high"], df_["low"], df_["close"])
    if adx_data is None or ha_data is None: return df_
    df_["ADX_Line"] = adx_data[f"ADX_{adx_period}"]
    df_["HA_Open"] = ha_data["HA_open"]
    df_["HA_Close"] = ha_data["HA_close"]
    entry_cond1 = df_["ADX_Line"] > adx_level
    entry_cond2 = df_["HA_Close"] > df_["HA_Open"]
    entry_cond3 = df_["HA_Close"].shift(1) <= df_["HA_Open"].shift(1)
    df_["ADX_HA_Entry"] = entry_cond1 & entry_cond2 & entry_cond3
    df_["ADX_HA_Exit"] = df_["HA_Close"] < df_["HA_Open"]
    return df_

# 15. PSAR RSI
def strategy_psar_rsi(df, initial_af=0.02, max_af=0.2, rsi_period=14, rsi_level=50):
    df_ = df.copy()
    psar_data = ta.psar(df_["high"], df_["low"], af0=initial_af, af=initial_af, max_af=max_af)
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    if psar_data is None: return df_
    df_ = df_.join(psar_data)
    psar_long_col = f"PSARl_{initial_af}_{max_af}"
    psar_short_col = f"PSARs_{initial_af}_{max_af}"
    psar_reversal_col = f"PSARr_{initial_af}_{max_af}"
    entry_cond1 = df_[psar_reversal_col] == 1 if psar_reversal_col in df_.columns else df_[psar_long_col].notna() & df_[psar_long_col].shift(1).isna()
    entry_cond2 = df_[f"RSI_{rsi_period}"] > rsi_level
    df_["PSAR_RSI_Entry"] = entry_cond1 & entry_cond2
    exit_cond1 = df_[psar_reversal_col] == -1 if psar_reversal_col in df_.columns else df_[psar_short_col].notna() & df_[psar_short_col].shift(1).isna()
    df_["PSAR_RSI_Exit"] = exit_cond1
    return df_

# 16. VWAP RSI
def strategy_vwap_rsi(df, rsi_period=14, rsi_level=50):
    df_ = df.copy()
    vwap_data = ta.vwap(df_["high"], df_["low"], df_["close"], df_["volume"])
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    if vwap_data is None: return df_
    df_["VWAP"] = vwap_data
    entry_cond1 = crossed_above_series(df_["close"], df_["VWAP"])
    entry_cond2 = df_[f"RSI_{rsi_period}"] > rsi_level
    df_["VWAP_RSI_Entry"] = entry_cond1 & entry_cond2
    df_["VWAP_RSI_Exit"] = crossed_below_series(df_["close"], df_["VWAP"])
    return df_



# 18. ADX Rising MFI Surge
def strategy_adx_rising_mfi_surge(df, adx_period=14, adx_level=25, mfi_period=14, mfi_surge_level=80):
    df_ = df.copy()
    adx_data = ta.adx(df_["high"], df_["low"], df_["close"], length=adx_period)
    mfi_data = ta.mfi(df_["high"], df_["low"], df_["close"], df_["volume"], length=mfi_period)
    if adx_data is None or mfi_data is None: return df_
    df_["ADX_Line"] = adx_data[f"ADX_{adx_period}"]
    df_["MFI"] = mfi_data
    entry_cond1 = df_["ADX_Line"] > adx_level
    entry_cond2 = crossed_above_level(df_["MFI"], mfi_surge_level)
    df_["ADX_MFI_Entry"] = entry_cond1 & entry_cond2
    df_["ADX_MFI_Exit"] = crossed_below_level(df_["MFI"], mfi_surge_level - 20)
    return df_

# 19. Fractal Breakout RSI
def strategy_fractal_breakout_rsi(df, fractal_lookback=2, rsi_period=14, rsi_level=50):
    df_ = df.copy()
    df_["Fractal_High"] = detect_fractal_high(df_["high"], lookback=fractal_lookback)
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    recent_high = df_["high"].rolling(window=fractal_lookback*2+1).max().shift(1)
    entry_cond1 = df_["Fractal_High"] & crossed_above_series(df_["close"], recent_high)
    entry_cond2 = df_[f"RSI_{rsi_period}"] > rsi_level
    df_["Fractal_RSI_Entry"] = entry_cond1 & entry_cond2
    df_["Fractal_RSI_Exit"] = crossed_below_series(df_["close"], df_["low"].rolling(window=fractal_lookback*2+1).min().shift(1))
    return df_

# 20. Chandelier Exit MACD
def strategy_chandelier_exit_macd(df, atr_period=22, atr_mult=3.0, macd_fast=12, macd_slow=26, macd_signal=9):
    df_ = df.copy()
    atr_data = ta.atr(df_["high"], df_["low"], df_["close"], length=atr_period)
    macd_data = ta.macd(df_["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if atr_data is None or macd_data is None: return df_
    df_["ATR"] = atr_data
    df_["MACD_Line"] = macd_data[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
    df_["MACD_Signal_Line"] = macd_data[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
    chandelier_long = df_["high"].rolling(window=atr_period).max() - df_["ATR"] * atr_mult
    entry_cond1 = crossed_above_series(df_["close"], chandelier_long)
    entry_cond2 = crossed_above_series(df_["MACD_Line"], df_["MACD_Signal_Line"])
    df_["CE_MACD_Entry"] = entry_cond1 & entry_cond2
    df_["CE_MACD_Exit"] = crossed_below_series(df_["close"], chandelier_long)
    return df_

# 21. SuperTrend RSI Pullback
def strategy_supertrend_rsi_pullback(df, atr_length=10, factor=3.0, rsi_period=14, rsi_pullback_level=50):
    df_ = df.copy()
    supertrend_data = ta.supertrend(df_["high"], df_["low"], df_["close"], length=atr_length, multiplier=factor)
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    if supertrend_data is None: return df_
    df_["SuperTrend"] = supertrend_data[f"SUPERT_{atr_length}_{factor}"]
    entry_cond1 = df_["close"] > df_["SuperTrend"]
    entry_cond2 = crossed_above_level(df_[f"RSI_{rsi_period}"], rsi_pullback_level)
    df_["SuperTrend_RSI_Entry"] = entry_cond1 & entry_cond2
    df_["SuperTrend_RSI_Exit"] = crossed_below_series(df_["close"], df_["SuperTrend"])
    return df_

# 22. TEMA Cross Volume
def strategy_tema_cross_volume(df, short_tema_len=10, long_tema_len=30, vol_ma_period=20, vol_factor=1.5):
    df_ = df.copy()
    df_["TEMA_Short"] = ta.tema(df_["close"], length=short_tema_len)
    df_["TEMA_Long"] = ta.tema(df_["close"], length=long_tema_len)
    df_["Volume_Avg"] = df_["volume"].rolling(window=vol_ma_period).mean()
    entry_cond1 = crossed_above_series(df_["TEMA_Short"], df_["TEMA_Long"])
    entry_cond2 = df_["volume"] > vol_factor * df_["Volume_Avg"]
    df_["TEMA_Vol_Entry"] = entry_cond1 & entry_cond2
    df_["TEMA_Vol_Exit"] = crossed_below_series(df_["TEMA_Short"], df_["TEMA_Long"])
    return df_

# 23. TSI Resistance Break
def strategy_tsi_resistance_break(df, fast=13, slow=25, signal=13, resistance_period=20):
    df_ = df.copy()
    tsi_data = ta.tsi(df_["close"], fast=fast, slow=slow, signal=signal)
    if tsi_data is None: return df_
    df_["TSI"] = tsi_data[f"TSI_{fast}_{slow}"]
    df_["TSI_Signal"] = tsi_data[f"TSIs_{fast}_{slow}_{signal}"]
    df_["Resistance"] = df_["high"].rolling(window=resistance_period).max().shift(1)
    entry_cond1 = crossed_above_series(df_["TSI"], df_["TSI_Signal"])
    entry_cond2 = crossed_above_series(df_["close"], df_["Resistance"])
    df_["TSI_ResBreak_Entry"] = entry_cond1 & entry_cond2
    df_["TSI_ResBreak_Exit"] = crossed_below_series(df_["TSI"], df_["TSI_Signal"])
    return df_

# 24. TRIX OBV
def strategy_trix_obv(df, trix_period=15, trix_signal=9):
    df_ = df.copy()
    trix_data = ta.trix(df_["close"], length=trix_period, signal=trix_signal)
    obv_data = ta.obv(df_["close"], df_["volume"])
    if trix_data is None or obv_data is None: return df_
    df_["TRIX"] = trix_data[f"TRIX_{trix_period}_{trix_signal}"]
    df_["TRIX_Signal"] = trix_data[f"TRIXs_{trix_period}_{trix_signal}"]
    df_["OBV"] = obv_data
    entry_cond1 = crossed_above_series(df_["TRIX"], df_["TRIX_Signal"])
    entry_cond2 = df_["OBV"] > df_["OBV"].shift(1)
    df_["TRIX_OBV_Entry"] = entry_cond1 & entry_cond2
    df_["TRIX_OBV_Exit"] = crossed_below_series(df_["TRIX"], df_["TRIX_Signal"])
    return df_

# 25. Awesome Oscillator Divergence MACD
def strategy_ao_divergence_macd(df, ao_fast=5, ao_slow=34, macd_fast=12, macd_slow=26, macd_signal=9, div_lookback=14):
    df_ = df.copy()
    ao_data = ta.ao(df_["high"], df_["low"], fast=ao_fast, slow=ao_slow)
    macd_data = ta.macd(df_["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if ao_data is None or macd_data is None: return df_
    df_["AO"] = ao_data
    df_["MACD_Line"] = macd_data[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
    df_["MACD_Signal_Line"] = macd_data[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
    bull_div = detect_divergence(df_["close"], df_["AO"], lookback=div_lookback, type='bullish')
    entry_cond1 = bull_div
    entry_cond2 = crossed_above_series(df_["MACD_Line"], df_["MACD_Signal_Line"])
    df_["AO_MACD_Div_Entry"] = entry_cond1 & entry_cond2
    df_["AO_MACD_Div_Exit"] = crossed_below_series(df_["MACD_Line"], df_["MACD_Signal_Line"])
    return df_

# 26. Heikin Ashi CMO
def strategy_heikin_ashi_cmo(df, cmo_period=14, cmo_level=0):
    df_ = df.copy()
    ha_data = ta.ha(df_["open"], df_["high"], df_["low"], df_["close"])
    cmo_data = ta.cmo(df_["close"], length=cmo_period)
    if ha_data is None or cmo_data is None: return df_
    df_["HA_Open"] = ha_data["HA_open"]
    df_["HA_Close"] = ha_data["HA_close"]
    df_["CMO"] = cmo_data
    entry_cond1 = df_["HA_Close"] > df_["HA_Open"]
    entry_cond2 = crossed_above_level(df_["CMO"], cmo_level)
    df_["HA_CMO_Entry"] = entry_cond1 & entry_cond2
    df_["HA_CMO_Exit"] = df_["HA_Close"] < df_["HA_Open"]
    return df_

# 27. CCI Bollinger
def strategy_cci_bollinger(df, cci_period=20, cci_extreme=-100, bb_period=20, bb_std=2.0):
    df_ = df.copy()
    cci_data = ta.cci(df_["high"], df_["low"], df_["close"], length=cci_period)
    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)
    if cci_data is None or bbands is None: return df_
    df_["CCI"] = cci_data
    df_[f"BB_Lower"] = bbands[f"BBL_{bb_period}_{bb_std}"]
    entry_cond1 = crossed_above_level(df_["CCI"], cci_extreme)
    entry_cond2 = df_["close"] <= df_[f"BB_Lower"]
    df_["CCI_BB_Entry"] = entry_cond1 & entry_cond2
    df_["CCI_BB_Exit"] = crossed_above_series(df_["close"], bbands[f"BBM_{bb_period}_{bb_std}"])
    return df_

# 28. CCI Reversion
def strategy_cci_reversion(df, cci_period=20, cci_extreme_low=-150, cci_revert_level=-100):
    df_ = df.copy()
    cci_data = ta.cci(df_["high"], df_["low"], df_["close"], length=cci_period)
    if cci_data is None: return df_
    df_["CCI"] = cci_data
    entry_cond1 = df_["CCI"] < cci_extreme_low
    entry_cond2 = crossed_above_level(df_["CCI"], cci_revert_level)
    df_["CCI_Revert_Entry"] = entry_cond1 & entry_cond2
    df_["CCI_Revert_Exit"] = crossed_above_level(df_["CCI"], 0)
    return df_

# 29. Keltner RSI Oversold
def strategy_keltner_rsi_oversold(df, kc_length=20, kc_atr_length=10, kc_mult=2.0, rsi_period=14, rsi_oversold=30):
    df_ = df.copy()
    kc_data = ta.kc(df_["high"], df_["low"], df_["close"], length=kc_length, atr_length=kc_atr_length, scalar=kc_mult)
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    if kc_data is None: return df_
    df_[f"KC_Lower"] = kc_data[f"KCL_{kc_length}_{kc_mult}"]
    entry_cond1 = df_["close"] <= df_[f"KC_Lower"]
    entry_cond2 = crossed_above_level(df_[f"RSI_{rsi_period}"], rsi_oversold)
    df_["KC_RSI_OS_Entry"] = entry_cond1 & entry_cond2
    df_["KC_RSI_OS_Exit"] = crossed_above_series(df_["close"], kc_data[f"KCM_{kc_length}_{kc_mult}"])
    return df_

# 30. Keltner MFI Oversold
def strategy_keltner_mfi_oversold(df, kc_length=20, kc_scalar=2.0, mfi_period=14, mfi_oversold=20):
    df_ = df.copy()
    min_data_needed = max(kc_length, mfi_period) + 1
    if len(df_) < min_data_needed or df_['volume'].isnull().all():
        df_["KeltnerMFI_Entry"] = False
        df_["KeltnerMFI_Exit"] = False
        return df_

    kc = ta.kc(df_["high"], df_["low"], df_["close"], length=kc_length, scalar=kc_scalar)
    mfi_col = f"MFI_{mfi_period}"
    df_[mfi_col] = ta.mfi(df_["high"], df_["low"], df_["close"], df_["volume"], length=mfi_period)

    df_ = safe_join(df_, kc)

    kcl_col = f"KCL_{kc_length}_{int(kc_scalar)}"
    entry_cond = (df_["close"] < df_[kcl_col]) & (df_[mfi_col] < mfi_oversold)
    df_["KeltnerMFI_Entry"] = entry_cond
    df_["KeltnerMFI_Exit"] = crossed_above_level(df_[mfi_col], mfi_oversold + 10)
    return df_

# 31. Double MA Pullback
def strategy_double_ma_pullback(df, short_ma_len=20, long_ma_len=50, ma_type="ema"):
    df_ = df.copy()
    ma_func = ta.ema if ma_type.lower() == "ema" else ta.sma
    df_["Short_MA"] = ma_func(df_["close"], length=short_ma_len)
    df_["Long_MA"] = ma_func(df_["close"], length=long_ma_len)
    is_uptrend = df_["Short_MA"] > df_["Long_MA"]
    pullback = (df_["low"].shift(1) <= df_["Short_MA"].shift(1)) & (df_["close"] > df_["Short_MA"])
    df_["MA_Pullback_Entry"] = is_uptrend & pullback
    df_["MA_Pullback_Exit"] = crossed_below_series(df_["close"], df_["Short_MA"])
    return df_

# 32. Bollinger Bounce Volume
def strategy_bollinger_bounce_volume(df, bb_period=20, bb_std=2.0, vol_ma_period=20, vol_factor=1.5):
    df_ = df.copy()
    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)
    if bbands is None: return df_
    df_[f"BB_Lower"] = bbands[f"BBL_{bb_period}_{bb_std}"]
    df_["Volume_Avg"] = df_["volume"].rolling(window=vol_ma_period).mean()
    entry_cond1 = crossed_above_series(df_["close"], df_[f"BB_Lower"])
    entry_cond2 = df_["volume"] > vol_factor * df_["Volume_Avg"]
    df_["BB_Bounce_Vol_Entry"] = entry_cond1 & entry_cond2
    df_["BB_Bounce_Vol_Exit"] = crossed_above_series(df_["close"], bbands[f"BBM_{bb_period}_{bb_std}"])
    return df_

# 33. RSI Range Breakout BB
def strategy_rsi_range_breakout_bb(df, rsi_period=14, rsi_low=40, rsi_high=60, bb_period=20, bb_std=2.0):
    df_ = df.copy()
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)
    if bbands is None: return df_
    df_[f"BB_Upper"] = bbands[f"BBU_{bb_period}_{bb_std}"]
    entry_cond1 = (df_[f"RSI_{rsi_period}"].shift(1).between(rsi_low, rsi_high)) & (df_[f"RSI_{rsi_period}"] > rsi_high)
    entry_cond2 = crossed_above_series(df_["close"], df_[f"BB_Upper"])
    df_["RSI_Range_BB_Entry"] = entry_cond1 & entry_cond2
    df_["RSI_Range_BB_Exit"] = crossed_below_series(df_["close"], bbands[f"BBM_{bb_period}_{bb_std}"])
    return df_

# 34. Keltner Middle RSI Divergence
def strategy_keltner_middle_rsi_divergence(df, kc_length=20, kc_atr_length=10, kc_mult=2.0, rsi_period=14, div_lookback=14):
    df_ = df.copy()
    kc_data = ta.kc(df_["high"], df_["low"], df_["close"], length=kc_length, atr_length=kc_atr_length, scalar=kc_mult)
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    if kc_data is None: return df_
    df_[f"KC_Middle"] = kc_data[f"KCM_{kc_length}_{kc_mult}"]
    bull_div = detect_divergence(df_["close"], df_[f"RSI_{rsi_period}"], lookback=div_lookback, type='bullish')
    entry_cond1 = df_["close"] <= df_[f"KC_Middle"]
    entry_cond2 = bull_div
    df_["KC_Mid_RSI_Div_Entry"] = entry_cond1 & entry_cond2
    df_["KC_Mid_RSI_Div_Exit"] = crossed_above_series(df_["close"], kc_data[f"KCU_{kc_length}_{kc_mult}"])
    return df_

# 35. Hammer on Keltner Volume
def strategy_hammer_on_keltner_volume(df, kc_length=20, kc_atr_length=10, kc_mult=2.0, vol_ma_period=20, vol_factor=1.2):
    df_ = df.copy()
    kc_data = ta.kc(df_["high"], df_["low"], df_["close"], length=kc_length, atr_length=kc_atr_length, scalar=kc_mult)
    if kc_data is None: return df_
    df_[f"KC_Lower"] = kc_data[f"KCL_{kc_length}_{kc_mult}"]
    df_["Volume_Avg"] = df_["volume"].rolling(window=vol_ma_period).mean()
    entry_cond1 = detect_hammer(df_)
    entry_cond2 = df_["close"] <= df_[f"KC_Lower"]
    entry_cond3 = df_["volume"] > vol_factor * df_["Volume_Avg"]
    df_["Hammer_KC_Vol_Entry"] = entry_cond1 & entry_cond2 & entry_cond3
    df_["Hammer_KC_Vol_Exit"] = crossed_above_series(df_["close"], kc_data[f"KCM_{kc_length}_{kc_mult}"])
    return df_

# 36. Bollinger Upper Break Volume
def strategy_bollinger_upper_break_volume(df, bb_period=20, bb_std=2.0, vol_ma_period=20, vol_factor=1.5):
    df_ = df.copy()
    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)
    if bbands is None: return df_
    df_[f"BB_Upper"] = bbands[f"BBU_{bb_period}_{bb_std}"]
    df_["Volume_Avg"] = df_["volume"].rolling(window=vol_ma_period).mean()
    entry_cond1 = crossed_above_series(df_["close"], df_[f"BB_Upper"])
    entry_cond2 = df_["volume"] > vol_factor * df_["Volume_Avg"]
    df_["BB_Break_Vol_Entry"] = entry_cond1 & entry_cond2
    df_["BB_Break_Vol_Exit"] = crossed_below_series(df_["close"], bbands[f"BBM_{bb_period}_{bb_std}"])
    return df_

# 37. RSI EMA Crossover
def strategy_rsi_ema_crossover(df, rsi_period=14, rsi_ma_period=10, ema_period=50):
    df_ = df.copy()
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    df_["RSI_MA"] = ta.ema(df_[f"RSI_{rsi_period}"], length=rsi_ma_period)
    df_["EMA"] = ta.ema(df_["close"], length=ema_period)
    entry_cond1 = crossed_above_series(df_[f"RSI_{rsi_period}"], df_["RSI_MA"])
    entry_cond2 = df_["close"] > df_["EMA"]
    df_["RSI_EMA_Cross_Entry"] = entry_cond1 & entry_cond2
    df_["RSI_EMA_Cross_Exit"] = crossed_below_series(df_["close"], df_["EMA"])
    return df_



# 39. VWAP Aroon
def strategy_vwap_aroon(df, aroon_period=14, aroon_level=70):
    df_ = df.copy()
    vwap_data = ta.vwap(df_["high"], df_["low"], df_["close"], df_["volume"])
    aroon_data = ta.aroon(df_["high"], df_["low"], length=aroon_period)
    if vwap_data is None or aroon_data is None: return df_
    df_["VWAP"] = vwap_data
    df_["Aroon_Up"] = aroon_data[f"AROONU_{aroon_period}"]
    entry_cond1 = crossed_above_series(df_["close"], df_["VWAP"])
    entry_cond2 = df_["Aroon_Up"] > aroon_level
    df_["VWAP_Aroon_Entry"] = entry_cond1 & entry_cond2
    df_["VWAP_Aroon_Exit"] = crossed_below_series(df_["close"], df_["VWAP"])
    return df_

# 40. Vortex ADX
def strategy_vortex_adx(df, vortex_period=14, adx_period=14, adx_trend_level=25):
    df_ = df.copy()
    vortex_data = ta.vortex(df_["high"], df_["low"], df_["close"], length=vortex_period)
    adx_data = ta.adx(df_["high"], df_["low"], df_["close"], length=adx_period)
    if vortex_data is None or adx_data is None: return df_
    df_["Vortex_Plus"] = vortex_data[f"VTXP_{vortex_period}"]
    df_["Vortex_Minus"] = vortex_data[f"VTXM_{vortex_period}"]
    df_["ADX"] = adx_data[f"ADX_{adx_period}"]
    entry_cond1 = crossed_above_series(df_["Vortex_Plus"], df_["Vortex_Minus"])
    entry_cond2 = df_["ADX"] > adx_trend_level
    df_["Vortex_ADX_Entry"] = entry_cond1 & entry_cond2
    df_["Vortex_ADX_Exit"] = crossed_below_series(df_["Vortex_Plus"], df_["Vortex_Minus"])
    return df_

# 41. EMA Ribbon Expansion CMF
def strategy_ema_ribbon_expansion_cmf(df, ema_lengths=[8, 13, 21, 34, 55], expansion_threshold=0.005, cmf_period=20, cmf_level=0.0):
    df_ = df.copy()
    for ema_len in ema_lengths:
        df_[f"EMA_{ema_len}"] = ta.ema(df_["close"], length=ema_len)
    cmf_data = ta.cmf(df_["high"], df_["low"], df_["close"], df_["volume"], length=cmf_period)
    if cmf_data is None: return df_
    df_["CMF"] = cmf_data
    ema_spread = df_[f"EMA_{ema_lengths[0]}"] - df_[f"EMA_{ema_lengths[-1]}"]
    ema_spread_avg = ema_spread.rolling(window=cmf_period).mean()
    expansion = ema_spread > ema_spread_avg * (1 + expansion_threshold)
    entry_cond1 = expansion
    entry_cond2 = df_["CMF"] > cmf_level
    df_["EMARibbonExp_CMF_Entry"] = entry_cond1 & entry_cond2
    df_["EMARibbonExp_CMF_Exit"] = crossed_below_level(df_["CMF"], cmf_level)
    return df_

# 42. Ross Hook Momentum
def strategy_ross_hook_momentum(df, ross_lookback=10, momentum_period=10, momentum_level=0):
    df_ = df.copy()
    df_["Momentum"] = df_["close"].pct_change(periods=momentum_period) * 100
    entry_cond1 = detect_ross_hook(df_, lookback=ross_lookback)
    entry_cond2 = df_["Momentum"] > momentum_level
    df_["RossHook_Mom_Entry"] = entry_cond1 & entry_cond2
    df_["RossHook_Mom_Exit"] = df_["Momentum"] < 0
    return df_

# 43. RSI Bullish Divergence Candlestick
def strategy_rsi_bullish_divergence_candlestick(df, rsi_period=14, div_lookback=14):
    df_ = df.copy()
    df_[f"RSI_{rsi_period}"] = ta.rsi(df_["close"], length=rsi_period)
    bull_div = detect_divergence(df_["close"], df_[f"RSI_{rsi_period}"], lookback=div_lookback, type='bullish')
    entry_cond1 = bull_div
    entry_cond2 = detect_hammer(df_)
    df_["RSI_Div_Candle_Entry"] = entry_cond1 & entry_cond2
    df_["RSI_Div_Candle_Exit"] = crossed_below_level(df_[f"RSI_{rsi_period}"], 50)
    return df_



# In strategy_functions.py

def strategy_ichimoku_basic_combo(df, tenkan_period=9, kijun_period=26, senkou_period=52):
    df_ = df.copy(); base_name = "IchimokuBasic"
    min_data_needed = max(tenkan_period, kijun_period, senkou_period) + kijun_period + 2
    if len(df_) < min_data_needed: return _add_empty_signals(df_, base_name)

    # ichi_results = ta.ichimoku(df_["high"], df_["low"], df_["close"], tenkan=tenkan_period, kijun=kijun_period, senkou=senkou_period, include_kumo=True)
    # if ichi_results is None or not isinstance(ichi_results, tuple) or len(ichi_results) < 2:
    #     return _add_empty_signals(df_, base_name)
    # ichi_df, kumo_df = ichi_results # kumo_df contains future spans
    
    # Using append=True with strategy to add directly to df_
    # This is often easier to manage column names if they are standard
    try:
        df_.ta.ichimoku(tenkan=tenkan_period, kijun=kijun_period, senkou=senkou_period, include_kumo=True, append=True)
    except Exception as e:
        print(f"Error calculating Ichimoku for {base_name}: {e}")
        return _add_empty_signals(df_, base_name)

    # Expected column names after append=True with include_kumo=True
    # Current lines
    tenkan_col = f"ITS_{tenkan_period}"
    kijun_col = f"IKS_{kijun_period}"
    chikou_col = f"ICS_{kijun_period}"
    # Current Kumo (calculated based on current data, not shifted)
    # span_a_curr = f"ISA_{tenkan_period}" # Check actual name
    # span_b_curr = f"ISB_{senkou_period}" # Check actual name
    # Future Kumo (shifted forward, but pandas-ta aligns its index to current df when include_kumo=True)
    span_a_future = f"ISA_{tenkan_period}_{kijun_period}" # Check actual name from df_.columns
    span_b_future = f"ISB_{kijun_period}_{senkou_period}" # Check actual name

    # Verify all expected columns are present
    required_cols = [tenkan_col, kijun_col, chikou_col, span_a_future, span_b_future]
    if not all(col in df_.columns and not df_[col].isnull().all() for col in required_cols):
        # print(f"Ichimoku columns not found/all NaN for {base_name}. Found: {df_.columns}")
        return _add_empty_signals(df_, base_name)

    price_above_kumo = df_["close"] > df_[[span_a_future, span_b_future]].max(axis=1).fillna(-np.inf)
    tk_cross_bullish = crossed_above_series(df_[tenkan_col], df_[kijun_col])
    price_kijun_period_ago = df_["close"].shift(kijun_period).fillna(-np.inf)
    chikou_above_price_ago = df_[chikou_col].fillna(-np.inf) > price_kijun_period_ago
    future_kumo_bullish = df_[span_a_future].fillna(np.inf) > df_[span_b_future].fillna(-np.inf)

    df_[f"{base_name}_Entry_Buy"] = tk_cross_bullish & price_above_kumo & chikou_above_price_ago & future_kumo_bullish
    df_[f"{base_name}_Exit_Buy"] = crossed_below_series(df_[tenkan_col], df_[kijun_col]) | \
                                 (df_["close"] < df_[[span_a_future, span_b_future]].min(axis=1).fillna(np.inf))
    return df_

# Apply similar robust column name handling for strategy_ichimoku_multi_line
def strategy_ichimoku_multi_line(df, tenkan_period=9, kijun_period=26, senkou_period=52):
    df_ = df.copy(); base_name = "IchimokuMultiLine"
    min_data_needed = max(tenkan_period, kijun_period, senkou_period) + kijun_period + 2
    if len(df_) < min_data_needed: return _add_empty_signals(df_, base_name)
    
    try:
        df_.ta.ichimoku(tenkan=tenkan_period, kijun=kijun_period, senkou=senkou_period, include_kumo=True, append=True)
    except Exception as e:
        print(f"Error calculating Ichimoku for {base_name}: {e}")
        return _add_empty_signals(df_, base_name)

    tenkan_col = f"ITS_{tenkan_period}"
    kijun_col = f"IKS_{kijun_period}"
    chikou_col = f"ICS_{kijun_period}"
    span_a_future = f"ISA_{tenkan_period}_{kijun_period}"
    span_b_future = f"ISB_{kijun_period}_{senkou_period}"

    required_cols = [tenkan_col, kijun_col, chikou_col, span_a_future, span_b_future]
    if not all(col in df_.columns and not df_[col].isnull().all() for col in required_cols):
         return _add_empty_signals(df_, base_name)

    tk_cross_bullish = crossed_above_series(df_[tenkan_col], df_[kijun_col])
    price_above_kumo_future = df_["close"] > df_[[span_a_future, span_b_future]].max(axis=1).fillna(-np.inf)
    price_kijun_period_ago = df_["close"].shift(kijun_period).fillna(-np.inf)
    chikou_above_price_ago = df_[chikou_col].fillna(-np.inf) > price_kijun_period_ago
    future_kumo_bullish = df_[span_a_future].fillna(np.inf) > df_[span_b_future].fillna(-np.inf)

    df_[f"{base_name}_Entry_Buy"] = tk_cross_bullish & price_above_kumo_future & chikou_above_price_ago & future_kumo_bullish
    df_[f"{base_name}_Exit_Buy"] = crossed_below_series(df_[tenkan_col], df_[kijun_col]) | \
                                 (df_["close"] < df_[[span_a_future, span_b_future]].min(axis=1).fillna(np.inf))
    return df_

# 47. EMA SAR
def strategy_ema_sar(df, ema_period=50, initial_af=0.02, max_af=0.2):
    df_ = df.copy()
    df_["EMA"] = ta.ema(df_["close"], length=ema_period)
    psar_data = ta.psar(df_["high"], df_["low"], af0=initial_af, af=initial_af, max_af=max_af)
    if psar_data is None: return df_
    df_ = df_.join(psar_data)
    psar_long_col = f"PSARl_{initial_af}_{max_af}"
    entry_cond1 = df_["close"] > df_["EMA"]
    entry_cond2 = df_[psar_long_col].notna() & df_[psar_long_col].shift(1).isna()
    df_["EMA_SAR_Entry"] = entry_cond1 & entry_cond2
    df_["EMA_SAR_Exit"] = df_[f"PSARs_{initial_af}_{max_af}"].notna()
    return df_

# 48. MFI Bollinger
def strategy_mfi_bollinger(df, mfi_period=14, bb_period=20, bb_std=2.0, mfi_buy_level=20, mfi_sell_level=80):
    df_ = df.copy()
    mfi_data = ta.mfi(df_["high"], df_["low"], df_["close"], df_["volume"], length=mfi_period)
    bbands = ta.bbands(df_["close"], length=bb_period, std=bb_std)
    if mfi_data is None or bbands is None: return df_
    df_["MFI"] = mfi_data
    df_[f"BB_Lower"] = bbands[f"BBL_{bb_period}_{bb_std}"]
    df_[f"BB_Upper"] = bbands[f"BBU_{bb_period}_{bb_std}"]
    entry_cond_buy1 = crossed_above_level(df_["MFI"], mfi_buy_level)
    entry_cond_buy2 = df_["close"] <= df_[f"BB_Lower"]
    entry_cond_sell1 = crossed_below_level(df_["MFI"], mfi_sell_level)
    entry_cond_sell2 = df_["close"] >= df_[f"BB_Upper"]
    df_["MFI_BB_Entry_Buy"] = entry_cond_buy1 & entry_cond_buy2
    df_["MFI_BB_Entry_Sell"] = entry_cond_sell1 & entry_cond_sell2
    df_["MFI_BB_Exit_Buy"] = crossed_above_series(df_["close"], bbands[f"BBM_{bb_period}_{bb_std}"])
    df_["MFI_BB_Exit_Sell"] = crossed_below_series(df_["close"], bbands[f"BBM_{bb_period}_{bb_std}"])
    return df_

# 49. Hammer Volume
def strategy_hammer_volume(df, vol_ma_period=20, vol_factor=1.5):
    df_ = df.copy()
    df_["Volume_Avg"] = df_["volume"].rolling(window=vol_ma_period).mean()
    entry_cond1 = detect_hammer(df_)
    entry_cond2 = df_["volume"] > vol_factor * df_["Volume_Avg"]
    df_["Hammer_Vol_Entry"] = entry_cond1 & entry_cond2
    df_["Hammer_Vol_Exit"] = crossed_below_series(df_["close"], df_["low"].rolling(window=5).min())
    return df_


# Dictionary to hold all strategy functions
ALL_STRATEGIES = {
    "Momentum Trading": strategy_momentum,
    "Scalping (Bollinger Bands)": strategy_scalping,
    "Breakout Trading": strategy_breakout,
    "Mean Reversion (RSI)": strategy_mean_reversion,
    "News Trading (Volatility Spike)": strategy_news,
    "Trend Following (EMA/ADX)": strategy_trend_following,
    "Pivot Point (Intraday S/R)": strategy_pivot_point,
    "Reversal (RSI/MACD)": strategy_reversal,
    "Pullback Trading (EMA)": strategy_pullback,
    "End-of-Day (Intraday Consolidation)": strategy_end_of_day_intraday,
    "Golden Cross RSI": strategy_golden_cross_rsi,
    "MACD Bullish ADX": strategy_macd_bullish_adx,
    "MACD RSI Oversold": strategy_macd_rsi_oversold,
    "ADX Heikin Ashi": strategy_adx_heikin_ashi,
    "PSAR RSI": strategy_psar_rsi,
    "VWAP RSI": strategy_vwap_rsi,
    "ADX Rising MFI Surge": strategy_adx_rising_mfi_surge,
    "Fractal Breakout RSI": strategy_fractal_breakout_rsi,
    "Chandelier Exit MACD": strategy_chandelier_exit_macd,
    "SuperTrend RSI Pullback": strategy_supertrend_rsi_pullback,
    "TEMA Cross Volume": strategy_tema_cross_volume,
    "TSI Resistance Break": strategy_tsi_resistance_break,
    "TRIX OBV": strategy_trix_obv,
    "Awesome Oscillator Divergence MACD": strategy_ao_divergence_macd,
    "Heikin Ashi CMO": strategy_heikin_ashi_cmo,
    "CCI Bollinger": strategy_cci_bollinger,
    "CCI Reversion": strategy_cci_reversion,
    "Keltner RSI Oversold": strategy_keltner_rsi_oversold,
    "Keltner MFI Oversold": strategy_keltner_mfi_oversold,
    "Double MA Pullback": strategy_double_ma_pullback,
    "Bollinger Bounce Volume": strategy_bollinger_bounce_volume,
    "RSI Range Breakout BB": strategy_rsi_range_breakout_bb,
    "Keltner Middle RSI Divergence": strategy_keltner_middle_rsi_divergence,
    "Hammer on Keltner Volume": strategy_hammer_on_keltner_volume,
    "Bollinger Upper Break Volume": strategy_bollinger_upper_break_volume,
    "RSI EMA Crossover": strategy_rsi_ema_crossover,
    "VWAP Aroon": strategy_vwap_aroon,
    "Vortex ADX": strategy_vortex_adx,
    "EMA Ribbon Expansion CMF": strategy_ema_ribbon_expansion_cmf,
    "Ross Hook Momentum": strategy_ross_hook_momentum,
    "RSI Bullish Divergence Candlestick": strategy_rsi_bullish_divergence_candlestick,
    "Ichimoku Basic Combo": strategy_ichimoku_basic_combo,
    "Ichimoku Multi-Line": strategy_ichimoku_multi_line,
    "EMA SAR": strategy_ema_sar,
    "MFI Bollinger": strategy_mfi_bollinger,
    "Hammer Volume": strategy_hammer_volume
}

if __name__ == '__main__':
    # Example: Create a dummy DataFrame for testing
    idx = pd.date_range(start='2023-01-01 09:30', periods=200, freq='1min')
    close_prices = 100 + np.random.randn(200).cumsum() * 0.1
    open_prices = close_prices - np.random.rand(200) * 0.2
    high_prices = np.maximum(close_prices, open_prices) + np.random.rand(200) * 0.1
    low_prices = np.minimum(close_prices, open_prices) - np.random.rand(200) * 0.1
    volumes = np.random.randint(1000, 5000, 200)
    sample_df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=idx)

    print("Testing Momentum Strategy:")
    df_momentum_test = strategy_momentum(sample_df.copy())
    print(df_momentum_test[['close', 'RSI_14', 'Volume_Avg', 'Momentum_Entry', 'Momentum_Exit']].tail())

    print("\nTesting Scalping Strategy:")
    df_scalping_test = strategy_scalping(sample_df.copy())
    print(df_scalping_test[['close', 'BB_Lower', 'BB_Upper', 'Scalping_Entry_Buy', 'Scalping_Entry_Sell']].tail())

    print("\nTesting Golden Cross RSI Strategy:")
    df_gc_rsi_test = strategy_golden_cross_rsi(sample_df.copy())
    print(df_gc_rsi_test[['close', 'EMA_50', 'EMA_200', 'RSI_14', 'GC_RSI_Entry']].tail())