import pandas as pd

def calc_multi_level_ofi(df: pd.DataFrame, levels: list[int] = list(range(10)), window: str | None = None) -> pd.DataFrame:
    """
    Calculate multi-level OFI (Level 0 to 9), optionally aggregated over a time window.
    Returns a DataFrame with one scaled OFI column per level, indexed by ts_event or window.
    """
    # 确保 index 是 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        df = df.set_index('ts_event')

    result = {}

    for level in levels:
        bid_px = f'bid_px_0{level}'
        bid_sz = f'bid_sz_0{level}'
        ask_px = f'ask_px_0{level}'
        ask_sz = f'ask_sz_0{level}'

        # Check columns exist
        if not all(col in df.columns for col in [bid_px, bid_sz, ask_px, ask_sz]):
            raise ValueError(f"Missing columns for level {level}: {bid_px}, {bid_sz}, {ask_px}, {ask_sz}")

        # Previous snapshot
        prev_bid_px = df[bid_px].shift(1)
        prev_bid_sz = df[bid_sz].shift(1)
        prev_ask_px = df[ask_px].shift(1)
        prev_ask_sz = df[ask_sz].shift(1)

        bid_contrib = (df[bid_px] > prev_bid_px) * df[bid_sz] \
                    + (df[bid_px] == prev_bid_px) * (df[bid_sz] - prev_bid_sz)

        ask_contrib = (df[ask_px] < prev_ask_px) * df[ask_sz] \
                    + (df[ask_px] == prev_ask_px) * (prev_ask_sz - df[ask_sz])

        ofi = bid_contrib - ask_contrib
        ofi.iloc[0] = 0  # First row undefined

        # calculate average depth
        avg_depth = (df[bid_sz] + df[ask_sz]) / 2
       
        if window is not None:
            avg_depth = avg_depth.rolling(window=window, min_periods=1).mean()
        # normalize
        scaled_ofi = ofi / avg_depth.replace(0, pd.NA)
        result[f'ofi_lvl{level+1}'] = scaled_ofi

    ofi_df = pd.DataFrame(result)
    ofi_df.index = df.index

    if window is not None:
        ofi_df = ofi_df.resample(window).sum()

    return ofi_df

if __name__ == "__main__":
    from data_loader import load_raw_events
    # Modify level range as needed
    levels = list(range(2))
    # get all required columns
    selected_columns = []
    for level in levels:
        selected_columns += [
            f'bid_px_0{level}',
            f'bid_sz_0{level}',
            f'ask_px_0{level}',
            f'ask_sz_0{level}',
        ]
    selected_columns = list(set(selected_columns))
    df = load_raw_events('first_25000_rows.csv', columns=selected_columns)
    
    # Calculate OFI for all levels
    ofi = calc_multi_level_ofi(df, levels=levels)
    ofi['multi_level_ofi'] = ofi.sum(axis=1)
    print(f"Event-level {len(levels)}-level multi_level_ofi:")
    print(ofi['multi_level_ofi'].head(10))
    
    # 1min window multi_level_ofi
    ofi_1min = calc_multi_level_ofi(df, levels=levels, window='1min')
    ofi_1min['multi_level_ofi'] = ofi_1min.sum(axis=1)
    print(f"\n1min window {len(levels)}-level multi_level_ofi:")
    print(ofi_1min['multi_level_ofi'].head(10)) 