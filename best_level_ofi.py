import pandas as pd

def calc_best_level_ofi(df: pd.DataFrame, window: str | None = None) -> pd.Series:
    """
    Calculate best-level OFI (Order Flow Imbalance)
    The input DataFrame must contain: ts_event, bid_px_00, bid_sz_00, ask_px_00, ask_sz_00
    :param window: Time window string (e.g., '1min', '5s', etc.). If None, returns event-level OFI; otherwise, returns window-aggregated OFI.
    :return: OFI series indexed by ts_event
    """
    # Previous row's order book data
    prev_bid_px = df['bid_px_00'].shift(1)
    prev_bid_sz = df['bid_sz_00'].shift(1)
    prev_ask_px = df['ask_px_00'].shift(1)
    prev_ask_sz = df['ask_sz_00'].shift(1)

    # Best bid contribution
    bid_contrib = (df['bid_px_00'] > prev_bid_px) * df['bid_sz_00'] \
                + (df['bid_px_00'] == prev_bid_px) * (df['bid_sz_00'] - prev_bid_sz)
    # Best ask contribution
    ask_contrib = (df['ask_px_00'] < prev_ask_px) * df['ask_sz_00'] \
                + (df['ask_px_00'] == prev_ask_px) * (prev_ask_sz - df['ask_sz_00'])

    ofi = bid_contrib - ask_contrib
    ofi.iloc[0] = 0  # The first row cannot be calculated, set to 0

    if window is not None:
        # Aggregate by time window
        ofi_agg = ofi.set_axis(df['ts_event']).resample(window).sum()
        return ofi_agg
    else:
        # Event-level OFI, also indexed by ts_event
        ofi.index = df['ts_event']
        return ofi

if __name__ == "__main__":
    from data_loader import load_raw_events
    selected_columns = ['bid_px_00', 'bid_sz_00', 'ask_px_00', 'ask_sz_00']
    df = load_raw_events('first_25000_rows.csv', columns=selected_columns)
    # event-level OFI
    ofi = calc_best_level_ofi(df)
    print(ofi.head(10))
    # 1min window OFI
    ofi_1min = calc_best_level_ofi(df, window='1min')
    print(ofi_1min.head(10)) 