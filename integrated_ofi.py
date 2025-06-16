import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from multi_level_ofi import calc_multi_level_ofi
from data_loader import load_raw_events

def compute_integrated_ofi(ofi_df: pd.DataFrame) -> pd.Series:
    """
    Input: ofi_df DataFrame containing multi-level OFI, each column represents one level of OFI
    Output: Series of integrated OFI
    """
    # 1. Extract first principal component using PCA
    pca = PCA(n_components=1)
    pca.fit(ofi_df.values)
    w1 = pca.components_[0]  # shape: (n_levels,)
    # 2. L1 normalization
    w1_norm = w1 / np.sum(np.abs(w1))
    # 3. Weighted sum
    integrated_ofi = ofi_df.values @ w1_norm
    return pd.Series(integrated_ofi, index=ofi_df.index, name='integrated_ofi')

if __name__ == "__main__":
    # 1. Load raw data
    levels = list(range(10))
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

    # 2. Calculate multi-level OFI (normalized)
    ofi_df = calc_multi_level_ofi(df, levels=levels)

    # 3. Calculate Integrated OFI
    integrated_ofi = compute_integrated_ofi(ofi_df)
    print("First 10 rows of Integrated OFI:")
    print(integrated_ofi.head(10))
