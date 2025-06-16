import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from integrated_ofi import compute_integrated_ofi
from multi_level_ofi import calc_multi_level_ofi
from data_loader import load_raw_events

def generate_mock_ofi(real_ofi: pd.Series, symbol: str, noise_level: float = 0.3) -> pd.Series:
    """
    Generates mock OFI data based on real OFI.
    Args:
        real_ofi: The real OFI series.
        symbol: The stock symbol.
        noise_level: The noise level.
    Returns:
        The mocked OFI series.
    """
    # Add random noise and lag effects
    noise = np.random.normal(0, noise_level, len(real_ofi))
    lag_effect = np.roll(real_ofi.values, np.random.randint(1, 5)) * 0.5
    mock_ofi = real_ofi.values + noise + lag_effect
    return pd.Series(mock_ofi, index=real_ofi.index, name=f'ofi_{symbol}')

def compute_returns(mid_prices: pd.Series) -> pd.Series:
    """
    Compute log returns.
    """
    # Ensure the price series is monotonically increasing by index
    mid_prices = mid_prices.sort_index()
    # Compute log returns
    returns = np.log(mid_prices / mid_prices.shift(1))
    return returns

def analyze_cross_asset_ofi():
    # 1. Load AAPL data
    print("Loading data...")
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
    print(f"Raw data shape: {df.shape}")
    print(f"Raw data index type: {type(df.index)}")
    print(f"Raw data index sample: {df.index[:5]}")
    
    # Check price columns in raw data
    print("\nChecking raw data:")
    print("Columns:", df.columns.tolist())
    print("\nbid_px_00 sample values:")
    print(df['bid_px_00'].head())
    print("\nask_px_00 sample values:")
    print(df['ask_px_00'].head())
    
    # 2. Compute Integrated OFI for AAPL
    print("\nCalculating Integrated OFI...")
    ofi_df = calc_multi_level_ofi(df, levels=levels)
    print(f"OFI data shape: {ofi_df.shape}")
    print(f"OFI data index sample: {ofi_df.index[:5]}")
    
    aapl_ofi = compute_integrated_ofi(ofi_df)
    print(f"Integrated OFI shape: {aapl_ofi.shape}")
    print(f"Integrated OFI index sample: {aapl_ofi.index[:5]}")
    
    # 3. Compute returns for AAPL
    print("\nCalculating returns...")
    # Use the same timestamp index as the OFI data
    mid_price = pd.Series(
        (df['bid_px_00'].values + df['ask_px_00'].values) / 2,
        index=ofi_df.index
    )
    
    # Check price data
    print("\nChecking price data:")
    print(f"Price series shape: {mid_price.shape}")
    print(f"Number of NaNs in price series: {mid_price.isna().sum()}")
    print(f"Price series sample values: {mid_price.head()}")
    print(f"Price series statistics:\n{mid_price.describe()}")
    
    # Ensure the price series is sorted by time
    mid_price = mid_price.sort_index()
    aapl_returns = compute_returns(mid_price)
    
    print("\nChecking returns data:")
    print(f"Returns series shape: {aapl_returns.shape}")
    print(f"Number of NaNs in returns series: {aapl_returns.isna().sum()}")
    print(f"Returns series sample values: {aapl_returns.head()}")
    print(f"Returns series statistics:\n{aapl_returns.describe()}")
    
    # 4. Generate mock OFI data for other stocks
    print("\nGenerating mock OFI data...")
    symbols = ['MSFT', 'GOOG', 'NVDA', 'META']
    mock_ofi_dict = {
        'AAPL': aapl_ofi,
        **{symbol: generate_mock_ofi(aapl_ofi, symbol) for symbol in symbols}
    }
    
    # 5. Build the feature matrix
    ofi_matrix = pd.DataFrame(mock_ofi_dict)
    print(f"OFI matrix shape: {ofi_matrix.shape}")
    print(f"OFI matrix index sample: {ofi_matrix.index[:5]}")
    
    # 6. Align data
    print("\nAligning data...")
    # Ensure all data uses the same index
    common_index = ofi_matrix.index.intersection(aapl_returns.index)
    if len(common_index) == 0:
        print("Warning: No common index values found!")
        print(f"OFI matrix index type: {type(ofi_matrix.index)}")
        print(f"Returns series index type: {type(aapl_returns.index)}")
        print(f"OFI matrix index sample: {ofi_matrix.index[:5]}")
        print(f"Returns series index sample: {aapl_returns.index[:5]}")
        raise ValueError("No common index values found!")
    
    print(f"Number of common indices: {len(common_index)}")
    print(f"Common index sample: {common_index[:5]}")
    
    X = ofi_matrix.loc[common_index]
    y = aapl_returns.loc[common_index]
    
    # 7. Data cleaning
    print("\nCleaning data...")
    # Remove rows with NaN values
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"Data shape after cleaning: X={X.shape}, y={y.shape}")
    
    if len(X) == 0:
        raise ValueError("No valid data after cleaning! Please check the return calculation process.")
    
    # 8. Use Lasso regression
    print("\nStarting Lasso regression...")
    model = LassoCV(cv=5, random_state=42)
    model.fit(X, y)
    
    # 9. Analyze results
    coef_df = pd.Series(model.coef_, index=X.columns)
    coef_df = coef_df.sort_values(key=abs, ascending=False)
    
    print("\nCross-Asset OFI Impact Analysis Results:")
    print("=" * 50)
    print("\nImpact coefficients of asset OFIs on AAPL returns:")
    print(coef_df)
    
    print("\nModel R² score:", model.score(X, y))
    
    # 10. Visualize results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    coef_df.plot(kind='bar')
    plt.title('Cross-Asset OFI Impact on AAPL Returns')
    plt.xlabel('Assets')
    plt.ylabel('Impact Coefficient')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cross_asset_ofi_impact.png')
    plt.close()

if __name__ == "__main__":
    analyze_cross_asset_ofi() 