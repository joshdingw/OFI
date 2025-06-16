# Order Flow Imbalance (OFI) Feature Construction

## Features Implemented

The following OFI features are constructed as per the task requirements:

1.  **Best-Level OFI**: This is the most fundamental OFI calculation, derived from changes at the best bid and ask prices (top of the book).
    -   Implementation: `best_level_ofi.py`

2.  **Multi-Level OFI**: This feature extends the OFI calculation to multiple levels of the order book, providing a more comprehensive view of the market depth and order flow. The OFI at each level is normalized by the average depth at that level.
    -   Implementation: `multi_level_ofi.py`

3.  **Integrated OFI**: To reduce dimensionality and capture the most significant information from the multi-level OFI, Principal Component Analysis (PCA) is used. The Integrated OFI is a weighted sum of the multi-level OFIs, where the weights are determined by the first principal component.
    -   Implementation: `integrated_ofi.py`

4.  **Cross-Asset OFI**: This module demonstrates how to analyze the impact of order flow from multiple assets on a single target asset's returns. It uses Lasso regression to model the relationship.
    -   Implementation: `cross_asset_ofi.py`

### **Important Note on Cross-Asset OFI Data**

In the `cross_asset_ofi.py` script, the analysis is performed on Apple (AAPL) as the target asset. Due to the absence of real order book data for other stocks, the OFI for other tickers (e.g., MSFT, GOOG, NVDA, META) is **synthetically generated mock data**. This is for illustrative purposes to demonstrate the methodology of cross-asset analysis. In a real-world scenario, you would use actual data for all assets involved.

## Project Structure

-   `data_loader.py`: A utility to load raw event data from CSV files.
-   `best_level_ofi.py`: Computes OFI using only the best bid/ask level.
-   `multi_level_ofi.py`: Computes OFI for multiple order book levels.
-   `integrated_ofi.py`: Computes a single integrated OFI from multi-level OFIs using PCA.
-   `cross_asset_ofi.py`: Analyzes the impact of OFI from multiple assets on a target asset's returns.
-   `first_25000_rows.csv`: Sample data file used for the analysis.

## How to Run

Each feature calculation script can be run independently to see the output.

```bash
python best_level_ofi.py
python multi_level_ofi.py
python integrated_ofi.py
python cross_asset_ofi.py
```

This will execute the example code within the `if __name__ == "__main__":` block in each file. The cross-asset analysis will also generate a plot named `cross_asset_ofi_impact.png`.
