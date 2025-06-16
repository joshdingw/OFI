import pandas as pd

def load_raw_events(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    '''
    read raw events from csv file
    :param path: path to the csv file
    :param columns: list of columns to select (ts_event will be automatically included)
    :return: DataFrame with events
    '''
    df = pd.read_csv(path, parse_dates=['ts_event'])
    
    if columns is not None:
        if 'ts_event' not in columns:
            columns = ['ts_event'] + columns
        df = df[columns]
    
    return df.sort_values('ts_event').reset_index(drop=True)

if __name__ == "__main__":
    # Example: select first order book data
    selected_columns = ['bid_px_00', 'bid_sz_00', 'ask_px_00', 'ask_sz_00']
    df = load_raw_events('first_25000_rows.csv', columns=selected_columns)
    print(df.head(10))