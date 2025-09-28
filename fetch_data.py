# fetch_data.py
# import yfinance as yf
# import pandas as pd
# from datetime import datetime, timedelta

# def fetch_symbol_data(symbol: str, period_days: int = 365, interval: str = "1d") -> pd.DataFrame:
#     """
#     Fetch historical OHLCV data for `symbol` for the past `period_days`.
#     Returns a pandas DataFrame with Date as index.
#     """
#     end = datetime.utcnow()
#     start = end - timedelta(days=period_days)
#     df = yf.download(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval=interval, progress=False)
#     if df.empty:
#         raise ValueError(f"No data returned for {symbol}. Check symbol and connection.")
#     df = df.reset_index().rename(columns={"Date": "date"})
#     df['date'] = pd.to_datetime(df['date'])
#     df.set_index('date', inplace=True)
#     return df

# if __name__ == "__main__":
#     import sys
#     sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
#     df = fetch_symbol_data(sym, period_days=365, interval="1d")
#     print(df.tail())


# fetch_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_symbol_data(symbol: str, period_days: int = 365, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for `symbol` for the past `period_days`.
    Returns a pandas DataFrame with Date as index.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=period_days)
    # set auto_adjust explicitly to avoid futurewarning about its default
    df = yf.download(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                    interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}. Check symbol and connection.")
    df = df.reset_index().rename(columns={"Date": "date"})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df = fetch_symbol_data(sym, period_days=365, interval="1d")
    print(df.tail())
