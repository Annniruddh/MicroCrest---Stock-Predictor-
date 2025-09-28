# features.py
import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # SMA/EMA
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    # Returns and volatility
    df['RET_1'] = df['Close'].pct_change(1)
    df['RET_5'] = df['Close'].pct_change(5)
    df['ROLL_STD_5'] = df['RET_1'].rolling(5).std()
    # RSI (14)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=13, adjust=False).mean()
    ma_down = down.ewm(com=13, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Bollinger Bands (20-day)
    df['BB_MID'] = df['Close'].rolling(20).mean()
    df['BB_STD'] = df['Close'].rolling(20).std()
    df['BB_UP'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LOW'] = df['BB_MID'] - 2 * df['BB_STD']
    # Volume features
    df['VOL_5'] = df['Volume'].rolling(5).mean()
    # Drop rows with NaN
    df = df.dropna().copy()
    return df

def make_supervised(df: pd.DataFrame, n_lag: int = 1):
    """
    Create dataset where X are features at day t and y is Close at day t + n_lag.
    For next-day prediction set n_lag=1.
    """
    df = df.copy()
    df['target_close'] = df['Close'].shift(-n_lag)
    df = df.dropna().copy()
    X = df.drop(columns=['target_close'])
    y = df['target_close']
    return X, y

if __name__ == "__main__":
    import yfinance as yf
    df = yf.download("AAPL", period="180d", interval="1d", progress=False)
    df = df.reset_index().set_index('Date')
    df = add_technical_indicators(df)
    X, y = make_supervised(df, n_lag=1)
    print(X.columns)
    print("X shape:", X.shape, "y shape:", y.shape)
