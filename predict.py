# predict.py
import os
import sys
import joblib
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from features import add_technical_indicators, make_supervised
from fetch_data import fetch_symbol_data
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_or_train_model(symbol: str):
    """Load model if exists; otherwise train on the fly."""
    model_path = os.path.join(MODEL_DIR, f"{symbol}_gb_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.joblib")
    features_path = os.path.join(MODEL_DIR, f"{symbol}_features.joblib")

    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        return model, scaler, features
    else:
        print(f"Model for {symbol} not found. Training now...")
        from train_model import train_for_symbol
        train_for_symbol(symbol)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            features = joblib.load(features_path)
            return model, scaler, features
        else:
            print(f"Failed to train {symbol}.")
            sys.exit(1)


def prepare_features(df):
    """Add technical indicators and prepare supervised dataset."""
    df = add_technical_indicators(df)
    X, y = make_supervised(df, n_lag=1)
    return X, y


def predict_stock(ticker: str):
    ticker = ticker.upper()
    model, scaler, trained_features = load_or_train_model(ticker)

    print(f"\nFetching latest data for {ticker}...")
    end = datetime.today()
    start = end - timedelta(days=365)
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        print("No data fetched. Check ticker symbol.")
        return

    df = add_technical_indicators(df)
    X, y = make_supervised(df, n_lag=1)
    df['Original_Close'] = df['Close'].iloc[-len(X):].values

    # Align features
    for col in trained_features:
        if col not in X.columns:
            X[col] = 0
    X = X[trained_features]

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    df_predictions = df.iloc[-len(predictions):].copy()
    df_predictions['Prediction'] = predictions

    latest_pred = float(predictions[-1])
    latest_close = float(df_predictions['Original_Close'].iloc[-1])
    signal = "BUY" if latest_pred > latest_close else "SELL"

    print(f"\nLatest Prediction for {ticker}: {latest_pred:.2f}")
    print(f"Signal: {signal}")
    return df_predictions


if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    df_pred = predict_stock(ticker)
    if df_pred is not None:
        print(df_pred.tail())
