# train_model.py
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from fetch_data import fetch_symbol_data
from features import add_technical_indicators, make_supervised

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_for_symbol(symbol: str, days: int = 720, n_lag: int = 20):
    """Train a Gradient Boosting model for a given stock symbol."""
    print(f"Fetching data for {symbol}...")
    df = fetch_symbol_data(symbol, period_days=days)
    df = add_technical_indicators(df)
    X, y = make_supervised(df, n_lag=n_lag)

    if len(X) < 50:
        print(f"Not enough data for {symbol} (found {len(X)} rows). Skipping training.")
        return

    # Split train/test
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_train, y_test = y.iloc[:split].copy(), y.iloc[split:].copy()

    # Clean NaNs and infs
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Gradient Boosting Regressor with tuned hyperparameters
    model = GradientBoostingRegressor(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    print("Training Gradient Boosting model...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    preds = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, preds))
    print(f"Test RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # Save model, scaler, features
    model_path = os.path.join(MODEL_DIR, f"{symbol}_gb_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.joblib")
    features_path = os.path.join(MODEL_DIR, f"{symbol}_features.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(X_train.columns.tolist(), features_path)

    print(f"Saved model -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")
    print(f"Saved features -> {features_path}")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    train_for_symbol(symbol, days=720, n_lag=20)
