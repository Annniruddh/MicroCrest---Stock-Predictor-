# # microcrest_app.py
# import streamlit as st
# import pandas as pd
# import yfinance as yf
# import joblib
# import json
# import os
# from datetime import datetime, timedelta
# from features import add_technical_indicators, make_supervised

# USERS_FILE = "users.json"

# # ------------------------
# # Helper functions
# # ------------------------

# def load_model_files(symbol):
#     """Load model, scaler, and feature list for a given symbol."""
#     model_path = f"models/{symbol}_rf_model.joblib"
#     scaler_path = f"models/{symbol}_scaler.joblib"
#     features_path = f"models/{symbol}_features.joblib"

#     missing_files = [p for p in [model_path, scaler_path, features_path] if not os.path.exists(p)]
#     if missing_files:
#         st.error(f"Missing files for {symbol}: {', '.join(missing_files)}. Please run train_model.py first.")
#         return None, None, None

#     model = joblib.load(model_path)
#     scaler = joblib.load(scaler_path)
#     features = joblib.load(features_path)
#     return model, scaler, features

# def prepare_features(df):
#     df = add_technical_indicators(df)
#     df_supervised = make_supervised(df, n_lag=1)[0]  # Only X
#     df_supervised['Original_Close'] = df['Close'].iloc[-len(df_supervised):].values
#     return df_supervised

# def predict_stock(ticker):
#     ticker = ticker.upper()
#     model, scaler, trained_features = load_model_files(ticker)
#     if model is None:
#         return None, None, None

#     st.info(f"Fetching latest data for {ticker}...")
#     end = datetime.today()
#     start = end - timedelta(days=365)

#     df = yf.download(ticker, start=start, end=end)
#     if df.empty:
#         st.error("No data fetched. Check ticker symbol.")
#         return None, None, None

#     df = prepare_features(df)

#     # Align features
#     for col in trained_features:
#         if col not in df.columns:
#             df[col] = 0
#     X = df[trained_features]

#     # Scale
#     X_scaled = scaler.transform(X)

#     # Predict
#     predictions = model.predict(X_scaled)
#     df["Prediction"] = predictions

#     latest_pred = float(predictions[-1])
#     latest_close = float(df['Original_Close'].iloc[-1])
#     signal = "BUY" if latest_pred > latest_close else "SELL"

#     return df, latest_pred, signal

# # ------------------------
# # User management
# # ------------------------

# def load_users():
#     if not os.path.exists(USERS_FILE):
#         with open(USERS_FILE, "w") as f:
#             json.dump({}, f)
#     with open(USERS_FILE, "r") as f:
#         users = json.load(f)
#     return users

# def save_users(users):
#     with open(USERS_FILE, "w") as f:
#         json.dump(users, f)

# def signup(username, password):
#     users = load_users()
#     if username in users:
#         return False, "Username already exists."
#     users[username] = password
#     save_users(users)
#     return True, "Signup successful! You can now login."

# def signin(username, password):
#     users = load_users()
#     if username in users and users[username] == password:
#         return True
#     return False

# # ------------------------
# # Streamlit App
# # ------------------------

# st.set_page_config(page_title="MicroCrest", layout="wide")
# st.title("MicroCrest - Stock Prediction App")

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
#     st.session_state.username = ""

# # ------------- Login / Signup Section -------------
# if not st.session_state.logged_in:
#     auth_option = st.selectbox("Select Option", ["Sign In", "Sign Up"])

#     with st.form("auth_form"):
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         submitted = st.form_submit_button(auth_option)

#         if submitted:
#             if auth_option == "Sign Up":
#                 success, message = signup(username, password)
#                 if success:
#                     st.success(message)
#                 else:
#                     st.error(message)
#             else:  # Sign In
#                 if signin(username, password):
#                     st.session_state.logged_in = True
#                     st.session_state.username = username
#                     st.success(f"Welcome, {username}!")
#                     st.stop()  # Refresh app to show main content
#                 else:
#                     st.error("Invalid username or password.")
#     st.stop()

# # ------------- Main App -------------
# st.sidebar.subheader(f"Welcome, {st.session_state.username}!")

# # ------------- Logout -------------
# if st.sidebar.button("Logout"):
#     st.session_state.logged_in = False
#     st.session_state.username = ""
#     # Update query params to trigger refresh
#     st.query_params["logged_in"] = ["false"]
#     st.stop()

# ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOG)", "AAPL")

# if st.sidebar.button("Predict"):
#     with st.spinner(f"Fetching and predicting {ticker_input}..."):
#         df, latest_pred, signal = predict_stock(ticker_input)
#         if df is not None:
#             st.success(f"Latest Prediction for {ticker_input}: {latest_pred:.2f} → {signal}")

#             # Display last 5 rows only
#             st.subheader("Recent Data")
#             st.dataframe(df.tail())

























git remote add origin https://github.com/Annniruddh/MicroCrest---Stock-Predictor-.git










# microcrest_app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import json
import os
import hashlib
from datetime import datetime, timedelta
import numpy as np
from features import add_technical_indicators, make_supervised
from fetch_data import fetch_symbol_data
import plotly.graph_objs as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

USERS_FILE = "users.json"

# ------------------------
# Helper functions
# ------------------------

def train_model_on_fly(symbol, days=720, n_lag=20):
    """Train model if it doesn't exist for the symbol (silent, no Streamlit messages)."""
    df = fetch_symbol_data(symbol, period_days=days)
    df = add_technical_indicators(df)
    X, y = make_supervised(df, n_lag=n_lag)

    if len(X) < 50:
        return None, None, None

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_train, y_test = y.iloc[:split].copy(), y.iloc[split:].copy()

    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate silently
    preds = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, preds))
    # No Streamlit messages here

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{symbol}_gb_model.joblib")
    joblib.dump(scaler, f"models/{symbol}_scaler.joblib")
    joblib.dump(X_train.columns.tolist(), f"models/{symbol}_features.joblib")

    return model, scaler, X_train.columns.tolist()


def load_or_train_model(symbol):
    model_path = f"models/{symbol}_gb_model.joblib"
    scaler_path = f"models/{symbol}_scaler.joblib"
    features_path = f"models/{symbol}_features.joblib"

    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        return model, scaler, features
    else:
        # Silent training
        return train_model_on_fly(symbol)


def predict_stock(ticker):
    ticker = ticker.upper()
    model, scaler, trained_features = load_or_train_model(ticker)
    if model is None:
        return None, None, None

    end = datetime.today()
    start = end - timedelta(days=365)

    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return None, None, None

    df = add_technical_indicators(df)
    df_supervised = make_supervised(df, n_lag=1)[0]
    df_supervised['Original_Close'] = df['Close'].iloc[-len(df_supervised):].values

    # Align features
    for col in trained_features:
        if col not in df_supervised.columns:
            df_supervised[col] = 0
    X = df_supervised[trained_features]

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    df_supervised["Prediction"] = predictions

    latest_pred = float(predictions[-1])
    latest_close = float(df_supervised['Original_Close'].iloc[-1])
    signal = "BUY" if latest_pred > latest_close else "SELL"

    return df_supervised, latest_pred, signal


def plot_predictions(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Original_Close'],
        mode='lines+markers',
        name='Actual Close',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Prediction'],
        mode='lines+markers',
        name='Predicted Close',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title=f"{ticker} - Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------
# User management
# ------------------------

def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
    return users

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup(username, password, first_name, last_name, email):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        "password": hash_password(password),
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
    }
    save_users(users)
    return True, "Signup successful! You can now login."

def signin(username, password):
    users = load_users()
    if username in users:
        if isinstance(users[username], str):
            return users[username] == password
        return users[username]["password"] == hash_password(password)
    return False

def get_user_profile(username):
    users = load_users()
    profile = users.get(username, {})
    if isinstance(profile, str):
        profile = {"password": profile, "first_name": "", "last_name": "", "email": ""}
        users[username] = profile
        save_users(users)
    return profile

def update_user_profile(username, updated_data):
    users = load_users()
    if username in users:
        users[username].update(updated_data)
        save_users(users)
        return True
    return False


# ------------------------
# Streamlit App
# ------------------------

st.set_page_config(page_title="MicroCrest", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1a73e8;'>📈 MicroCrest – Stock Prediction</h1>", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# Login/Signup
if not st.session_state.logged_in:
    st.subheader("Authentication")
    auth_option = st.radio("Choose an option", ["LogIn", "Sign Up"], horizontal=True)
    with st.form("auth_form"):
        first_name = last_name = username = email = password = ""
        if auth_option == "Sign Up":
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
        else:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(auth_option)
        if submitted:
            if auth_option == "Sign Up":
                if not (first_name and last_name and username and email and password):
                    st.error("All fields are required.")
                else:
                    success, message = signup(username, password, first_name, last_name, email)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            else:
                if signin(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome, {username}!")
                    st.stop()
                else:
                    st.error("Invalid username or password.")
    st.stop()

# Sidebar
st.sidebar.subheader(f"👋 Welcome, {st.session_state.username}!")
menu = st.sidebar.radio("Navigate", ["Home", "Profile"])

# ---------- Home Tab ----------
if menu == "Home":
    st.subheader("Stock Prediction")
    ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOG, MSFT)", "")
    if st.button("Predict"):
        with st.spinner(f"Fetching and predicting {ticker_input}..."):
            df, latest_pred, signal = predict_stock(ticker_input)
            if df is not None:
                color = "green" if signal=="BUY" else "red"
                st.markdown(f"<h3>Latest Prediction for {ticker_input}: <span style='color:{color}'>{latest_pred:.2f} → {signal}</span></h3>", unsafe_allow_html=True)
                st.subheader("Recent Data")
                st.dataframe(df.tail())
                plot_predictions(df, ticker_input)

# ---------- Profile Tab ----------
elif menu == "Profile":
    st.subheader("Your Profile")
    profile = get_user_profile(st.session_state.username)
    with st.form("profile_form"):
        first_name = st.text_input("First Name", profile.get("first_name", ""))
        last_name = st.text_input("Last Name", profile.get("last_name", ""))
        email = st.text_input("Email", profile.get("email", ""))
        update_btn = st.form_submit_button("Update Profile")
        if update_btn:
            update_user_profile(st.session_state.username, {"first_name": first_name, "last_name": last_name, "email": email})
            st.success("Profile updated successfully!")

# ---------- Logout ----------
st.sidebar.markdown("---")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("You have been logged out.")
