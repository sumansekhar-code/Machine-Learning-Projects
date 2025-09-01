import streamlit as st
import yfinance as yf
import pandas as pd
import pickle

def add_features(df):
    df = df.copy()
    df['MA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['MA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['Return'] = df['Close'].pct_change().fillna(0)
    df['Volatility'] = df['Return'].rolling(window=5, min_periods=1).std().fillna(0)
    df['Lag_1'] = df['Close'].shift(1).fillna(method='bfill')
    
    # For training only
    if 'Target' not in df.columns:
        df['Target'] = df['Close'].shift(-1)
    
    df.dropna(inplace=True)
    return df

st.title("📈 Stock Price Prediction (Next Day)")

ticker = st.text_input("Enter Stock Symbol (e.g., AMZN):", "AMZN")

if st.button("Fetch and Predict"):
    df = pd.read_csv('data/stock_data.csv')
    df = add_features(df)
    st.line_chart(df['Close'])

    X = df[['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 'Return', 'Volatility', 'Lag_1']]
    latest = X.tail(1)

    model = pickle.load(open("model.pkl", "rb"))
    pred = model.predict(latest)
    st.success(f"✅ Predicted Next-Day Price: ₹{pred[0]:.2f}")
