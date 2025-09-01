import pandas as pd

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
