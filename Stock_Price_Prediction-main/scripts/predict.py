import pickle
import pandas as pd
from features import add_features

# Load latest data
df = pd.read_csv('data/stock_data.csv')
df = add_features(df)

X = df[['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 'Return', 'Volatility', 'Lag_1']]
latest_input = X.tail(1)

# Load model
model = pickle.load(open("model.pkl", "rb"))
predicted_price = model.predict(latest_input)

print(f"Predicted Next-Day Closing Price: ₹{predicted_price[0]:.2f}")