import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

from features import add_features

df = pd.read_csv('data/stock_data.csv')
df = add_features(df)

X = df[['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 'Return', 'Volatility', 'Lag_1']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, pred))
print("R² Score:", r2_score(y_test, pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))