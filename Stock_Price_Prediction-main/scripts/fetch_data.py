import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker='AMZN', start='2019-01-01', end='2023-12-31'):
    df = yf.download(ticker, start=start, end=end)
    df.to_csv('data/stock_data.csv')
    print("Data saved to data/stock_data.csv")

if __name__ == "__main__":
    fetch_stock_data()
