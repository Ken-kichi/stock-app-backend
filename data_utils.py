import yfinance as yf
import pandas as pd


def get_stock_data(stock_code: str, last_close: float, predict_date: str, years: int = 3) -> pd.DataFrame:
    """
    Retrieve and prepare historical stock data for a given company to be used
    in machine learning-based price prediction.

    This function fetches several years of historical stock prices from Yahoo Finance,
    appends the most recent closing price provided by the user, generates
    moving average features, and aligns the target variable (future price)
    to the specified prediction date.

    Args:
        stock_code (str): The stock code (e.g., "7203" for Toyota Motor Corporation).
        last_close (float): The most recent closing price provided by the user.
        predict_date (str): The target date (in YYYY-MM-DD format) for which the stock price will be predicted.
        years (int, optional): Number of years of historical data to retrieve. Defaults to 3.

    Returns:
        pd.DataFrame: A processed DataFrame containing historical stock data with
        added moving averages and a target column corresponding to the specified future date.

    Notes:
        - The function automatically appends the user-provided closing price to the last
          available trading day prior to the prediction date.
        - The "MA5" and "MA25" columns represent 5-day and 25-day moving averages.
        - The "target" column is created by shifting the "Close" column so that it
          represents the stock price on the specified prediction date.
    """
    df = yf.Ticker(stock_code + ".T").history(period=f"{years}y")

    latest_date = pd.to_datetime(predict_date) - pd.Timedelta(days=1)
    df.loc[latest_date] = [last_close]*5 + [0]

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA25"] = df["Close"].rolling(25).mean()

    target_date = pd.to_datetime(predict_date)
    df["target"] = df["Close"].shift((target_date - df.index).days * -1)

    df = df.dropna()
    return df
