from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Data model for stock price prediction requests.

    This model represents the input payload sent from the frontend to the backend API.
    It includes the stock code, the most recent closing price, and the target date
    for which the model should predict the future stock price.

    Attributes:
        stock_code (str): The stock code (e.g., "7203" for Toyota Motor Corporation).
        last_close (float): The most recent closing price (e.g., 1500.0).
            Used as the latest data point for prediction.
        predict_date (str): The target date for prediction in YYYY-MM-DD format.
            The machine learning model will predict the stock price on this date.
    """

    stock_code: str = Field(..., example="7203")
    last_close: float = Field(..., example=1500.0)
    predict_date: str = Field(..., example="2024-12-31")
