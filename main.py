from fastapi import FastAPI
import requests
from data_utils import get_stock_data
from dotenv import load_dotenv
import os
from models import PredictRequest

load_dotenv()

app = FastAPI()

AZURE_ML_ENDPOINT = os.getenv("AZURE_ML_ENDPOINT")
AZURE_ML_API_KEY = os.getenv("AZURE_ML_API_KEY")


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        df = get_stock_data(req.stock_code, req.last_close, req.predict_date)

        payload = {
            "data": df[["Open", "High", "Low", "Close", "Volume", "MA5", "MA25"]].values.tolist()
        }
        headers = {
            "Authorization": f"Bearer {AZURE_ML_API_KEY}"
        }
        response = requests.post(
            AZURE_ML_ENDPOINT, json=payload, headers=headers)
        prediction = response.json()

        return {
            "dates": df.index.strftime("%Y-%m-%d").tolist(),
            "predicted_values": prediction
        }
    except Exception as e:
        return {"error": str(e)}
