# === app/main.py ===
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import time
from xgboost import XGBRegressor
from app.model import preprocess_and_predict

app = FastAPI()

# Load model and metadata
model = XGBRegressor()
model.load_model("models/xgb_rank_model.json")  # Use XGBoost's load_model, not joblib

metadata = joblib.load("models/floor_price_metadata.pkl")
domain_means = metadata['domain_means']
country_freq = metadata['country_freq']
X_train_columns = metadata['X_train_columns']

# Input Schema
class InputData(BaseModel):
    Country: str
    Domain: str
    Browser: str
    Os: str

# Output Schema
class PredictionOutput(BaseModel):
    predicted_rank: int
    prediction_time_ms: float

@app.post("/predict", response_model=PredictionOutput)
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    start_time = time.time()
    prediction = preprocess_and_predict(df, model, domain_means, country_freq, X_train_columns)
    end_time = time.time()
    return PredictionOutput(
        predicted_rank=int(prediction[0]),
        prediction_time_ms=(end_time - start_time) * 1000
    )
