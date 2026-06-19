
from fastapi import FastAPI
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    result = PredictPipeline().predict(df)
    return {"prediction": int(result[0])}
