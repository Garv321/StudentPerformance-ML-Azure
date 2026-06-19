from fastapi import FastAPI, HTTPException
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI(
    title="Student Performance ML API",
    description="Predict student scores using ML model",
    version="1.0"
)

# Load model once (efficient)
pipeline = PredictPipeline()

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Student Performance ML API is live 🚀"
    }

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Prediction
        result = pipeline.predict(df)

        return {
            "prediction": float(result[0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))