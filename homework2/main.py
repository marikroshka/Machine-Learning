from fastapi import FastAPI, HTTPException
from joblib import load
import pandas as pd
from pydantic import BaseModel

# FastAPI application for deposit prediction

app = FastAPI()

class ModelInput(BaseModel):
    duration: float
    campaign: int
    contact_unknown: int
    housing_yes: int
    poutcome_success: int
    balance: float
    contact_cellular: int
    day: int
    month_may: int
    loan_no: int #all these fields are the top 10 features from the model

pipeline = load('model10_features.joblib') # Load the trained model with top 10 features

@app.get("/")
def read_root():
    return {"message":  "Deposit Prediction API. Use POST /predict to get predictions."} #for checking if the API is running


@app.post("/predict")
async def predict(data: ModelInput):
    try:
        df = pd.DataFrame([data.dict()])
       
        proba = pipeline.predict_proba(df)[0][1]  # Probability of the positive class
        prediction = pipeline.predict(df)[0]  # Class prediction
        return {
            "status": "success",
            "probability": float(proba),
            "prediction": int(prediction)   # for returning the prediction and probability
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) #for handling exceptions and returning error messages