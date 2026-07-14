from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import pandas as pd
import boto3
import joblib
import os

# ==========================
# FastAPI App
# ==========================

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# ==========================
# S3 Configuration
# ==========================

BUCKET_NAME = "model-bootcamp"
MODEL_KEY = "customer_churn_model.joblib"
LOCAL_MODEL = "customer_churn_model.joblib"

print("Downloading model from S3...")

if not os.path.exists(LOCAL_MODEL):
    s3 = boto3.client("s3")
    s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL)

print("Loading model...")

saved = joblib.load(LOCAL_MODEL)

model = saved["model"]
feature_names = saved["feature_names"]

print("Model loaded successfully!")

# ==========================
# Request Model
# ==========================

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ==========================
# Routes
# ==========================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get("/health")
def health():
    return {"status": "healthy"}
@app.post("/predict")
def predict(data: Customer):

    df = pd.DataFrame([data.model_dump()])

    # Ensure columns match training order
    df = df[feature_names]

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "probability": round(float(probability), 4)
    }