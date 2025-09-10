from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model at startup
model = joblib.load("model.pkl")

# Define input data format
class InputData(BaseModel):
    features: list

# Create FastAPI app
app = FastAPI()
@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": prediction[0]}