from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load your ML model
model = pickle.load(open("model.pkl", "rb"))

# Pydantic model for input validation
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "ML API working!"}

@app.post("/predict")
def predict(data: InputData):
    # Convert input list to numpy array
    X = np.array(data.features).reshape(1, -1)
    pred = model.predict(X)[0]
    return {"prediction": int(pred)}

# New feature: return probabilities
@app.post("/predict_proba")
def predict_proba(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0].tolist()
        return {"probabilities": probs}
    else:
        return {"error": "Model does not support probability prediction."}

@app.get("/health")
def health():
    return {"status": "ok"}
