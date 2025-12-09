from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "ML API working!"}

@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    pred = model.predict(data)[0]
    return {"prediction": int(pred)}

@app.get("/health")
def health():
    return {"status": "ok"}
