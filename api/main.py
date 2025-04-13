from fastapi import FastAPI, Request
import pickle
import pandas as pd

app = FastAPI()

# Carica il modello e gli encoder
model = pickle.load(open("notebook/model.pkl", "rb"))
label_encoders = pickle.load(open("notebook/label_encoders.pkl", "rb"))

@app.get("/")
def root():
    return {"status": "API is running 🎉"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    if isinstance(data, dict):  # singolo record
        df = pd.DataFrame([data])
    else:  # lista di record
        df = pd.DataFrame(data)

    # encoding
    for col in df.columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])

    preds = model.predict(df)
    pred_labels = label_encoders["class"].inverse_transform(preds)

    return {"predictions": pred_labels.tolist()}
