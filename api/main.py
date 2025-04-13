from fastapi import FastAPI, Request
import pickle
import pandas as pd

app = FastAPI()

model = pickle.load(open("notebook/model.pkl", "rb"))
label_encoders = pickle.load(open("notebook/label_encoders.pkl", "rb"))

@app.get("/")
def root():
    return {"status": "API is running 🎉"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    input_df = pd.DataFrame([data])

    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    pred_encoded = model.predict(input_df)[0]
    pred_label = label_encoders["class"].inverse_transform([pred_encoded])[0]

    return {"prediction": pred_label}