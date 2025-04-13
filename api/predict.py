
# # Local predictions

import requests
import pandas as pd


df = pd.read_csv("data/mushrooms.csv")
X = df.drop("class", axis=1)


payload = [X.iloc[i].to_dict() for i in range(5)]


url = "https://mushroom-api-413607434670.europe-west1.run.app/predict"  
res = requests.post(url, json=payload)

print(res.json())

