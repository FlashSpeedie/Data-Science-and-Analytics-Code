import pandas as pd
import numpy as np
import torch
import joblib
from train_model import HousePriceModel

input_size = 13
model = HousePriceModel(input_size)
model.load_state_dict(torch.load("house_price_model.pth"))
model.eval()
scaler = joblib.load("scaler.pkl")

dataset_path = "housing_data.csv"
df = pd.read_csv(dataset_path)

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1, inplace=True)
df.fillna(df.median(), inplace=True)

df.fillna(df.median(), inplace=True)

X = df.drop('price', axis=1).values
X_scaled = scaler.transform(X)

X_tensor = torch.tensor(X_scaled[-10:], dtype=torch.float32)

with torch.no_grad():
    predictions = model(X_tensor).numpy()

price_scaler = joblib.load("price_scaler.pkl")
predicted_prices = price_scaler.inverse_transform(predictions.flatten().reshape(-1, 1)).flatten()

for idx, price in enumerate(predicted_prices, start=1):
    print(f"Prediction {idx}: ${price:,.2f}")
