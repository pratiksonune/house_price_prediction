from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import pickle
import numpy as np
import json
import uvicorn

app = FastAPI()

# Load the trained model
with open("House_price_predication_model.pickle", "rb") as f:
    model = pickle.load(f)

# Load column names from JSON
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

locations = data_columns[3:]  # Extracting locations from column names

class HousePriceResponse(BaseModel):
    location: str
    sqft: int
    bath: int
    bhk: int
    estimated_price: str  # Add predicted price field

def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # Find location index in feature list
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1  # One-hot encoding for location

    return model.predict([x])[0]  # Predict using the trained model

@app.post("/predict/")
def get_prediction(
    location: str = Form(...),
    sqft: int = Form(...),
    bath: int = Form(...),
    bhk: int = Form(...)
):
    # Validate location
    if location not in data_columns:
        raise HTTPException(status_code=400, detail="Invalid location")

    # Predict price
    final_price = predict_price(location, sqft, bath, bhk)

    # Format price in lakhs/crores
    if final_price < 100:
        formatted_price = f"₹ {round(final_price, 2)} lakh"
    else:
        formatted_price = f"₹ {round(final_price / 100, 2)} crore"

    # Return the response
    return {
        "location": location,
        "sqft": sqft,
        "bath": bath,
        "bhk": bhk,
        "estimated_price": formatted_price
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
