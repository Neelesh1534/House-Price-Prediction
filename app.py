from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# List of all cities (used in one-hot encoding)
cities = ['Bangalore', 'Chennai', 'Hyderabad', 'Kolkata', 'Mumbai', 'Pune']

@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.json
    
    # Get input data from request
    bedrooms = int(data['bedrooms'])
    stories = int(data['stories'])
    bathrooms = int(data['bathrooms'])
    parking_lots = int(data['parking_lots'])
    city = data['city']

    # One-hot encode city
    city_features = [1 if c == city else 0 for c in cities[1:]]  # drop_first=True during training
    features = [bedrooms, stories, bathrooms, parking_lots] + city_features

    # Scale features
    scaled_features = scaler.transform([features])

    # Predict price
    predicted_price = model.predict(scaled_features)[0]

    return jsonify({
        'predicted_price': int(predicted_price),
        'city': city
    })

if __name__ == '__main__':
    app.run(debug=True)
