import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load dataset
df = pd.read_csv('datasets/allcitydata.csv')

# Features and target
X = df[['bedrooms', 'stories', 'bathrooms', 'parking_lots', 'city', 'area_sqft']]
y = df['price']

# One-hot encode city
X = pd.get_dummies(X, columns=['city'], drop_first=True)

# Save city names for consistent encoding later (optional but useful)
with open('city_columns.txt', 'w') as f:
    for col in X.columns:
        f.write(col + '\n')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model and scaler
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully.")
