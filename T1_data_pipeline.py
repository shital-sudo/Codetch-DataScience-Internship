# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 08:47:49 2026

@author: hp
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Extract: read raw data
df = pd.read_csv("input_data.csv")
print("Raw data:")
print(df.head())

# 2. Transform: basic cleaning
# remove duplicates
df = df.drop_duplicates()

# handle missing values (if any)
df = df.dropna()

# separate features and target
X = df[["area", "bedrooms"]]
y = df["price"]

# feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# save cleaned & transformed data
clean_df = pd.DataFrame(X_scaled, columns=["area_scaled", "bedrooms_scaled"])
clean_df["price"] = y.values
clean_df.to_csv("cleaned_data.csv", index=False)
print("\nCleaned data saved to cleaned_data.csv")

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
score = model.score(X_test, y_test)
print("\nModel R2 score:", score)

# 6. Save pipeline objects
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved!")