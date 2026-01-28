import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import joblib
import os


def preprocess_data(file_path):
    df = pd.read_csv(
        file_path,
        engine="python",
        sep=",",
        quoting=3,  # Ignore quotation parsing
        on_bad_lines="skip",
    )

    # Drop rows missing essential columns
    df = df.dropna(
        subset=["Price", "Location", "Total_Area(SQFT)", "BHK", "Total_Rooms"]
    )

    # --- DATA QUALITY FILTERS ---
    df = df[df["Price"] <= 10_00_00_000]  # Cap at ₹10 Cr
    df = df[
        (df["Total_Area(SQFT)"] >= 300) & (df["Total_Area(SQFT)"] <= 10000)
    ]  # Area bounds
    df = df[(df["BHK"] >= 1) & (df["BHK"] <= 6)]  # Valid BHK range
    df = df[df["Total_Rooms"] >= df["BHK"]]  # Logical room-BHK consistency

    # Fill missing categorical values (robustness)
    for col in ["Balcony", "city", "property_type"]:
        df[col] = df[col].fillna("Unknown")

    # Define input features and target
    X = df[
        [
            "Location",
            "Total_Area(SQFT)",
            "Total_Rooms",
            "Balcony",
            "city",
            "property_type",
            "BHK",
        ]
    ]
    y = df["Price"]

    # Split into feature types
    categorical_features = ["Location", "Balcony", "city", "property_type"]
    numeric_features = ["Total_Area(SQFT)", "Total_Rooms", "BHK"]

    # Impute numeric values
    numeric_imputer = SimpleImputer(strategy="median")
    X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

    # Target encoding
    target_encoder = ce.TargetEncoder(cols=categorical_features)
    X[categorical_features] = target_encoder.fit_transform(X[categorical_features], y)

    # Maintain consistent feature order
    feature_order = numeric_features + categorical_features
    X = X[feature_order]

    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save preprocessor artifacts
    os.makedirs("../models", exist_ok=True)
    joblib.dump(target_encoder, "../models/target_encoder.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")

    return X_scaled, df["Price"], (target_encoder, scaler, feature_order)


if __name__ == "__main__":
    file_path = "../data/Indian_Real_Estate_Clean_Data.csv"
    X_processed, y, _ = preprocess_data(file_path)
    print("Preprocessing complete with filters and saved artifacts.")
