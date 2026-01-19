import os
import joblib
import numpy as np
import pandas as pd


class PricePredictor:
    def __init__(self):
        # Path to saved model artifacts
        artifacts_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "trained_artifacts.pkl"
        )

        # === Auto-train if artifacts not found (for Streamlit Cloud) ===
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(
                "Model artifacts not found. Please run train_models.py locally before deployment."
            )

        # === Load artifacts ===
        self.artifacts = joblib.load(artifacts_path)

        self.model = self.artifacts["models"]["Ensemble"]
        self.target_encoder = self.artifacts["preprocessors"]["target_encoder"]
        self.scaler = self.artifacts["preprocessors"]["scaler"]
        self.feature_order = self.artifacts["preprocessors"]["feature_order"]

        # Price ranges may or may not exist — safe getter
        self.price_ranges = self.artifacts.get("metadata", {}).get("price_ranges", {})

        # Default fallback range
        self.default_range = (15_00_000, 1_25_00_000)

    def _validate_input(self, input_data):
        """Validate input structure and basic sanity checks"""
        required = {
            "Location": str,
            "Total_Area(SQFT)": (int, float),
            "Total_Rooms": (int, float),
            "Balcony": str,
            "city": str,
            "property_type": str,
            "BHK": int,
        }

        for field, types in required.items():
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(input_data[field], types):
                raise ValueError(f"Invalid type for {field}. Expected {types}")

        # Basic value validation
        if not (300 <= input_data["Total_Area(SQFT)"] <= 10000):
            raise ValueError("Area must be between 300 and 10000 sqft")
        if not (1 <= input_data["BHK"] <= 6):
            raise ValueError("BHK must be between 1 and 6")

    def predict(self, input_data):
        """Make a price prediction with automatic range handling"""
        try:
            self._validate_input(input_data)
            city = input_data["city"]

            # Construct DataFrame with feature order
            df = pd.DataFrame(
                {col: [input_data.get(col)] for col in self.feature_order}
            )

            # Target encode categorical features
            cat_cols = ["Location", "Balcony", "city", "property_type"]
            df[cat_cols] = self.target_encoder.transform(df[cat_cols])

            # Scale all features
            X = self.scaler.transform(df)

            # Predict log price and inverse-transform
            log_pred = self.model.predict(X)[0]
            price = np.expm1(log_pred)

            # Clamp price within city-specific or default range
            min_price, max_price = self.price_ranges.get(city, self.default_range)
            final_price = np.clip(price, min_price, max_price)

            return {
                "predicted_price": round(final_price, 2),
                "city": city,
                "price_range": f"₹{min_price:,.2f} - ₹{max_price:,.2f}",
                "success": True,
            }

        except Exception as e:
            return {"error": str(e), "success": False}


if __name__ == "__main__":
    # Example usage
    predictor = PricePredictor()

    test_property = {
        "Location": "Uppal",
        "Total_Area(SQFT)": 2000,
        "Total_Rooms": 4,
        "Balcony": "No",
        "city": "Hyderabad",
        "property_type": "Flat",
        "BHK": 2,
    }

    result = predictor.predict(test_property)
    if result["success"]:
        print(f"Predicted Price: ₹{result['predicted_price']:,.2f}")
        print(f"Expected Range for {result['city']}: {result['price_range']}")
    else:
        print(f"Error: {result['error']}")
