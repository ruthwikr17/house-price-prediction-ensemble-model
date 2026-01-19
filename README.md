# House Price Prediction using Advanced ML Models 
House Price Prediction using Advanced XGBoost Techniques 


This project presents a robust machine learning system designed to predict residential property prices across Indian cities using advanced ensemble models including XGBoost, Random Forest, and CatBoost. The system is supported by a fully interactive Streamlit dashboard, offering users real-time prediction, filtering, and AI-assisted search capabilities.


# Live Demo <b>
https://house-price-ensemble-predict.streamlit.app/ <br>

---

## Problem Statement

Accurately estimating house prices is challenging due to diverse factors like location, area, number of rooms, property type, and city-specific trends. Buyers and real estate professionals require reliable tools to support informed decisions in this complex market.

---

## Proposed Solution

- A data-driven regression system using ensemble ML models.
- Real-time predictions with dynamic user input.
- AI-powered natural language search interface (via Gemini API).
- Support for city-specific price ranges and outlier filtering.

---

## Dataset

- **Source:** Curated Indian real estate dataset.
- **Key Features:**
  - Location, City, Property Type
  - Total Area (SQFT), Total Rooms, BHK, Balcony
  - Target: Price (in INR)

---

## Machine Learning Models

- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**
- **CatBoost Regressor**
- **Stacking Ensemble** (Final Model)

> Target variable is log-transformed to handle skewed distribution. City-wise price ranges are used to clamp outliers.

---

## Model Evaluation

| Model        | MSE       | R² Score |
|--------------|-----------|----------|
| LinearReg    | 0.3287    | 0.5878   |
| RandomForest | 0.1544    | 0.8064   |
| XGBoost      | 0.1525    | 0.8088   |
| CatBoost     | 0.1520    | 0.8093   |
| **Ensemble** | 0.1550    | 0.8057   |

---

## Streamlit Dashboard Features

- **Visualizations**: Price distributions, area vs price, filters by city, type, and BHK.
- **Price Prediction**: Based on user input.
- **AI-Powered Search**: Natural language queries using Gemini API (e.g., “3BHK flats in LB Nagar under 50L?”).
- **Filtered City Insights**: Neighborhood-level predictions.

---

## Tech Stack

**Backend / ML**
- Python (scikit-learn, XGBoost, CatBoost)
- pandas, numpy, joblib

**Frontend**
- Streamlit (Interactive UI)
- Plotly (Charts)
- Gemini Flash API (LLM)

**Others**
- Git, GitHub
- Git LFS (for large model files)
- Environment Variables for API keys (`.env`)

---

## Folder Structure

```bash
Project/
├── app/                  # Streamlit UI
├── src/                  # Model code
│   ├── train_models.py
│   ├── ensemble_predict.py
│   └── preprocess.py
├── models/               # Saved model artifacts (.pkl files)
├── data/                 # Dataset CSV
├── .env                  # (not tracked) Gemini API Key
├── .gitignore
├── requirements.txt
└── README.md
```
