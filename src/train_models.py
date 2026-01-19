import os, sys
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform, loguniform
from preprocess import preprocess_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def tune_model(model, param_dist, X_train, y_train, cv=5, n_iter=50):
    print(f"\n🔧 Tuning {model.__class__.__name__}...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    print(f"✅ Best params: {search.best_params_}")
    print(f"🏆 Best score: {-search.best_score_:.4f}")
    return search.best_estimator_


def train_and_save_models(data_path):
    print("\n📊 Loading and preprocessing data...")
    X, y, (target_encoder, scaler, feature_order) = preprocess_data(data_path)
    y_log = np.log1p(y)

    df = pd.read_csv(data_path)
    city_stats = df.groupby("city")["Price"].agg(["median"])
    price_ranges = {
        city: (
            max(10_00_000, row["median"] * 0.5),
            min(10_00_00_000, row["median"] * 2),
        )
        for city, row in city_stats.iterrows()
    }

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    tune_configs = {
        "RandomForest": {
            "n_estimators": randint(200, 1000),  # Number of Trees
            "max_depth": randint(5, 30),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", 0.8],
            "bootstrap": [True, False],
        },
        "XGBoost": {
            "n_estimators": randint(200, 1000),
            "max_depth": randint(3, 15),
            "learning_rate": loguniform(1e-3, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "gamma": uniform(0, 0.5),
            "reg_alpha": loguniform(1e-5, 1),
            "reg_lambda": loguniform(1e-5, 1),
        },
        "CatBoost": {
            "iterations": randint(500, 2000),
            "depth": randint(4, 12),
            "learning_rate": loguniform(0.005, 0.3),
            "l2_leaf_reg": loguniform(1, 10),
            "border_count": randint(32, 255),
            "bagging_temperature": uniform(0, 1),
        },
    }

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror", random_state=42, n_jobs=-1
        ),
        "CatBoost": CatBoostRegressor(
            verbose=0, random_state=42, thread_count=-1, od_type="Iter", od_wait=50
        ),
    }

    print("\n🚀 Training and tuning models...")
    trained_models = {}
    for name, model in models.items():
        if name in tune_configs:
            model = tune_model(model, tune_configs[name], X_train, y_train)
        else:
            if name == "XGBoost":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=50,
                    verbose=False,
                )
            else:
                model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\n📈 {name} Performance:")
        print(f"MSE: {mse:.4f} | R2: {r2:.4f}")

    print("\n🧩 Building optimized ensemble...")
    ensemble = StackingRegressor(
        estimators=[
            ("rf", trained_models["RandomForest"]),
            ("xgb", trained_models["XGBoost"]),
            ("catboost", trained_models["CatBoost"]),
        ],
        final_estimator=GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42
        ),
        cv=5,
        n_jobs=-1,
        passthrough=True,
    )
    ensemble.fit(X_train, y_train)
    trained_models["Ensemble"] = ensemble

    print("\n📊 Final Model Performance:")
    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        }
        print(
            f"{name:<15} MSE: {results[name]['MSE']:.4f} | R2: {results[name]['R2']:.4f}"
        )

    print("\n💾 Saving artifacts...")
    artifacts = {
        "models": trained_models,
        "preprocessors": {
            "target_encoder": target_encoder,
            "scaler": scaler,
            "feature_order": feature_order,
        },
        "metadata": {
            "price_ranges": price_ranges,
        },
        "performance": results,
    }

    os.makedirs("../models", exist_ok=True)
    joblib.dump(artifacts, "../models/trained_artifacts.pkl")
    print("All artifacts saved successfully!")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(
        BASE_DIR, "..", "data", "Indian_Real_Estate_Clean_Data.csv"
    )
    train_and_save_models(DATA_PATH)
