import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("data/Indian_Real_Estate_Clean_Data.csv")


sample_df = df.head(10)

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis("off")
table = ax.table(
    cellText=sample_df.values,
    colLabels=sample_df.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title("Dataset Sample", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Normalize text case for consistent matching
df.columns = df.columns.str.strip().str.lower()

# Drop rows with missing values
df.dropna(inplace=True)

# Apply log transformation to target variable (Price)
df["price"] = np.log1p(df["price"])

# Convert 'Location' to categorical encoding
df["location"] = df["location"].astype("category").cat.codes

# Define features and target
X = df[["area", "no._of_bedrooms", "location"]]  # Select features
y = df["price"]  # Target variable

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict and evaluate Linear Regression
y_pred_lr = lr_model.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Print the metrics for Linear Regression
print(f"Linear Regression MSE: {mse_lr:.2f}")
print(f"Linear Regression R² Score: {r2_lr:.4f}")

# Try Random Forest to see if ensemble methods improve performance
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the metrics for Random Forest
print(f"Random Forest MSE: {mse_rf:.2f}")
print(f"Random Forest R² Score: {r2_rf:.4f}")

# Visualize the actual vs predicted prices for Random Forest
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Prices (in Lakhs)")
plt.ylabel("Predicted Prices (in Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# Hyperparameter tuning for RandomForest using GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(), param_grid=param_grid, cv=5, n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters found by GridSearchCV
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Train Random Forest with the best parameters
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate with the best Random Forest model
y_pred_best_rf = best_rf_model.predict(X_test_scaled)
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)

# Print the improved metrics
print(f"Best Random Forest MSE: {mse_best_rf:.2f}")
print(f"Best Random Forest R² Score: {r2_best_rf:.4f}")


# More Visualizations
# 1. Distribution of House Prices
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True, bins=30)
plt.title("Distribution of House Prices")
plt.xlabel("Price (in Lakhs)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 3. Actual vs Predicted Prices
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred_lr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Prices (in Lakhs)")
plt.ylabel("Predicted Prices (in Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Residuals Plot
residuals = y_test - y_pred_lr
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
