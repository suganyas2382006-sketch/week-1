import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Load CSV and print first 10 rows
# ==========================================
# Load the dataset (replace 'house_prices.csv' with your file path)
df = pd.read_csv("house_prices.csv")

print("--- First 10 Rows ---")
print(df.head(10))

# ==========================================
# 2. Split dataset into train/test
# ==========================================
# Assuming 'area' is the feature and 'price' is the target variable
X = df[["area"]]
y = df["price"]

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(
    f"\nDataset split into {len(X_train)} training rows and {len(X_test)} testing rows."
)

# ==========================================
# 3. Train Linear Regression model & evaluate
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

# Check model performance on the test set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nModel R² Score (Accuracy): {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")


# ==========================================
# 4. Predict house price based on area input
# ==========================================
def predict_price(area_input):
    input_data = pd.DataFrame([[area_input]], columns=["area"])
    return model.predict(input_data)[0]


# Example Prediction
sample_area = 1500  # Square feet/meters
predicted_price = predict_price(sample_area)

print(f"\nPredicted price for area = {sample_area}: ${predicted_price:,.2f}")
