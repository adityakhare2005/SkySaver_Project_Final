import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load preprocessed data
df = pd.read_csv("D:/4th sem/Elective Project/Project_11/data/preprocessed_flight_data.csv")

# Define features and target variable
X = df.drop(columns=["price"])
y = df["price"]

# Ensure no categorical columns remain
print("Features before training:", X.select_dtypes(include=['object']).columns.tolist())  # Should be empty

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model correctly for Streamlit app
joblib.dump((model, X_train.columns), "D:/4th sem/Elective Project/Project_11/models/price_predictor.pkl")

print("✅ Model training complete using Linear Regression!")