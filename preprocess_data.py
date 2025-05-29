import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load dataset
df = pd.read_csv("D:/4th sem/Elective Project/Project_11/data/flight_data.csv")

# Rename columns
df.rename(columns={
    'source_city': 'source',
    'destination_city': 'destination',
    'departure_time': 'departure_time',
    'arrival_time': 'arrival_time',
    'class': 'class',
    'days_left': 'days_till_travel'
}, inplace=True)

# Save original categories for UI reference (includes `stops`)
df[['source', 'destination', 'departure_time', 'arrival_time', 'class', 'airline', 'stops']].to_csv(
    "D:/4th sem/Elective Project/Project_11/data/original_categories.csv", index=False)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)
df.dropna(inplace=True)

# Identify categorical columns
categorical_cols = ['source', 'destination', 'departure_time', 'arrival_time', 'class', 'stops', 'airline']
label_cols = ['flight']  # Label Encoding for flight numbers

# Apply One-Hot Encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

# Apply Label Encoding for flight numbers
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save encoders for inference
joblib.dump(encoder, "D:/4th sem/Elective Project/Project_11/models/encoder.pkl")
joblib.dump(label_encoders, "D:/4th sem/Elective Project/Project_11/models/label_encoders.pkl")

# Drop original categorical columns and merge with encoded features
df = df.drop(columns=categorical_cols).reset_index(drop=True)
df = pd.concat([df, encoded_df], axis=1)

# Save preprocessed data
df.to_csv("D:/4th sem/Elective Project/Project_11/data/preprocessed_flight_data.csv", index=False)

print("✅ Data Preprocessing Complete!")