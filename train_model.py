import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Read the data
df = pd.read_csv('synthetic_bike_data.csv')

# Prepare features
X = df[['brand', 'kms_driven', 'power', 'age', 'city']]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_cols = ['brand', 'city']
encoder.fit(X_train[categorical_cols])

# Transform the categorical columns
X_train_encoded = encoder.transform(X_train[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])

# Get numerical columns
numerical_cols = ['kms_driven', 'power', 'age']
X_train_numerical = X_train[numerical_cols].values
X_test_numerical = X_test[numerical_cols].values

# Combine encoded categorical and numerical features
X_train_combined = np.hstack([X_train_encoded, X_train_numerical])
X_test_combined = np.hstack([X_test_encoded, X_test_numerical])

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_combined, y_train)

# Save the model and encoder
with open('bike_predictor_rf.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Save the search text for the app
search_txt = {
    'brand': sorted(df['brand'].unique().tolist()),
    'city': sorted(df['city'].unique().tolist())
}

with open('search.pkl', 'wb') as f:
    pickle.dump(search_txt, f)

print("Model training completed and saved!") 