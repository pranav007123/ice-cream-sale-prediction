import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Define the base directory and data path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data', 'ice_cream_sales.csv')

# Load dataset

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(data_path)

# Data Preprocessing Steps
df.drop_duplicates(inplace=True)  # Remove duplicate rows
df.dropna(inplace=True)  # Drop missing values

# Feature Scaling (if necessary)
scaler = StandardScaler()
df[['Temperature']] = scaler.fit_transform(df[['Temperature']])


# Separate the predictor (Temperature) and target (Sales)
X = df[['Temperature']]  # Predictor must be 2D
y = df['Sales']

# Train the simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the model for later use in Django views
model_path = os.path.join(BASE_DIR, 'ice_cream_model.pkl')
joblib.dump(model, model_path)
