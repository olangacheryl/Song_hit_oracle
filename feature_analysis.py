import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import shap
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Step 1: Load and Preprocess the Dataset

folder_path = '/Users/chery/OneDrive/Desktop/SPOTIFY_DATA'
all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
data = pd.concat((pd.read_csv(file) for file in all_files), ignore_index=True) 

# Features and target

data.drop(["track", "artist", "uri"], axis=1, inplace=True)
y = data['target']  # Popularity score (target)

# Normalize the features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(data)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)


# Step 2: Build and Train a Neural Network
model = Sequential([
    Dense(64, input_dim=data.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Linear activation for regression
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])


# Train the model
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Step 3: Analyze Feature Contribution using SHAP
# Initialize the SHAP Explainer
explainer = shap.Explainer(model, X_train)

# Compute SHAP values for test data
shap_values = explainer(X_test)

# Step 4: Visualize Feature Contributions
# Summary plot


shap.summary_plot(shap_values, X_test, feature_names = data.columns)
