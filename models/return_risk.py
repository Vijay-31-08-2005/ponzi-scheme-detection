import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load dataset
df = pd.read_csv("models/return_risk_training_data.csv")

# Features and target
X = df[["promised_return_percent", "return_frequency_days", "time_to_roi_days", "minimum_deposit_usd"]]
y = df["return_risk_score"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output in range [0, 1]
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Optional early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print("Test MAE:", mae)

# Save model and scaler
model.save("tf_return_risk_model.keras")  # Recommended format
joblib.dump(scaler, "tf_return_risk_scaler.pkl")
print("Model and scaler saved.")
