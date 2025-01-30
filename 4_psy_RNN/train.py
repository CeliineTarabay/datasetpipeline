from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


# Load dataset
data = np.load("lstm_train_test_split.npz")
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# Initialize writer
writer = SummaryWriter("logs")

# Check for NaN or Inf values
print("NaN in X_train:", np.isnan(X_train).sum())
print("NaN in y_train:", np.isnan(y_train).sum())
print("Inf in X_train:", np.isinf(X_train).sum())
print("Inf in y_train:", np.isinf(y_train).sum())

# Identify which rows have NaNs
nan_rows = np.isnan(y_train).any(axis=1)
print(f"Found {nan_rows.sum()} NaN rows in y_train!")

# Print some problematic rows
print("First few NaN rows in y_train:", y_train[nan_rows][:5])

# === STEP 2: BUILD LSTM MODEL ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(3)  # 3 output neurons (Arousal, Restoration, Valence)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === STEP 3: TRAIN THE MODEL ===
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# === STEP 4: EVALUATE & SAVE THE MODEL ===
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")

model.save("lstm_psych_model.h5")
print("LSTM Model saved as lstm_psych_model.h5")
