import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import EarlyStopping

# === STEP 1: LOAD PREPROCESSED FULL-PHASE DATA ===
data = np.load("lstm_train_test_full_phase.npz")
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# === STEP 2: DEFINE LSTM MODEL WITH MASKING ===
model = Sequential([
    Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])),  # Ignore padding
    LSTM(128, return_sequences=True),  # First LSTM layer
    LSTM(64, return_sequences=False),  # Second LSTM layer
    Dense(32, activation='relu'),
    Dense(3)  # Output: Arousal, Restoration, Valence
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === STEP 3: TRAIN MODEL WITH EARLY STOPPING ===
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30, batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# === STEP 4: SAVE MODEL ===
model.save("lstm_psych_full_phase.h5")
print("LSTM Model trained and saved as lstm_psych_full_phase.h5")
