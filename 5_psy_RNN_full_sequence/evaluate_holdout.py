import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load holdout dataset
data = np.load("lstm_holdout.npz", allow_pickle=True)
X_holdout, y_holdout = data["X_holdout"], data["y_holdout"]

# Load SubjectIDs and Phases (ensure they were saved separately in dataset creation)
df_holdout_metadata = pd.read_csv("holdout_metadata.csv")  # SubjectID, Phase mapping

# Load trained model
model = load_model("lstm_psych_full_phase.h5", compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Evaluate model
loss, mae = model.evaluate(X_holdout, y_holdout)
print(f"Holdout Evaluation -> Loss: {loss:.4f}, MAE: {mae:.4f}")

# Generate predictions
y_pred = model.predict(X_holdout)

# Convert to DataFrame
df_results = df_holdout_metadata.copy()
df_results["True_Arousal"] = y_holdout[:, 0]
df_results["True_Restoration"] = y_holdout[:, 1]
df_results["True_Valence"] = y_holdout[:, 2]

df_results["Pred_Arousal"] = y_pred[:, 0]
df_results["Pred_Restoration"] = y_pred[:, 1]
df_results["Pred_Valence"] = y_pred[:, 2]

# Calculate Absolute Errors
df_results["Error_Arousal"] = abs(df_results["True_Arousal"] - df_results["Pred_Arousal"])
df_results["Error_Restoration"] = abs(df_results["True_Restoration"] - df_results["Pred_Restoration"])
df_results["Error_Valence"] = abs(df_results["True_Valence"] - df_results["Pred_Valence"])

# Save results for further analysis
df_results.to_csv("holdout_results.csv", index=False)
print("✅ Holdout results saved as holdout_results.csv")
