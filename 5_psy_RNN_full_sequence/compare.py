import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load train dataset
data = np.load("lstm_train_test_full_phase.npz", allow_pickle=True)
X_train, y_train = data["X_train"], data["y_train"]

# Convert labels into a DataFrame for easier analysis
df_labels = pd.DataFrame(y_train, columns=["Arousal", "Restoration", "Valence"])

# Plot histograms for each label
plt.figure(figsize=(12, 4))
for i, label in enumerate(df_labels.columns, 1):
    plt.subplot(1, 3, i)
    plt.hist(df_labels[label], bins=20, color="skyblue", edgecolor="black")
    plt.title(f"{label} Distribution")
    plt.xlabel(label)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Load predictions and holdout set
predictions = np.load("holdout_predictions.npy")
data = np.load("lstm_holdout.npz", allow_pickle=True)
X_holdout, y_holdout = data["X_holdout"], data["y_holdout"]

# Convert to DataFrame
df_results = pd.DataFrame({
    "True_Arousal": y_holdout[:, 0],
    "Pred_Arousal": predictions[:, 0],
    "True_Restoration": y_holdout[:, 1],
    "Pred_Restoration": predictions[:, 1],
    "True_Valence": y_holdout[:, 2],
    "Pred_Valence": predictions[:, 2],
})

# Scatter plots to compare True vs. Predicted
plt.figure(figsize=(15, 5))
for i, (true_col, pred_col) in enumerate([("True_Arousal", "Pred_Arousal"),
                                           ("True_Restoration", "Pred_Restoration"),
                                           ("True_Valence", "Pred_Valence")], 1):
    plt.subplot(1, 3, i)
    plt.scatter(df_results[true_col], df_results[pred_col], alpha=0.5, color="blue")
    plt.plot([-4, 4], [-4, 4], color="red", linestyle="--")  # Perfect prediction line
    plt.title(f"{true_col} vs {pred_col}")
    plt.xlabel(true_col)
    plt.ylabel(pred_col)
    plt.grid(True)
plt.tight_layout()
plt.show()
