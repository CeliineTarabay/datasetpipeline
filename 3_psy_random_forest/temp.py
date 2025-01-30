import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load datasets
aggregated_bio_path = "combined_dataset.csv"  # Ensure this path is correct
df_combined = pd.read_csv(aggregated_bio_path)

# Convert Phase to numerical (one-hot encoding)
df_combined = pd.get_dummies(df_combined, columns=["Phase"], drop_first=True)

# Split features (X) and labels (y)
features = df_combined.drop(columns=["Arousal", "Restoration", "Valence"])  # Biological signals
labels = df_combined[["Arousal", "Restoration", "Valence"]]  # Psychological responses

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# Save processed datasets for model training
np.savez("train_test_split.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("Training and test sets successfully saved.")
