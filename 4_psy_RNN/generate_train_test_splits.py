import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# === STEP 1: LOAD DATA ===
dataset2_path = "../Dataset2.csv"  # Biological signals
psychological_data_path = "../psychological_scores.csv"  # Psychological labels

df_bio_full = pd.read_csv(dataset2_path)
df_psych = pd.read_csv(psychological_data_path)

# Reconstruct "Phase" from one-hot encoding
phase_columns = ["Baseline", "Cafe", "Square", "Crosswalk", "U3", "Trivulziana", "U4", "Tram"]
df_bio_full["Phase"] = df_bio_full[phase_columns].idxmax(axis=1)
df_bio_full = df_bio_full.drop(columns=phase_columns)

# Merge with psychological scores
df_merged = df_bio_full.merge(df_psych, on=["SubjectID", "Phase"], how="inner")

# === STEP 2: SPLIT SUBJECTS INTO TRAIN/TEST & HOLDOUT ===
unique_subjects = df_merged["SubjectID"].unique()
holdout_subjects = np.random.choice(unique_subjects, size=5, replace=False)  # Keep 5 subjects as holdout

df_holdout = df_merged[df_merged["SubjectID"].isin(holdout_subjects)]
df_train_test = df_merged[~df_merged["SubjectID"].isin(holdout_subjects)]

print(f"Holdout subjects: {holdout_subjects}")

# === STEP 3: PROCESS TRAIN/TEST DATA ===
sequence_length = 30  # Sliding window (3.75 sec at 8Hz)
bio_columns = [
    "Skin Conductance (microS)", "Phasic Skin Conductance (a.u.)",
    "Tonic Skin Conductance (a.u.)", "Skin Conductance Phasic Driver (a.u.)",
    "Heart Rate (bpm)", "Emotional Index (a.u.)", "Sympathovagal Balance (a.u.)"
]
label_columns = ["Arousal", "Restoration", "Valence"]

X, y = [], []
for _, phase_data in df_train_test.groupby(["SubjectID", "Phase"]):
    phase_bio = phase_data[bio_columns].values
    phase_label = phase_data[label_columns].iloc[0].values

    if len(phase_bio) >= sequence_length:
        for i in range(len(phase_bio) - sequence_length + 1):
            X.append(phase_bio[i : i + sequence_length])
            y.append(phase_label)

X, y = np.array(X), np.array(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 4: PROCESS HOLDOUT DATA ===
X_holdout, y_holdout = [], []
for _, phase_data in df_holdout.groupby(["SubjectID", "Phase"]):
    phase_bio = phase_data[bio_columns].values
    phase_label = phase_data[label_columns].iloc[0].values

    if len(phase_bio) >= sequence_length:
        X_holdout.append(phase_bio[:sequence_length])  # Use only the first `sequence_length`
        y_holdout.append(phase_label)

X_holdout, y_holdout = np.array(X_holdout), np.array(y_holdout)

df_holdout_metadata = df_holdout[["SubjectID", "Phase"]].drop_duplicates()
df_holdout_metadata.to_csv("holdout_metadata.csv", index=False)

# === STEP 5: SAVE SPLITS ===
np.savez("lstm_train_test_limited.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
np.savez("lstm_holdout_limited.npz", X_holdout=X_holdout, y_holdout=y_holdout)

print(f"Train/Test dataset saved -> X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Holdout dataset saved -> X_holdout: {X_holdout.shape}, y_holdout: {y_holdout.shape}")
