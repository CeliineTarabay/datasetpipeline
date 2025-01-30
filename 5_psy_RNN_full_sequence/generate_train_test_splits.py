import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === STEP 1: LOAD DATA ===
dataset2_path = "../Dataset2.csv"  # Biological signals
psychological_data_path = "../psychological_scores.csv"  # Psychological labels

df_bio_full = pd.read_csv(dataset2_path)
df_psych = pd.read_csv(psychological_data_path)

# === STEP 2: RECONSTRUCT "PHASE" FROM ONE-HOT ENCODING ===
phase_columns = ["Baseline", "Cafe", "Square", "Crosswalk", "U3", "Trivulziana", "U4", "Tram"]

# Convert one-hot encoding back into a single "Phase" column
df_bio_full["Phase"] = df_bio_full[phase_columns].idxmax(axis=1)

# Drop one-hot encoded columns (optional)
df_bio_full = df_bio_full.drop(columns=phase_columns)

# === STEP 3: REMOVE NaNs FROM PSYCHOLOGICAL DATA BEFORE MERGING ===
df_psych_clean = df_psych.dropna(subset=["Arousal", "Restoration", "Valence"])  # Drop before merging

print("🔹 NaN count in df_psych before merging:")
print(df_psych_clean.isna().sum())

# Merge cleaned psychological scores with biological data
df_merged = df_bio_full.merge(df_psych_clean, on=["SubjectID", "Phase"], how="inner")

# === STEP 4: CHECK & REMOVE NaNs AFTER MERGING ===
print("🔹 NaN count in df_merged after merging:")
print(df_merged.isna().sum())

# Drop remaining NaN values if any
df_merged_clean = df_merged.dropna(subset=["Arousal", "Restoration", "Valence"])

print("✅ After final NaN removal, dataset size:", df_merged_clean.shape)

# === STEP 5: PROCESS TRAIN/TEST DATA ===
bio_columns = [
    "Skin Conductance (microS)", "Phasic Skin Conductance (a.u.)",
    "Tonic Skin Conductance (a.u.)", "Skin Conductance Phasic Driver (a.u.)",
    "Heart Rate (bpm)", "Emotional Index (a.u.)", "Sympathovagal Balance (a.u.)"
]
label_columns = ["Arousal", "Restoration", "Valence"]

X, y = [], []
for _, phase_data in df_merged_clean.groupby(["SubjectID", "Phase"]):
    phase_bio = phase_data[bio_columns].values
    phase_label = phase_data[label_columns].iloc[0].values  # One label per phase

    X.append(phase_bio)
    y.append(phase_label)

# === STEP 6: TRIM LONG SEQUENCES TO MAX_SEQUENCE_LENGTH ===
MAX_SEQUENCE_LENGTH = 6000  # Trim sequences longer than this

X_trimmed, y_trimmed = [], []
for i in range(len(X)):  
    if len(X[i]) > MAX_SEQUENCE_LENGTH:
        X_trimmed.append(X[i][:MAX_SEQUENCE_LENGTH])  # Keep only first 6000 steps
    else:
        X_trimmed.append(X[i])
    y_trimmed.append(y[i])  

# Convert to NumPy arrays with padding
X_trimmed_padded = pad_sequences(X_trimmed, maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')
X_trimmed_padded = np.array(X_trimmed_padded)  # Ensure X is NumPy

# Convert y to array
y_trimmed = np.array(y_trimmed)

print(f"✅ After trimming & padding: X shape: {X_trimmed_padded.shape}, y shape: {y_trimmed.shape}")
print(f"🔹 Max sequence length after trimming: {max(len(seq) for seq in X_trimmed)}")

# === STEP 7: REMOVE ANY REMAINING NaNs BEFORE SPLITTING ===
print("🔹 NaNs in y before splitting:", np.isnan(y_trimmed).sum())

# Remove NaN rows
nan_mask = ~np.isnan(y_trimmed).any(axis=1)
X_trimmed_padded, y_trimmed = X_trimmed_padded[nan_mask], y_trimmed[nan_mask]  # ✅ Fixed indexing

# === STEP 8: CREATE A HOLDOUT SUBJECT SET (E.G., 10% OF SUBJECTS) ===
holdout_fraction = 0.1  # 10% of unique subjects
unique_subjects = df_merged_clean["SubjectID"].unique()
holdout_subjects = np.random.choice(unique_subjects, size=int(len(unique_subjects) * holdout_fraction), replace=False)

# Create masks for train/test vs holdout
holdout_mask = df_merged_clean["SubjectID"].isin(holdout_subjects)
train_test_mask = ~holdout_mask

# Apply masks
df_train_test = df_merged_clean[train_test_mask]
df_holdout = df_merged_clean[holdout_mask]

print(f"✅ Holdout subjects: {len(holdout_subjects)} out of {len(unique_subjects)} total subjects")

# === STEP 9: PREPARE HOLDOUT DATA ===
X_holdout, y_holdout = [], []
for _, phase_data in df_holdout.groupby(["SubjectID", "Phase"]):
    phase_bio = phase_data[bio_columns].values
    phase_label = phase_data[label_columns].iloc[0].values  # One label per phase

    if len(phase_bio) > MAX_SEQUENCE_LENGTH:
        X_holdout.append(phase_bio[:MAX_SEQUENCE_LENGTH])  # Trim
    else:
        X_holdout.append(phase_bio)  # Keep as is
    y_holdout.append(phase_label)

# Convert holdout data to NumPy arrays
X_holdout_padded = pad_sequences(X_holdout, maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')
y_holdout = np.array(y_holdout)

# === STEP 10: REMOVE NaNs FROM HOLDOUT DATA ===
nan_mask = ~np.isnan(y_holdout).any(axis=1)
X_holdout_padded, y_holdout = X_holdout_padded[nan_mask], y_holdout[nan_mask]

print(f"✅ Holdout dataset shape: X={X_holdout_padded.shape}, y={y_holdout.shape}")

df_holdout_metadata = df_holdout[["SubjectID", "Phase"]].drop_duplicates()
df_holdout_metadata.to_csv("holdout_metadata.csv", index=False)

# === STEP 11: CONTINUE TRAIN/TEST SPLITTING ===
X_train, X_test, y_train, y_test = train_test_split(X_trimmed_padded, y_trimmed, test_size=0.2, random_state=42)

# === STEP 12: SAVE ALL DATASETS ===
np.savez("lstm_train_test_full_phase.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
np.savez("lstm_holdout.npz", X_holdout=X_holdout_padded, y_holdout=y_holdout)

print(f"✅ Final datasets saved with holdout subjects excluded from training/testing. (Max {MAX_SEQUENCE_LENGTH} time steps).")
