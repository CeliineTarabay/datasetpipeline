import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Load and preprocess the dataset
df = pd.read_csv("./dataset2.csv")
df = df.sort_values(by=["SubjectID", "Time since start (s)"]).reset_index(drop=True)

# Function to calculate and display evaluation metrics
def evaluate_metrics(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean(axis=0)  # Per-variable MSE
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')  # Per-variable MAE
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')  # Per-variable R²
    
    # Print the metrics for each biological variable
    print("Evaluation Metrics Per Variable:")
    print(f"{'Variable':<40}{'MSE':<12}{'MAE':<12}{'R²':<12}")
    for i, var in enumerate(target_cols):
        print(f"{var:<40}{mse[i]:<12.4f}{mae[i]:<12.4f}{r2[i]:<12.4f}")
    
    # Aggregate metrics across all variables
    print("\nOverall Metrics:")
    print(f"Mean MSE: {mse.mean():.4f}")
    print(f"Mean MAE: {mae.mean():.4f}")
    print(f"Mean R²: {r2.mean():.4f}")

def interactive_training_loop(model, optimizer, criterion, X_train_t, Y_train_t, X_val_t, Y_val_t, 
                              epochs=20, batch_size=32, check_interval=5, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []  # To store training loss
    val_losses = []    # To store validation loss

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        batch_losses = []
        
        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            x_batch, y_batch = X_train_t[indices], Y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)  # Save training loss

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, Y_val_t).item()
        val_losses.append(val_loss)  # Save validation loss
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
        # User interaction
        if (epoch + 1) % check_interval == 0 and (epoch + 1) < epochs:
            response = input("Do you want to continue training? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Training interrupted by user. Saving model and exiting.")
                torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
                break
    
    print("Training completed. Best validation loss: {:.4f}".format(best_val_loss))
    return train_losses, val_losses  # Return loss history


# Split the dataset chronologically within each subject
train_list, val_list, test_list = [], [], []
groups = df.groupby("SubjectID")

for subject_id, group in groups:
    n = len(group)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    train_data = group.iloc[:train_end]
    val_data = group.iloc[train_end:val_end]
    test_data = group.iloc[val_end:]
    train_list.append(train_data)
    val_list.append(val_data)
    test_list.append(test_data)

train_df = pd.concat(train_list).reset_index(drop=True)
val_df = pd.concat(val_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

from sklearn.preprocessing import StandardScaler

# Suppose you want to scale all columns except 'SubjectID' and 'Time since start (s)' 
# and maybe the target columns if you want them in original scale.
# Let's define which columns are your features (environmental + maybe some bio, if you wish).

feature_cols = [
    "Temperature (°C)",
    "Relative Humidity (%)",
    "Absolute Humidity (g/m³)",
    "Barometric Pressure (mmHg)",
    "Dew Point (°C)",
    "Wind Chill (°C)",
    "Humidex (°C)",
    "Altitude (m)",
    "Speed (m/s)",
    "UV Index",
    "Illuminance (lx)",
    "Solar Irradiance (W/m²)",
    "Solar PAR (μmol/m²/s)",
    "Wind Direction (°)",
    "Magnetic Heading (°)",
    "True Heading (°)",
    "Temperature (°C)_Calibrata",
    "Relative Humidity (%)_Calibrata",
    "distance_to_start",
    "distance_to_end",
    "Distance2Path",
    "Wind Speed (km/hr) Run 1",
    "DayOfYear",
    "Year",
    "SecondsOfDay",
    "DayOfWeek",
    "DaysFromJuly15",
    # If there are any other relevant environmental variables in your dataset, you can add them here.
]

target_cols = [
    "Skin Conductance (microS)",
    "Phasic Skin Conductance (a.u.)",
    "Tonic Skin Conductance (a.u.)",
    "Skin Conductance Phasic Driver (a.u.)",
    "Heart Rate (bpm)",
    "Emotional Index (a.u.)",
    "Sympathovagal Balance (a.u.)",
]

# Filter the feature columns to include only those present in the DataFrame
available_feature_cols = [col for col in feature_cols if col in train_df.columns]

# Check if any target columns are missing
available_target_cols = [col for col in target_cols if col in train_df.columns]

if not available_feature_cols:
    raise ValueError("No feature columns are available in the dataset!")

if not available_target_cols:
    raise ValueError("No target columns are available in the dataset!")

# Normalize environmental and biological variables separately
env_scaler = StandardScaler()
bio_scaler = StandardScaler()

# Fit scalers on training data only
env_scaler.fit(train_df[available_feature_cols])
bio_scaler.fit(train_df[available_target_cols])

# Transform datasets
for df in [train_df, val_df, test_df]:
    df[available_feature_cols] = env_scaler.transform(df[available_feature_cols])
    df[available_target_cols] = bio_scaler.transform(df[available_target_cols])


# Create sequences for time-series forecasting
def make_sequences(df, feature_cols, target_cols, seq_length=10):
    feature_array = df[feature_cols].values
    target_array = df[target_cols].values
    X, Y = [], []
    for i in range(len(df) - seq_length):
        X.append(feature_array[i:i + seq_length])
        Y.append(target_array[i + seq_length])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def build_dataset(df, feature_cols, target_cols, seq_length=10):
    X_list, Y_list = [], []
    for _, group in df.groupby("SubjectID"):
        X, Y = make_sequences(group, feature_cols, target_cols, seq_length)
        X_list.append(X)
        Y_list.append(Y)
    return np.concatenate(X_list), np.concatenate(Y_list)

seq_length = 10
X_train, Y_train = build_dataset(train_df, available_feature_cols, available_target_cols, seq_length)
X_val, Y_val = build_dataset(val_df, available_feature_cols, available_target_cols, seq_length)
X_test, Y_test = build_dataset(test_df, available_feature_cols, available_target_cols, seq_length)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

print("Train:", X_train_t.shape, Y_train_t.shape)
print("Val:", X_val_t.shape, Y_val_t.shape)
print("Test:", X_test_t.shape, Y_test_t.shape)

# Define LSTM with Attention
class LSTMAttn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=10):
        super(LSTMAttn, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_scores = torch.tanh(self.attn(out))
        attn_weights = torch.softmax(self.v(attn_scores), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return self.fc(context)

# Initialize the model, loss function, and optimizer
input_size = len(available_feature_cols)  # Use the filtered feature columns
output_size = len(available_target_cols)  # Use the filtered target columns
hidden_size = 128
epochs = 40  # Total number of epochs you intend to train for
batch_size = 128

model = LSTMAttn(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

# Call the interactive training loop
train_losses, val_losses = interactive_training_loop(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    X_train_t=X_train_t,
    Y_train_t=Y_train_t,
    X_val_t=X_val_t,
    Y_val_t=Y_val_t,
    epochs=epochs,
    batch_size=batch_size,
    check_interval=5,  # Ask every 5 epochs
    patience=8         # Early stopping patience
)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss", color='blue')
plt.plot(val_losses, label="Validation Loss", color='orange')
plt.title("Training and Validation Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluate on test data
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t).cpu().numpy()  # Predicted values
    y_test_true = Y_test_t.cpu().numpy()  # True values

# Scale back biological variables to original scale
test_outputs_rescaled = bio_scaler.inverse_transform(test_outputs)
y_test_true_rescaled = bio_scaler.inverse_transform(y_test_true)

# Evaluate metrics
evaluate_metrics(y_test_true_rescaled, test_outputs_rescaled)

# Evaluate on test data
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t).cpu().numpy()  # Predicted values
    y_test_true = Y_test_t.cpu().numpy()  # True values

# Scale back biological variables to original scale
test_outputs_rescaled = bio_scaler.inverse_transform(test_outputs)
y_test_true_rescaled = bio_scaler.inverse_transform(y_test_true)

# Evaluate metrics
evaluate_metrics(y_test_true_rescaled, test_outputs_rescaled)

# Evaluate metrics
evaluate_metrics(y_test_true_rescaled, test_outputs_rescaled)

# Example metrics (replace with actual metrics from evaluate_metrics)
variables = target_cols
mse = ((y_test_true_rescaled - test_outputs_rescaled) ** 2).mean(axis=0)
mae = np.abs(y_test_true_rescaled - test_outputs_rescaled).mean(axis=0)
from sklearn.metrics import r2_score
r2 = r2_score(y_test_true_rescaled, test_outputs_rescaled, multioutput='raw_values')

# Plot MSE, MAE, and R²
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# MSE
axes[0].bar(variables, mse, color='skyblue')
axes[0].set_title("Mean Squared Error (MSE) per Variable")
axes[0].set_ylabel("MSE")

# MAE
axes[1].bar(variables, mae, color='orange')
axes[1].set_title("Mean Absolute Error (MAE) per Variable")
axes[1].set_ylabel("MAE")

# R²
axes[2].bar(variables, r2, color='green')
axes[2].set_title("R² Score per Variable")
axes[2].set_ylabel("R²")
axes[2].set_xlabel("Variables")

plt.tight_layout()
plt.show()

# Plot true vs. predicted values for a single variable
var_index = 0  # Index of the variable to plot
var_name = variables[var_index]

true_values = y_test_true_rescaled[:, var_index]
predicted_values = test_outputs_rescaled[:, var_index]

plt.figure(figsize=(12, 6))
plt.plot(true_values[:200], label="True", color='blue')  # Limit to 200 points for clarity
plt.plot(predicted_values[:200], label="Predicted", color='orange')
plt.title(f"Actual vs. Predicted for {var_name}")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()
plt.show()

