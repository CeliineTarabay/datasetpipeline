import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# Load and preprocess the dataset
df = pd.read_csv("./dataset2.csv")
df = df.sort_values(by=["SubjectID", "Time since start (s)"]).reset_index(drop=True)

# Initialize writer
writer = SummaryWriter("logs")

input_groups = {
    "environmental": [
        "Temperature (°C)", "Relative Humidity (%)", "Absolute Humidity (g/m³)", 
        "Barometric Pressure (mmHg)", "Dew Point (°C)", "Wind Chill (°C)", 
        "Humidex (°C)", "UV Index", "Illuminance (lx)", 
        "Solar Irradiance (W/m²)", "Solar PAR (μmol/m²/s)"
    ],
    "positional": [
        "Latitude (°)", "Longitude (°)", "Altitude (m)", "Speed (m/s)", 
        "Distance2Path", "distance_to_start", "distance_to_end", 
        "azimuth", "altitude"
    ],
    "temporal": [
        "Time (s) Run 1", "DayOfYear", "Year", "SecondsOfDay", 
        "DayOfWeek", "DaysFromJuly15"
    ],
    "derived": [
        "Temperature (°C)_Calibrata", "Relative Humidity (%)_Calibrata", 
        "Wind Speed (km/hr) Run 1", "Wind Speed Run 1"
    ]
}

def make_group_sequences(df, input_groups, target_cols, seq_length=10):
    group_arrays = {group: [] for group in input_groups}
    target_array = []
    
    for i in range(len(df) - seq_length):
        for group, features in input_groups.items():
            available_features = [f for f in features if f in df.columns]
            group_arrays[group].append(df.iloc[i:i + seq_length][available_features].values)
        target_array.append(df.iloc[i + seq_length][target_cols].values)
    
    group_tensors = {group: torch.tensor(np.array(group_arrays[group]), dtype=torch.float32)
                     for group in group_arrays}
    target_tensor = torch.tensor(np.array(target_array), dtype=torch.float32)
    return group_tensors, target_tensor


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
    grace_threshold = 0.1
    grace_patience = 0
    patience_counter = 0
    patience = 10
    train_losses = []  # To store training loss
    val_losses = []    # To store validation loss

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        batch_losses = []
        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            x_batch_groups = {group: x_train_groups[group][indices] for group in x_train_groups}
            y_batch = Y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(x_batch_groups)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # === Gradient Clipping ===
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust `max_norm` as needed
            
            # === Gradient Monitoring ===
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    writer.add_scalar(f"Gradients/{name}_norm", param_norm, epoch)
            total_norm = total_norm ** 0.5
            writer.add_scalar("Gradients/Total_Norm", total_norm, epoch)
            # === End Monitoring ===
            
            optimizer.step()
           
        writer.close()
        
        # Initialize permutation inside the epoch loop
        permutation = torch.randperm(X_train_t.size(0))
        batch_losses = []
        
        # Determine which variables to include based on the epoch
        if epoch == 10:  # Introduce medium variables
            patience += 5  # Extend patience
        if epoch == 20:  # Introduce hard variables
            patience += 5  # Extend patience
            
        if epoch < 10:
            active_variables = easy_variables
        elif epoch < 20:
            active_variables = easy_variables + medium_variables
        else:
            active_variables = easy_variables + medium_variables + hard_variables

        active_indices = [variable_indices[var] for var in active_variables]
        
        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i + batch_size]  # Create batch indices
            
            # Extract grouped inputs
            x_batch_groups = {group: x_train_groups[group][indices] for group in x_train_groups}
            y_batch = Y_train_t[indices]
                        
            optimizer.zero_grad()
            outputs = model(x_batch_groups)
            
            # Compute loss
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # Validation step
        model.eval()
        with torch.no_grad():
            x_val_groups = {group: X_val_t[:, :, indices_list] 
                            for group, indices_list in feature_indices.items()}
            val_outputs = model(x_val_groups)[:, active_indices]
            val_loss = sum(criterion(val_outputs[:, j:j+1], Y_val_t[:, active_indices][:, j:j+1]) 
                           for j in range(len(active_indices))).item()
        
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
             
        # Patience and grace threshold logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            grace_patience = 0
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        elif val_loss <= best_val_loss * (1 + grace_threshold):
            grace_patience += 1
            print(f"Validation loss increased slightly but is within the grace threshold: {val_loss:.4f}")
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

# Initialize scalers for each group
group_scalers = {group: StandardScaler() for group in input_groups}

for group, features in input_groups.items():
    available_features = [f for f in features if f in train_df.columns]
    if available_features:
        group_scalers[group].fit(train_df[available_features])

# Fit scalers on training data
for group, features in input_groups.items():
    available_features = [f for f in features if f in train_df.columns]
    if available_features:
        group_scalers[group].fit(train_df[available_features])

# Transform each group in the train, validation, and test datasets
for df in [train_df, val_df, test_df]:
    for group, features in input_groups.items():
        available_features = [f for f in features if f in df.columns]
        if available_features:
            df[available_features] = group_scalers[group].transform(df[available_features])


from sklearn.preprocessing import StandardScaler

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

# Define variable difficulty (manual or based on metrics)
easy_variables = ["Skin Conductance (microS)", "Tonic Skin Conductance (a.u.)"]
medium_variables = ["Heart Rate (bpm)", "Phasic Skin Conductance (a.u.)"]
hard_variables = ["Skin Conductance Phasic Driver (a.u.)", "Emotional Index (a.u.)", "Sympathovagal Balance (a.u.)"]

# Map variable names to indices
variable_indices = {var: i for i, var in enumerate(target_cols)}


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

class LSTMAttnMultiHead(nn.Module):
    def __init__(self, input_sizes, hidden_size, output_sizes, num_layers=3):
        super().__init__()
        self.input_heads = nn.ModuleDict({
            group_name: nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)  # Reduced dropout
            ) for group_name, input_size in input_sizes.items()
        })
        self.lstm = nn.LSTM(
            hidden_size * len(input_sizes), 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2  # Reduced dropout
        )
        self.bn = nn.LayerNorm(hidden_size)  # Replace BatchNorm1d with LayerNorm
        self.heads = nn.ModuleList([nn.Linear(hidden_size, sz) for sz in output_sizes])

    def forward(self, x_groups):
        processed_groups = []
        for group_name, x_group in x_groups.items():
            x_flat = x_group.reshape(-1, x_group.size(-1))
            processed = self.input_heads[group_name](x_flat)
            processed = processed.reshape(x_group.size(0), x_group.size(1), -1)
            processed_groups.append(processed)
        combined = torch.cat(processed_groups, dim=-1)
        out, (h_n, _) = self.lstm(combined)
        context = self.bn(h_n[-1])  # BatchNorm applied
        return torch.cat([head(context) for head in self.heads], dim=1)

# Initialize the multi-head model
# Map feature names to their indices in available_feature_cols
feature_indices = {
    group: [available_feature_cols.index(f) for f in features if f in available_feature_cols]
    for group, features in input_groups.items()
}

# Update input sizes based on available features
input_sizes = {group: len(indices) for group, indices in feature_indices.items()}
hidden_size = 128
output_sizes = [1] * len(available_target_cols)
epochs = 40  # Total number of epochs you intend to train for
batch_size = 64

# Initialize the model with updated input sizes
model = LSTMAttnMultiHead(input_sizes=input_sizes, hidden_size=128, output_sizes=output_sizes)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)


# Create grouped inputs
x_train_groups = {group: torch.tensor(X_train[:, :, indices], dtype=torch.float32)
                  for group, indices in feature_indices.items()}
x_val_groups = {group: torch.tensor(X_val[:, :, indices], dtype=torch.float32)
                for group, indices in feature_indices.items()}
x_test_groups = {group: torch.tensor(X_test[:, :, indices], dtype=torch.float32)
                 for group, indices in feature_indices.items()}


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
    check_interval=20,  # Ask every 5 epochs
    patience=8         # Early stopping patience
)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss", color='blue', linewidth=2)
plt.plot(val_losses, label="Validation Loss", color='orange', linewidth=2)
plt.title("Training and Validation Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate on test data
model.eval()
with torch.no_grad():
    # Group the test data
    x_test_groups = {group: X_test_t[:, :, indices_list] 
                     for group, indices_list in feature_indices.items()}
    
    # Pass grouped test data to the model
    test_outputs = model(x_test_groups).cpu().numpy()  # Predicted values
    y_test_true = Y_test_t.cpu().numpy()  # True values

# Scale back biological variables to the original scale
test_outputs_rescaled = bio_scaler.inverse_transform(test_outputs)
y_test_true_rescaled = bio_scaler.inverse_transform(y_test_true)

# Evaluate metrics
evaluate_metrics(y_test_true_rescaled, test_outputs_rescaled)

# Validation or testing
model.eval()
with torch.no_grad():
    x_val_groups = {group: x_val_groups[group] for group in x_val_groups}
    val_outputs = model(x_val_groups)
    val_loss = criterion(val_outputs, Y_val_t)

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

# Print metrics for debugging (optional)
for i, var in enumerate(variables):
    print(f"{var}: MSE={mse[i]:.4f}, MAE={mae[i]:.4f}, R²={r2[i]:.4f}")

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


# Plot time-series for multiple variables
for var_index, var_name in enumerate(variables):
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

