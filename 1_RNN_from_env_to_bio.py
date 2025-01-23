import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


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
    """
    Training loop with user interaction and early stopping.
    
    Args:
        model: The PyTorch model to train.
        optimizer: The optimizer for training.
        criterion: The loss function.
        X_train_t: Training input data (tensor).
        Y_train_t: Training target data (tensor).
        X_val_t: Validation input data (tensor).
        Y_val_t: Validation target data (tensor).
        epochs: Total number of epochs to train.
        batch_size: Batch size for training.
        check_interval: Number of epochs after which to ask user if they want to continue.
        patience: Number of epochs to wait for validation loss improvement before stopping.
    """
    best_val_loss = float('inf')
    patience_counter = 0

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
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, Y_val_t).item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {np.mean(batch_losses):.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
        # Check every `check_interval` epochs
        if (epoch + 1) % check_interval == 0 and (epoch + 1) < epochs:
            response = input("Do you want to continue training? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Training interrupted by user. Saving model and exiting.")
                torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
                break
    
    print("Training completed. Best validation loss: {:.4f}".format(best_val_loss))


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

# Normalize environmental and biological variables separately
env_scaler = StandardScaler()
bio_scaler = StandardScaler()

# Fit scalers on training data only
env_scaler.fit(train_df[feature_cols])
bio_scaler.fit(train_df[target_cols])

# Transform datasets
for df in [train_df, val_df, test_df]:
    df[feature_cols] = env_scaler.transform(df[feature_cols])
    df[target_cols] = bio_scaler.transform(df[target_cols])


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
X_train, Y_train = build_dataset(train_df, feature_cols, target_cols, seq_length)
X_val, Y_val = build_dataset(val_df, feature_cols, target_cols, seq_length)
X_test, Y_test = build_dataset(test_df, feature_cols, target_cols, seq_length)

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
input_size = len(feature_cols)
output_size = len(target_cols)
hidden_size = 128
epochs = 40  # Total number of epochs you intend to train for
batch_size=16

model = LSTMAttn(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Call the interactive training loop
interactive_training_loop(
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
