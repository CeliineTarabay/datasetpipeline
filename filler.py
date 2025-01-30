import pandas as pd
from scipy.interpolate import interp1d
import argparse
from datetime import datetime

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fill missing values in a dataset.")
parser.add_argument('--method', type=str, required=True, choices=['linear', 'quadratic', 'cubic', 'nearest', 'mean', 'median', 'mode', 'zero'], help="Method to fill missing values: linear, quadratic, cubic, nearest, mean, median, mode, or zero.")
args = parser.parse_args()

# Load the dataset
input_file = 'dataset1.csv'
output_file = 'dataset2.csv'
method = args.method

data = pd.read_csv(input_file)

# Handle other missing values in the dataset if needed
def fill_remaining_values(df, method):
    for column in df.columns:
        if column not in one_hot_columns and df[column].isnull().any():
            if df[column].dtype in ['float64', 'int64']:
                # Interpolate or use the specified method for numeric columns
                if method in ['linear', 'quadratic', 'cubic', 'nearest']:
                    df[column] = df[column].interpolate(method=method, limit_direction='both')
                df[column] = df[column].ffill().bfill()
            else:
                # Forward and backward fill for non-numeric columns
                df[column] = df[column].ffill().bfill()
    return df

# Fill the gaps in one-hot encoded columns using forward-fill logic
def fill_one_hot_columns(df):
    for column in one_hot_columns:
        if column in df.columns:
            df[column] = df[column].fillna(method='ffill')
    return df

def split_geometry_column(df):
    if 'geometry' in df.columns:
        # Extract Latitude and Longitude from the 'geometry' column
        df[['Longitude', 'Latitude']] = df['geometry'].str.extract(r'POINT \(([-\d\.]+) ([-\d\.]+)\)').astype(float)
        # Drop the original 'geometry' column if no longer needed
        df = df.drop(columns=['geometry'])
    return df

# Standardize the Timestamp column
def preprocess_timestamp_column(df):
    # Fill missing or empty timestamps with NaT
    df['Timestamp'] = df['Timestamp'].replace(r'^\s*$', pd.NaT, regex=True)
    
    # Append '.000' to timestamps without milliseconds
    df['Timestamp'] = df['Timestamp'].astype(str).str.replace(
        r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', r'\g<0>.000', regex=True
    )

    # Convert to datetime, coercing errors
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', format='%Y-%m-%d %H:%M:%S.%f')

    # Add derived columns, ensuring NaT values do not cause errors
    df['DayOfYear'] = df['Timestamp'].dt.dayofyear
    df['Year'] = df['Timestamp'].dt.year
    df['SecondsOfDay'] = df['Timestamp'].dt.hour * 3600 + df['Timestamp'].dt.minute * 60 + df['Timestamp'].dt.second
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6

    # Calculate minimum distance from 15th of July
    def calculate_jul_15(year):
        if pd.isna(year):
            return pd.NaT
        return datetime(int(year), 7, 15)

    jul_15 = df['Year'].apply(calculate_jul_15)
    df['DaysFromJuly15'] = (df['Timestamp'] - jul_15).dt.days.abs()
    return df

# Define the columns for one-hot encoding
one_hot_columns = [
    'PRE', 'Baseline', 'Baseline -> Cafe', 'Cafe', 'Cafe -> Square', 'Square',
    'Square -> Crosswalk', 'Crosswalk', 'Crosswalk -> U3', 'U3', 
    'U3 -> Trivulziana', 'Trivulziana', 'Trivulziana -> U4', 'U4', 
    'U4 -> Tram', 'Tram', 'POST', 'static'
]



# Main logic
# Preprocess the Timestamp column to standardize format
data = preprocess_timestamp_column(data)

# Handle missing numerical values before dropping Timestamp
data = fill_remaining_values(data, method)

# Count unique SubjectIDs with invalid (non-empty, but unparseable) timestamps
invalid_timestamps = data[data['Timestamp'].isna()]
subjects_with_nat = invalid_timestamps['SubjectID'].nunique()
subjects_with_nat_list = invalid_timestamps['SubjectID'].unique()

# Display a message about the number of subjects with invalid timestamps
if subjects_with_nat > 0:
    print(f"{subjects_with_nat} unique subjects have unparseable timestamps.")
    print(f"The affected SubjectIDs are: {', '.join(map(str, subjects_with_nat_list))}")
else:
    print("No subjects have unparseable timestamps.")


# Split 'geometry' column into 'Longitude' and 'Latitude'
data = split_geometry_column(data)

# Drop the original 'geometry' column if no longer needed
if 'geometry' in data.columns:
    data = data.drop(columns=['geometry'])



# Drop 'Timestamp' column
if 'Timestamp' in data.columns:
    data = data.drop(columns=['Timestamp'], errors='raise')

# Process one-hot encoded columns, preserving SubjectID
filled_data = data.groupby('SubjectID', group_keys=False).apply(
    lambda group: fill_one_hot_columns(group)
)

# Ensure remaining missing values are handled
filled_data = filled_data.groupby('SubjectID', group_keys=False).apply(
    lambda group: fill_remaining_values(group, method)
).reset_index(drop=True)

# Save the result
filled_data.to_csv(output_file, index=False)
print(f"Missing values filled using {method} method and saved to {output_file}.")
