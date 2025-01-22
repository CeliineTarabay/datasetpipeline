import os
import pandas as pd

# Define source and destination directories
env_source = "./envsource"
bio_source = "./biosource"
destination = "./data"

# Ensure destination directory exists
os.makedirs(destination, exist_ok=True)

# Conversion factor for wind speed (mph to km/h)
CONVERSION_FACTOR = 1.60934

# Function to process CSV files
def process_files(source_path, suffix):
    if not os.path.exists(source_path):
        print(f"Source directory '{source_path}' does not exist.")
        return

    files = [f for f in os.listdir(source_path) if f.lower().endswith('.csv')]

    if not files:
        print(f"No CSV files found in '{source_path}'.")
        return

    for file_name in files:
        original_path = os.path.join(source_path, file_name)

        # Extract two-digit number from filename
        number = ''.join(filter(str.isdigit, file_name))[:2]

        if number:
            padded_number = number.zfill(3)
            new_name = f"{padded_number}_{suffix}.csv"
            dest_path = os.path.join(destination, new_name)

            try:
                # Read the CSV file
                data = pd.read_csv(original_path)

                if suffix == "env":
                    # Identify wind speed columns
                    wind_columns = [col for col in data.columns if "wind speed" in col.lower()]

                    for i, col in enumerate(wind_columns, start=1):
                        # Check if the column name contains "mph" for conversion
                        if "mph" in col.lower():
                            # Convert values and rename column
                            new_col_name = f"Wind Speed (km/hr) Run {i}"
                            data[new_col_name] = data[col] * CONVERSION_FACTOR
                        else:
                            # Rename column without conversion
                            new_col_name = f"Wind Speed Run {i}"
                            data[new_col_name] = data[col]

                        # Drop the original column
                        data.drop(columns=[col], inplace=True)

                    # Rename "Tempo (s) Raccolta 1" to "Time (s) Run 1", if it exists
                    if "Tempo (s) Raccolta 1" in data.columns:
                        data.rename(columns={"Tempo (s) Raccolta 1": "Time (s) Run 1"}, inplace=True)

                # Save to the destination directory
                data.to_csv(dest_path, index=False)
                print(f"Copied and processed '{file_name}' to '{new_name}'.")

            except Exception as e:
                print(f"Failed to process '{file_name}'. Error: {e}")
        else:
            print(f"No valid number found in '{file_name}'. Skipping.")

# Process environment and bio source files
process_files(env_source, "env")
process_files(bio_source, "bio")

print("File processing completed.")
