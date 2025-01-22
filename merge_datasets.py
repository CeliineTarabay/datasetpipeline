import os
import pandas as pd
import re

def get_subject_ids(data_dir):
    """
    Scans the data directory and returns a set of subject IDs
    that have both bio and env files.
    """
    env_pattern = re.compile(r'(\d{3})_env\.csv$', re.IGNORECASE)
    bio_pattern = re.compile(r'(\d{3})_bio\.csv$', re.IGNORECASE)
    
    env_files = set()
    bio_files = set()
    
    for file in os.listdir(data_dir):
        env_match = env_pattern.match(file)
        bio_match = bio_pattern.match(file)
        if env_match:
            env_files.add(env_match.group(1))
        if bio_match:
            bio_files.add(bio_match.group(1))
    
    # Only include subjects that have both env and bio files
    subject_ids = env_files.intersection(bio_files)
    return subject_ids

def process_subject(data_dir, subject_id):
    """
    Processes a single subject by merging bio and env files.

    Parameters:
        data_dir (str): Path to the data directory.
        subject_id (str): Three-digit subject ID.

    Returns:
        pd.DataFrame or None: Merged DataFrame for the subject, or None if skipped.
    """
    bio_file = os.path.join(data_dir, f"{subject_id}_bio.csv")
    env_file = os.path.join(data_dir, f"{subject_id}_env.csv")
    
    # Read Bio File
    try:
        bio_df = pd.read_csv(bio_file)
        print(f"[DEBUG] Bio file columns for subject {subject_id}: {bio_df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading bio file for subject {subject_id}: {e}")
        return None
    
    # Ensure required columns exist in bio_df
    if 'Recording start' not in bio_df.columns:
        print(f"'Recording start' column not found in bio file for subject {subject_id}.")
        return None
    
    # Exclude 'ID' and 'Recording start' columns from Bio
    bio_columns_to_keep = [col for col in bio_df.columns if col not in ['ID', 'Recording start']]
    bio_df = bio_df[bio_columns_to_keep]
    
    # Read Timestamp from 'Recording start' in second row
    try:
        bio_timestamp = pd.read_csv(bio_file, nrows=2).iloc[1]['Recording start']
        bio_timestamp = pd.to_datetime(bio_timestamp)
        print(f"[DEBUG] Extracted bio_timestamp for subject {subject_id}: {bio_timestamp}")
    except Exception as e:
        print(f"Error reading timestamp from bio file for subject {subject_id}: {e}")
        return None
    
    # Read Env File
    try:
        env_df = pd.read_csv(env_file)
        print(f"[DEBUG] Env file columns for subject {subject_id}: {env_df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading env file for subject {subject_id}: {e}")
        return None

    # Drop specific columns: 'index' and 'Unnamed: 0'
    columns_to_drop = ['index', 'Unnamed: 0']
    env_df = env_df.drop(columns=columns_to_drop, errors='ignore')
    
    # Ensure 'Timestamp' column exists in env_df
    if 'Timestamp' not in env_df.columns:
        print(f"Timestamp column 'Timestamp' not found in env file for subject {subject_id}.")
        return None
    
    # Convert 'Timestamp' from nanoseconds since epoch to datetime
    try:
        env_df['Timestamp'] = pd.to_datetime(env_df['Timestamp'], unit='ns')
    except Exception as e:
        print(f"Error converting timestamps in env file for subject {subject_id}: {e}")
        return None
    
    # Filter env data starting from bio_timestamp
    env_df = env_df[env_df['Timestamp'] >= bio_timestamp].reset_index(drop=True)
    print(f"[DEBUG] Filtered env data for subject {subject_id}: {env_df.shape[0]} rows")

    # Check if env_df is empty or insufficient
    num_bio_rows = bio_df.shape[0]
    num_env_rows = env_df.shape[0]
    frequency_ratio = 8  # Number of bio rows per env row

    # Ensure enough env rows to align with bio rows
    expected_env_rows = num_bio_rows // frequency_ratio
    if num_env_rows < expected_env_rows:
        print(f"Not enough env data for subject {subject_id}. Skipping.")
        return None

    # Trim env_df to expected_env_rows
    env_df = env_df.head(expected_env_rows)

    # Assign env data to every 8th bio row
    bio_df['Timestamp'] = pd.NaT  # Initialize 'Timestamp' column with NaT

    # Add all relevant env columns to bio_df
    for column in env_df.columns:
        if column != 'Timestamp':  # Skip Timestamp since it's already handled
            bio_df[column] = pd.NA  # Initialize with NaN
            bio_df.loc[::frequency_ratio, column] = env_df[column].values

    # Assign Timestamp separately
    bio_df.loc[::frequency_ratio, 'Timestamp'] = env_df['Timestamp'].values

    # Add Subject ID column
    bio_df.insert(0, 'SubjectID', subject_id)

    return bio_df


def main():
    # Determine the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set data_dir to the 'data' folder within the script's directory
    data_dir = os.path.join(script_dir, 'data')
    
    # Ensure the data directory exists
    if not os.path.isdir(data_dir):
        print(f"Data directory '{data_dir}' does not exist.")
        return
    
    # Get list of subject IDs with both env and bio files
    subject_ids = get_subject_ids(data_dir)
    
    if not subject_ids:
        print("No subjects with both env and bio files found.")
        return
    
    print(f"Found {len(subject_ids)} subjects with both env and bio files.")
    
    # Lists to track processing status
    processed_subjects = []
    skipped_subjects = []
    
    # List to hold merged data for all subjects
    all_subjects_data = []
    
    for subject_id in sorted(subject_ids):
        print(f"Processing subject {subject_id}...")
        merged_df = process_subject(data_dir, subject_id)
        if merged_df is not None:
            all_subjects_data.append(merged_df)
            processed_subjects.append(subject_id)
            print(f"Subject {subject_id} processed successfully.")
        else:
            skipped_subjects.append(subject_id)
            print(f"Subject {subject_id} skipped due to processing errors.")
    
    if not all_subjects_data:
        print("No data merged. Exiting.")
        return
    
    # Concatenate all subjects' data into a single DataFrame
    total_dataset = pd.concat(all_subjects_data, ignore_index=True)
    
    # Save the total dataset to a CSV file
    output_file = os.path.join(script_dir, 'dataset1.csv')
    try:
        total_dataset.to_csv(output_file, index=False)
        print(f"Total dataset saved to '{output_file}'.")
    except Exception as e:
        print(f"Error saving total dataset: {e}")
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Processed subjects: {len(processed_subjects)}")
    print(f"Skipped subjects: {len(skipped_subjects)}")
    if skipped_subjects:
        print(f"Skipped subject IDs: {', '.join(skipped_subjects)}")

if __name__ == "__main__":
    main()
