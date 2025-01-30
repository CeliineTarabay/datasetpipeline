import pandas as pd
import os
from datetime import datetime, time

# Configuration
EXCEL_PATH = 'TIMESTAMPS.xlsx'
SHEET_NAME = 'TIMESTAMP PROCOMP TEMPO PC'
DATA_DIR = './data'
STATIC_PHASES = ['Baseline', 'Cafe', 'Square', 'Crosswalk', 'U3', 'Trivulziana', 'U4', 'Tram']
PHASE_COLUMNS = {
    'Baseline': ('D', 'E'),
    'Cafe': ('F', 'G'),
    'Square': ('H', 'I'),
    'Crosswalk': ('J', 'K'),
    'U3': ('L', 'M'),
    'Trivulziana': ('N', 'O'),
    'U4': ('P', 'Q'),
    'Tram': ('R', 'S')
}

def get_phase_intervals(row):
    intervals = []
    prev_end = None
    prev_phase = None

    # Static and Transition Phases
    for phase in STATIC_PHASES:
        start_col, end_col = PHASE_COLUMNS[phase]
        start = row[ord(start_col) - ord('A')]  # Already a datetime.time object
        end = row[ord(end_col) - ord('A')]  # Already a datetime.time object
        intervals.append((phase, start, end))

        # Add transition phase if there was a previous static phase
        if prev_end is not None:
            transition_phase = f"{prev_phase} -> {phase}"
            intervals.append((transition_phase, prev_end, start))
        
        prev_end = end
        prev_phase = phase

    # Add PRE and POST phases
    baseline_start = row[ord(PHASE_COLUMNS['Baseline'][0]) - ord('A')]
    tram_end = row[ord(PHASE_COLUMNS['Tram'][1]) - ord('A')]
    intervals.insert(0, ('PRE', None, baseline_start))
    intervals.append(('POST', tram_end, None))

    return intervals


def time_in_interval(t, start, end):
    # Convert datetime to time if necessary
    if isinstance(start, datetime):
        start = start.time()
    if isinstance(end, datetime):
        end = end.time()

    # Ensure `t` is compatible
    if isinstance(t, datetime):
        t = t.time()

    if start is None:
        return t < end
    elif end is None:
        return t >= start
    else:
        return start <= t < end


def main():
    # Read Excel data
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=None)
    subjects = df.iloc[2:]

    # Iterate over all CSV files in the data folder
    for csv_file in os.listdir(DATA_DIR):
        if csv_file.endswith('_env.csv'):
            subject_id = int(csv_file.split('_')[0])  # Extract subject ID from filename
            csv_path = os.path.join(DATA_DIR, csv_file)

            # Check if subject ID is within the range of available data
            if subject_id > len(subjects):
                print(f"Skipping {csv_file}: Subject ID {subject_id} out of range.")
                continue

            # Get phase intervals for the subject
            try:
                subject_row = subjects.iloc[subject_id - 1]
                intervals = get_phase_intervals(subject_row)
            except Exception as e:
                print(f"Skipping {csv_file}: Failed to retrieve or process timestamps for Subject ID {subject_id}. Error: {e}")
                continue

            # Read and process CSV
            try:
                df_csv = pd.read_csv(csv_path)
                if 'Timestamp' not in df_csv.columns:
                    print(f"Skipping {csv_file}: 'Timestamp' column not found.")
                    continue

                # Initialize phase columns
                phase_names = [phase for phase, _, _ in intervals]
                for phase in phase_names:
                    df_csv[phase] = 0
                df_csv['static'] = 0  # 1 for static

                # Process rows without modifying the Timestamp column
                for index, csv_row in df_csv.iterrows():
                    try:
                        t = pd.to_datetime(csv_row['Timestamp']).time()  # Use for internal logic only
                        for phase, start, end in intervals:
                            if time_in_interval(t, start, end):  # Compare with intervals
                                df_csv.at[index, phase] = 1
                                df_csv.at[index, 'static'] = 1 if phase in STATIC_PHASES else 0
                                break
                    except Exception as e:
                        print(f"Error processing row {index} in {csv_file}: {e}")
                        continue
                        
                # Define the desired column order
                ordered_phases = ['PRE', 'Baseline', 'Baseline -> Cafe', 'Cafe', 'Cafe -> Square',
                                  'Square', 'Square -> Crosswalk', 'Crosswalk', 'Crosswalk -> U3',
                                  'U3', 'U3 -> Trivulziana', 'Trivulziana', 'Trivulziana -> U4',
                                  'U4', 'U4 -> Tram', 'Tram', 'POST', 'static']

                # Move the one-hot encoded columns to the correct order
                existing_columns = list(df_csv.columns)
                reordered_columns = [col for col in existing_columns if col not in ordered_phases] + ordered_phases
                df_csv = df_csv[reordered_columns]

                # Save the reordered CSV
                df_csv.to_csv(csv_path, index=False)
                print(f"Processed and saved {csv_path}")

            except Exception as e:
                print(f"Skipping {csv_file}: Failed to process data for Subject ID {subject_id}. Error: {e}")
                continue


if __name__ == '__main__':
    main()
