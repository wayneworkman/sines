import pandas as pd
import os
import requests
from io import StringIO

# URL of the metadata file
url = 'https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/M4-info.csv'

# Download the CSV file content from the URL
response = requests.get(url)
if response.status_code == 200:
    csv_data = StringIO(response.text)
    m4_info_df = pd.read_csv(csv_data)
else:
    raise Exception(f"Failed to download CSV. Status code: {response.status_code}")

# Define output directory for reconstructed time series files
output_dir = '/mnt/data/reconstructed_series'
os.makedirs(output_dir, exist_ok=True)

# Define frequency mapping
freq_map = {1: 'Y', 2: 'Q', 3: 'M', 4: 'W', 6: 'D'}

# Function to generate a time series based on row metadata
def reconstruct_time_series(row):
    m4_id = row['M4id']
    starting_date = pd.to_datetime(row['StartingDate'], format='%d-%m-%y %H:%M')
    frequency = row['Frequency']
    horizon = row['Horizon']
    
    # Map frequency to pandas frequency code
    freq_code = freq_map.get(frequency, 'D')  # Default to 'D' if frequency not in map

    # Create date range for time series index
    date_index = pd.date_range(start=starting_date, periods=horizon, freq=freq_code)
    
    # Generate placeholder data (zeros)
    data = [0] * horizon

    # Create DataFrame
    time_series_df = pd.DataFrame({'timestamp': date_index, 'value': data})

    # Save to CSV file
    output_file = os.path.join(output_dir, f'{m4_id}.csv')
    time_series_df.to_csv(output_file, index=False)

# Process each row in the metadata DataFrame
for _, row in m4_info_df.iterrows():
    reconstruct_time_series(row)

print(f"Reconstructed time series saved to {output_dir}")
