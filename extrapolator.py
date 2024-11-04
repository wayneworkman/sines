import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
import argparse
from datetime import datetime, timedelta

# Constants for date range
START_DATE = datetime(1677, 9, 22)
END_DATE = datetime(2262, 4, 10)

# Download and preprocess sunspot data
def download_and_process_sunspot_data(url):
    """Downloads, processes, and applies a 27-day moving average to the sunspot data."""
    response = requests.get(url)
    data = response.text.strip().splitlines()
    sunspot_data = []

    for line in data:
        fields = line.split(';')
        if len(fields) < 5:
            continue
        try:
            year, month, day = int(fields[0]), int(fields[1]), int(fields[2])
            sn_value = int(float(fields[4].strip()))
            sunspot_data.append((datetime(year, month, day), sn_value))
        except (ValueError, IndexError):
            continue

    # Create DataFrame and apply a 27-day moving average
    sunspot_df = pd.DataFrame(sunspot_data, columns=['date', 'sunspot'])
    sunspot_df.set_index('date', inplace=True)
    sunspot_df['sunspot'] = sunspot_df['sunspot'].rolling(window=27, min_periods=1).mean()
    return sunspot_df

# Generate sine wave based on parameters
def generate_sine_wave(params, num_points):
    """Generates a sine wave using amplitude, frequency, and phase shift."""
    amplitude, frequency, phase_shift = params["amplitude"], params["frequency"], params["phase_shift"]
    t = np.arange(num_points, dtype=np.float32)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    return np.maximum(sine_wave, 0)  # Enforce non-negative values

# Load previous sine waves and combine them
def load_and_combine_waves(start_date, end_date, waves_dir):
    """Loads sine wave parameters from JSON files in the specified directory and combines them."""
    num_points = (end_date - start_date).days + 1
    combined_wave = np.zeros(num_points, dtype=np.float32)

    # Load each wave and sum their daily values into combined_wave
    for filename in sorted(os.listdir(waves_dir)):
        if filename.endswith(".json"):
            file_path = os.path.join(waves_dir, filename)
            with open(file_path, "r") as f:
                wave_params = json.load(f)
                wave = generate_sine_wave(wave_params, num_points)
                combined_wave += wave

    # Ensure no negative values
    combined_wave = np.maximum(combined_wave, 0)
    return combined_wave

# Main function for generating and plotting the data
def main():
    parser = argparse.ArgumentParser(description="Sunspot Data Extrapolator Using Sine Waves")
    parser.add_argument('--waves-dir', type=str, default="waves",
                        help="Directory containing sine wave JSON files (default: 'waves')")
    args = parser.parse_args()

    waves_dir = args.waves_dir

    # Check if waves directory exists
    if not os.path.exists(waves_dir):
        print(f"Error: Waves directory '{waves_dir}' does not exist.")
        return

    # Fixed date range
    start_date = START_DATE
    end_date = END_DATE

    # Download actual sunspot data
    url = "http://www.sidc.be/silso/INFO/sndtotcsv.php"
    sunspot_data = download_and_process_sunspot_data(url)

    # Check if sunspot_data DataFrame is empty
    if sunspot_data.empty:
        print("Warning: No actual sunspot data was downloaded or processed.")
        return

    print(f"Actual data available from {sunspot_data.index.min()} to {sunspot_data.index.max()}")

    # Filter actual data to match the specified date range
    actual_data_in_range = sunspot_data[(sunspot_data.index >= start_date) & (sunspot_data.index <= end_date)]

    # Check if filtered data is empty
    if actual_data_in_range.empty:
        print("Warning: No actual sunspot data available for the specified date range.")
    else:
        print(f"Actual sunspot data from {actual_data_in_range.index.min()} to {actual_data_in_range.index.max()}")

    # Generate reconstructed data for the specified range
    reconstructed_data = load_and_combine_waves(start_date, end_date, waves_dir)
    reconstructed_dates = pd.date_range(start=start_date, end=end_date)
    
    # Ensure the lengths of reconstructed data and dates match
    if len(reconstructed_data) != len(reconstructed_dates):
        raise ValueError("Mismatch between the length of reconstructed data and date range.")

    reconstructed_series = pd.Series(reconstructed_data, index=reconstructed_dates, name='Reconstructed Sunspots')

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot actual data if it falls within the requested range
    if not actual_data_in_range.empty:
        plt.plot(actual_data_in_range.index, actual_data_in_range['sunspot'], label="Actual Sunspots", color="blue")

    # Plot reconstructed data
    plt.plot(reconstructed_series.index, reconstructed_series.values, label="Reconstructed Sunspots", color="orange")

    # Configure plot
    plt.title(f"Sunspot Data from {START_DATE.year} to {END_DATE.year}")
    plt.xlabel("Date")
    plt.ylabel("Sunspot Number")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
