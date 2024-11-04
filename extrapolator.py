import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from datetime import datetime

# Load and process sunspot data from a provided CSV file
def load_data(file_path, date_col="date", value_col="sunspot", moving_average=None):
    try:
        df = pd.read_csv(file_path, parse_dates=[date_col])
        df = df.sort_values(by=date_col)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

    if moving_average:
        df[value_col] = df[value_col].rolling(window=moving_average, min_periods=1).mean()
    
    return df.set_index(date_col)[value_col]

# Generate sine wave based on indices
def generate_sine_wave_with_indices(params, indices, set_negatives_zero=False):
    amplitude = params["amplitude"]
    frequency = params["frequency"]
    phase_shift = params["phase_shift"]
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * indices + phase_shift)
    if set_negatives_zero:
        sine_wave = np.maximum(sine_wave, 0)
    return sine_wave

# Load previous sine waves and combine them based on date indices
def load_and_combine_waves_with_indices(date_index_tuples, waves_dir, set_negatives_zero=False):
    num_points = len(date_index_tuples)
    combined_wave = np.zeros(num_points, dtype=np.float32)

    for filename in sorted(os.listdir(waves_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(waves_dir, filename), "r") as f:
                wave_params = json.load(f)
                # Extract indices from the tuples
                indices = np.array([idx for _, idx in date_index_tuples], dtype=np.float32)
                wave = generate_sine_wave_with_indices(wave_params, indices, set_negatives_zero)
                combined_wave += wave

    if set_negatives_zero:
        return np.maximum(combined_wave, 0)
    else:
        return combined_wave

# Main function for generating and plotting the data
def main():
    parser = argparse.ArgumentParser(description="Sunspot Data Extrapolator Using Sine Waves")
    parser.add_argument('--predict-before', type=float, default=5.0,
                        help="Percentage of time to predict before the observed data (default: 5%%)")
    parser.add_argument('--predict-after', type=float, default=5.0,
                        help="Percentage of time to predict after the observed data (default: 5%%)")
    parser.add_argument('--data-file', type=str, required=True, help="Path to the sunspot data file")
    parser.add_argument('--date-col', type=str, default="date", help="Name of the date column in the data")
    parser.add_argument('--value-col', type=str, default="sunspot", help="Name of the value column in the data")
    parser.add_argument('--moving-average', type=int, help="Apply a moving average filter to smooth the data")
    parser.add_argument('--waves-dir', type=str, default="waves", help="Directory containing sine wave JSON files (default: 'waves')")
    parser.add_argument('--set-negatives-zero', action='store_true', default=False,
                        help="Set sine wave values below zero to zero (default: False)")
    args = parser.parse_args()

    # Load sunspot data
    sunspot_data = load_data(args.data_file, date_col=args.date_col, value_col=args.value_col, moving_average=args.moving_average)
    if sunspot_data is None:
        print("No data loaded.")
        return

    # Determine observed start and end dates
    observed_start = sunspot_data.index.min()
    observed_end = sunspot_data.index.max()
    time_range_days = (observed_end - observed_start).days

    # Calculate number of days to predict before and after
    predict_before_days = int(time_range_days * (args.predict_before / 100))
    predict_after_days = int(time_range_days * (args.predict_after / 100))

    # Define extended start and end dates
    extended_start_date = observed_start - pd.Timedelta(days=predict_before_days)
    extended_end_date = observed_end + pd.Timedelta(days=predict_after_days)

    # Generate the full date range
    full_date_range = pd.date_range(start=extended_start_date, end=extended_end_date, freq='D')

    # Create list of tuples (datetime, index)
    # Index 0 corresponds to observed_start
    date_index_tuples = []
    for single_date in full_date_range:
        index = (single_date - observed_start).days
        date_index_tuples.append((single_date, index))

    # For debugging: Print a sample of date-index mapping
    print("Sample Date-Index Mapping:")
    for dt, idx in date_index_tuples[:5]:
        print(f"Date: {dt.date()}, Index: {idx}")
    print("...")

    # Generate reconstructed data using the extended date range
    reconstructed_data = load_and_combine_waves_with_indices(
        date_index_tuples, 
        args.waves_dir, 
        set_negatives_zero=args.set_negatives_zero
    )
    reconstructed_dates = [dt for dt, idx in date_index_tuples]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(sunspot_data.index, sunspot_data.values, label="Observed Data", color="blue")
    plt.plot(reconstructed_dates, reconstructed_data, label="Combined Sine Waves", color="orange")
    plt.title(f"Sunspot Data from {extended_start_date.year} to {extended_end_date.year}")
    plt.xlabel("Date")
    plt.ylabel("Sunspot Number")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
