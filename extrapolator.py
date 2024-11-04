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

# Generate sine wave based on parameters

def generate_sine_wave(params, t):
    amplitude, frequency, phase_shift = params["amplitude"], params["frequency"], params["phase_shift"]
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    return np.maximum(sine_wave, 0)


# Load previous sine waves and combine them
def load_and_combine_waves(start_date, end_date, waves_dir):

    num_points = (end_date - start_date).days + 1
    t = np.arange(num_points, dtype=np.float32)  # Index-based t values to match sines.py
    combined_wave = np.zeros(num_points, dtype=np.float32)

    for filename in sorted(os.listdir(waves_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(waves_dir, filename), "r") as f:
                wave_params = json.load(f)
                wave = generate_sine_wave(wave_params, t)
                combined_wave += wave

    return np.maximum(combined_wave, 0)

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
    args = parser.parse_args()

    # Load sunspot data
    sunspot_data = load_data(args.data_file, date_col=args.date_col, value_col=args.value_col, moving_average=args.moving_average)
    if sunspot_data is None:
        print("No data loaded.")
        return

    # Calculate start and end dates to match the observed data
    observed_start = sunspot_data.index.min()
    observed_end = sunspot_data.index.max()
    time_range = observed_end - observed_start

    start_date = observed_start - pd.Timedelta(days=int(time_range.days * (args.predict_before / 100)))
    end_date = observed_end + pd.Timedelta(days=int(time_range.days * (args.predict_after / 100)))
    actual_data_in_range = sunspot_data[(sunspot_data.index >= start_date) & (sunspot_data.index <= end_date)]


    # Generate reconstructed data
    reconstructed_data = load_and_combine_waves(start_date, end_date, args.waves_dir)
    reconstructed_dates = pd.date_range(start=start_date, end=end_date)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data_in_range.index, actual_data_in_range.values, label="Observed Data", color="blue")
    plt.plot(reconstructed_dates, reconstructed_data, label="Combined Sine Waves", color="orange")
    plt.title(f"Sunspot Data from {start_date.year} to {end_date.year}")
    plt.xlabel("Date")
    plt.ylabel("Sunspot Number")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()