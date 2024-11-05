import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from datetime import datetime

# Reset Matplotlib to default settings to prevent residual configurations
plt.rcParams.update(plt.rcParamsDefault)

# Load and process sunspot data from a provided CSV file
def load_data(file_path, date_col="date", value_col="sunspot", moving_average=None):
    try:
        # Parse the date column and sort the DataFrame by date
        df = pd.read_csv(file_path, parse_dates=[date_col])
        df = df.sort_values(by=date_col)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing '{file_path}': {e}")
        return None

    if value_col not in df.columns:
        print(f"Error: Value column '{value_col}' not found in the data.")
        return None

    # Apply moving average if specified
    if moving_average:
        df[value_col] = df[value_col].rolling(window=moving_average, min_periods=1).mean()

    return df.set_index(date_col)[value_col]

# Generate sine wave based on indices with frequency scaling
def generate_sine_wave_with_indices(params, indices, set_negatives_zero=False, step_conversion_factor=1.0):
    amplitude = params.get("amplitude", 1.0)
    frequency = params.get("frequency", 0.001) * step_conversion_factor  # Adjust frequency based on step
    phase_shift = params.get("phase_shift", 0.0)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * indices + phase_shift)
    if set_negatives_zero:
        sine_wave = np.maximum(sine_wave, 0)
    return sine_wave

# Load previous sine waves and combine them based on date indices
def load_and_combine_waves_with_indices(date_index_tuples, waves_dir, set_negatives_zero=False, step_conversion_factor=1.0):
    num_points = len(date_index_tuples)
    combined_wave = np.zeros(num_points, dtype=np.float32)

    if not os.path.exists(waves_dir):
        print(f"Error: Waves directory '{waves_dir}' does not exist.")
        return combined_wave

    # Retrieve all JSON wave files
    wave_files = sorted([f for f in os.listdir(waves_dir) if f.endswith(".json")])
    print(f"Loading {len(wave_files)} wave(s) from '{waves_dir}'")

    for filename in wave_files:
        wave_path = os.path.join(waves_dir, filename)
        try:
            with open(wave_path, "r") as f:
                wave_params = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Wave file '{filename}' is not a valid JSON. Skipping.")
            continue
        except Exception as e:
            print(f"Error reading '{filename}': {e}. Skipping.")
            continue

        amplitude = wave_params.get("amplitude", 1.0)
        frequency = wave_params.get("frequency", 0.001)
        phase_shift = wave_params.get("phase_shift", 0.0)
        print(f"Loading Wave: {filename}, Amplitude: {amplitude}, Frequency: {frequency}, Phase Shift: {phase_shift}")

        # Extract indices from date_index_tuples
        indices = np.array([idx for _, idx in date_index_tuples], dtype=np.float32)
        # Generate the sine wave based on indices
        wave = generate_sine_wave_with_indices(wave_params, indices, set_negatives_zero, step_conversion_factor)
        # Combine the wave
        combined_wave += wave
        print(f"After adding {filename}, Combined Wave max: {combined_wave.max()}, min: {combined_wave.min()}")

    if set_negatives_zero:
        combined_wave = np.maximum(combined_wave, 0)

    print(f"Final Combined Wave Statistics:\nMin: {combined_wave.min()}, Max: {combined_wave.max()}, Mean: {combined_wave.mean()}")
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

    # Close any existing plots to prevent overlap
    plt.close('all')

    # Load sunspot data
    sunspot_data = load_data(args.data_file, date_col=args.date_col, value_col=args.value_col, moving_average=args.moving_average)
    if sunspot_data is None:
        print("No data loaded. Exiting.")
        return

    # Determine observed start and end dates
    observed_start = sunspot_data.index.min()
    observed_end = sunspot_data.index.max()
    time_range_days = (observed_end - observed_start).days

    print(f"Observed Data Start Date: {observed_start.date()}")
    print(f"Observed Data End Date: {observed_end.date()}")
    print(f"Time Range (days): {time_range_days}")

    # Infer data frequency
    data_freq = sunspot_data.index.inferred_freq
    if data_freq is None:
        # Calculate average time difference
        if len(sunspot_data.index) > 1:
            step_deltas = sunspot_data.index.to_series().diff().dropna()
            avg_time_diff_days = step_deltas.dt.days.mean()
            step_conversion_factor = avg_time_diff_days
            print(f"Could not infer frequency. Calculated average time difference: {avg_time_diff_days:.2f} days.")
            # Assuming monthly data if average is around 30 days
            if 28 <= avg_time_diff_days <= 31:
                data_freq = 'MS'  # Month Start frequency
                print("Assuming monthly frequency ('MS').")
                step_conversion_factor = 30  # Approximate days per month
            else:
                # Default to daily frequency
                data_freq = 'D'
                print("Defaulting to daily frequency ('D').")
                step_conversion_factor = 1.0
        else:
            step_conversion_factor = 1.0
            print("Only one data point present. Setting step_conversion_factor to 1.0.")
    else:
        print(f"Inferred data frequency: {data_freq}")
        # Determine step_conversion_factor based on frequency
        if 'M' in data_freq:
            step_conversion_factor = 30  # Approximate days per month
        elif 'D' in data_freq:
            step_conversion_factor = 1.0
        else:
            # Add more frequency mappings as needed
            step_conversion_factor = 1.0
            print(f"Frequency '{data_freq}' not specifically handled. Setting step_conversion_factor to 1.0.")

    # Calculate number of steps to predict before and after based on percentage
    total_steps = len(sunspot_data)
    predict_before_steps = int(total_steps * (args.predict_before / 100))
    predict_after_steps = int(total_steps * (args.predict_after / 100))

    print(f"Predicting {predict_before_steps} step(s) before the observed data.")
    print(f"Predicting {predict_after_steps} step(s) after the observed data.")

    # Define extended start and end dates using the inferred frequency
    try:
        extended_start_date = observed_start - pd.tseries.frequencies.to_offset(data_freq) * predict_before_steps
        extended_end_date = observed_end + pd.tseries.frequencies.to_offset(data_freq) * predict_after_steps
    except ValueError as e:
        print(f"Error in date offset calculation: {e}")
        return

    print(f"Extended Start Date: {extended_start_date.date()}")
    print(f"Extended End Date: {extended_end_date.date()}")

    # Generate the full date range based on inferred frequency
    full_date_range = pd.date_range(start=extended_start_date, end=extended_end_date, freq=data_freq)
    print(f"Total Number of Points: {len(full_date_range)}")

    # Create list of tuples (datetime, index)
    # Index 0 corresponds to the first date in the observed data
    # Each subsequent index increments by 1 per step (month)
    date_index_tuples = [(dt, i) for i, dt in enumerate(full_date_range)]

    # For debugging: Print a sample of date-index mapping
    print("Sample Date-Index Mapping:")
    for dt, idx in date_index_tuples[:5]:
        print(f"Date: {dt.date()}, Index: {idx}")
    print("...")

    # Generate reconstructed data using the extended date range
    reconstructed_data = load_and_combine_waves_with_indices(
        date_index_tuples, 
        args.waves_dir, 
        set_negatives_zero=args.set_negatives_zero,
        step_conversion_factor=step_conversion_factor
    )
    reconstructed_dates = [dt for dt, idx in date_index_tuples]

    # Additional Debugging: Print reconstructed data statistics
    print(f"Reconstructed Data Statistics:\nMin: {reconstructed_data.min()}, Max: {reconstructed_data.max()}, Mean: {reconstructed_data.mean()}")

    # Save reconstructed data for external verification
    np.save('reconstructed_data.npy', reconstructed_data)
    print("Reconstructed data saved to 'reconstructed_data.npy'")

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot Observed Data with Markers for Clarity
    plt.plot(sunspot_data.index, sunspot_data.values, label="Observed Data", color="blue", marker='o')

    # Plot Combined Sine Waves
    plt.plot(reconstructed_dates, reconstructed_data, label="Combined Sine Waves", color="orange")

    # Annotate the Plot with Statistics
    stats_text = (
        f"Combined Wave Stats:\n"
        f"Min: {reconstructed_data.min():.2f}\n"
        f"Max: {reconstructed_data.max():.2f}\n"
        f"Mean: {reconstructed_data.mean():.2f}"
    )
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.title(f"Sunspot Data from {extended_start_date.year} to {extended_end_date.year}")
    plt.xlabel("Date")
    plt.ylabel("Sunspot Number")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
