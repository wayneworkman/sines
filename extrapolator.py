import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import shlex
from datetime import datetime

def load_observed_data(file_path, date_col='Timestamp', value_col='Value'):
    """
    Load the observed data from a CSV file and assign sequential indices.

    Parameters:
        file_path (str): Path to the CSV data file.
        date_col (str): Name of the column containing date information.
        value_col (str): Name of the column containing the observed values.

    Returns:
        dates (pd.Series): Series of datetime objects.
        indices (np.ndarray): Array of indices.
        data_values (np.ndarray): Array of observed data values.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=[date_col])
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found in the data.")
        df = df.sort_values(by=date_col).reset_index(drop=True)
        dates = df[date_col]
        data_values = df[value_col].values
        indices = np.arange(len(data_values))
        return dates, indices, data_values
    except FileNotFoundError:
        logging.error(f"Data file '{file_path}' not found.")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def load_sine_waves(waves_dir):
    """
    Load sine wave parameters from JSON files in the specified directory.

    Parameters:
        waves_dir (str): Directory containing sine wave JSON files.

    Returns:
        sine_waves (list): List of dictionaries with sine wave parameters.
    """
    sine_waves = []
    if not os.path.isdir(waves_dir):
        logging.error(f"Waves directory '{waves_dir}' does not exist.")
        raise FileNotFoundError(f"Waves directory '{waves_dir}' does not exist.")
    
    for filename in sorted(os.listdir(waves_dir)):
        if filename.endswith('.json'):
            wave_path = os.path.join(waves_dir, filename)
            try:
                with open(wave_path, 'r') as f:
                    wave_params = json.load(f)
                # Validate required parameters
                if not all(k in wave_params for k in ('amplitude', 'frequency', 'phase_shift')):
                    raise ValueError(f"Wave file '{filename}' is missing required parameters.")
                sine_waves.append(wave_params)
            except json.JSONDecodeError:
                logging.warning(f"Wave file '{filename}' is not a valid JSON. Skipping.")
            except Exception as e:
                logging.warning(f"Could not load wave file '{filename}': {e}. Skipping.")
    if not sine_waves:
        logging.error(f"No valid sine wave JSON files found in '{waves_dir}'.")
        raise ValueError(f"No valid sine wave JSON files found in '{waves_dir}'.")
    return sine_waves

def calculate_average_timespan(dates):
    """
    Calculate the average timespan difference between consecutive data points.

    Parameters:
        dates (pd.Series): Series of datetime objects.

    Returns:
        avg_timespan (float): Average timespan in days.
    """
    if len(dates) < 2:
        logging.warning("Not enough data points to calculate average timespan.")
        return None
    time_deltas = dates.diff().dropna()
    avg_timespan = time_deltas.dt.total_seconds().mean() / 86400  # Convert to days
    logging.info(f"Average timespan between data points: {avg_timespan:.2f} days.")
    return avg_timespan

def generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum'):
    """
    Generate and combine multiple sine waves based on their parameters.

    Parameters:
        sine_waves (list): List of dictionaries with sine wave parameters.
        indices (np.ndarray): Array of indices.
        set_negatives_zero (str): 'after_sum', 'per_wave', or 'none' to determine negative handling.

    Returns:
        combined_wave (np.ndarray): Combined sine wave values.
    """
    if set_negatives_zero not in ['after_sum', 'per_wave', 'none']:
        raise ValueError("set_negatives_zero must be either 'after_sum', 'per_wave', or 'none'")

    # Adjust indices to start from zero
    indices_zero_based = indices - indices[0]

    combined_wave = np.zeros_like(indices_zero_based, dtype=np.float64)
    for idx, wave in enumerate(sine_waves, start=1):
        amplitude = wave['amplitude']
        frequency = wave['frequency']
        phase_shift = wave['phase_shift']
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * indices_zero_based + phase_shift)
        
        if set_negatives_zero == 'per_wave':
            sine_wave = np.maximum(sine_wave, 0)  # Set negative values to zero per sine wave
        
        combined_wave += sine_wave
        logging.debug(f"Added Wave {idx}: Amplitude={amplitude}, Frequency={frequency}, Phase Shift={phase_shift}")
    
    if set_negatives_zero == 'after_sum':
        combined_wave = np.maximum(combined_wave, 0)  # Set negative values to zero after sum
    
    return combined_wave

def plot_data(dates, indices, data_values, combined_wave, extended_dates=None):
    """
    Plot the observed data and the combined sine wave with dates on the x-axis.

    Parameters:
        dates (pd.Series): Series of datetime objects with indices matching 'indices'.
        indices (np.ndarray): Array of indices.
        data_values (np.ndarray): Array of observed data values.
        combined_wave (np.ndarray): Combined sine wave values.
        extended_dates (dict, optional): Extended dates including predictions.
    """
    plt.figure(figsize=(14, 7))
    
    # Plot Observed Data
    plt.plot(dates, data_values, label='Observed Data', color='blue', linestyle='-')
    
    # Plot Combined Sine Waves
    plt.plot(dates, combined_wave[indices - indices[0]], label='Combined Sine Waves', color='red', linestyle='-')
    
    # Plot Predicted Data Before
    if extended_dates and 'before' in extended_dates:
        before_dates = extended_dates['before']['dates']
        before_indices = extended_dates['before']['indices']
        before_wave = combined_wave[before_indices - indices[0]]
        plt.plot(before_dates, before_wave, label='Predicted Before', color='green', linestyle='--')
    
    # Plot Predicted Data After
    if extended_dates and 'after' in extended_dates:
        after_dates = extended_dates['after']['dates']
        after_indices = extended_dates['after']['indices']
        after_wave = combined_wave[after_indices - indices[0]]
        plt.plot(after_dates, after_wave, label='Predicted After', color='orange', linestyle='--')
    
    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Observed Data and Combined Sine Waves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Extrapolator: Combine Sine Waves with Observed Data')
    parser.add_argument('--data-file', type=str, required=True, help='Path to the observed data CSV file')
    parser.add_argument('--project-dir', type=str, required=True, help='Directory containing project data including waves and logs')
    parser.add_argument('--date-col', type=str, default='Timestamp', help='Name of the column containing date information (default: Timestamp)')
    parser.add_argument('--value-col', type=str, default='Value', help='Name of the column containing observed values (default: Value)')
    parser.add_argument('--set-negatives-zero', type=str, choices=['after_sum', 'per_wave', 'none'], default='none',
                        help="How to handle negative sine wave values: 'after_sum', 'per_wave', or 'none' (default)")
    
    # New Arguments for Predictions
    parser.add_argument('--predict-before', type=float, default=5.0,
                        help="Percentage of data points to predict before the observed data (default: 5.0)")
    parser.add_argument('--predict-after', type=float, default=5.0,
                        help="Percentage of data points to predict after the observed data (default: 5.0)")
    
    parser.add_argument('--moving-average', type=int, default=None, help="Apply a moving average filter to smooth the data")
    
    args = parser.parse_args()
    
    # Replace waves-dir and log-dir with project-dir
    project_dir = args.project_dir
    waves_dir = os.path.join(project_dir, "waves")
    log_dir = os.path.join(project_dir, "logs")
    
    # Create project directories if they don't exist
    os.makedirs(waves_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging after setting project directories
    log_filename = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    
    # **Special Command Log Entry**
    command_log_path = os.path.join(log_dir, "command_log.log")
    with open(command_log_path, "a") as cmd_log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        command = 'python3 ' + 'extrapolator.py ' + ' '.join(shlex.quote(arg) for arg in sys.argv[1:])
        cmd_log_file.write(f"{timestamp} {command}\n")
    
    logging.info("Extrapolator is Starting")
    
    # Load Observed Data
    dates, indices, data_values = load_observed_data(args.data_file, date_col=args.date_col, value_col=args.value_col)
    logging.info(f"Loaded Observed Data: {len(indices)} data points.")
    
    # Apply Moving Average if specified
    if args.moving_average and len(data_values) >= args.moving_average:
        data_values = pd.Series(data_values).rolling(window=args.moving_average, min_periods=1).mean().values
        logging.info(f"Applied moving average with window size {args.moving_average}.")
    
    # Calculate Average Timespan Difference
    avg_timespan = calculate_average_timespan(dates)
    
    # Load Sine Waves
    sine_waves = load_sine_waves(waves_dir)
    logging.info(f"Loaded {len(sine_waves)} sine wave(s) from '{waves_dir}'.")
    
    # Calculate number of steps to predict before and after based on percentage
    total_steps = len(data_values)
    predict_before_steps = max(int(total_steps * (args.predict_before / 100)), 1) if args.predict_before > 0 else 0
    predict_after_steps = max(int(total_steps * (args.predict_after / 100)), 1) if args.predict_after > 0 else 0
    
    logging.info(f"Predicting {predict_before_steps} step(s) before the observed data.")
    logging.info(f"Predicting {predict_after_steps} step(s) after the observed data.")
    
    # Generate new indices
    original_start = indices[0] if len(indices) > 0 else 0
    original_end = indices[-1] if len(indices) > 0 else 0
    
    new_before_indices = np.arange(original_start - predict_before_steps, original_start)
    new_after_indices = np.arange(original_end + 1, original_end + 1 + predict_after_steps)
    
    # Combine all indices
    extended_indices = np.concatenate([new_before_indices, indices, new_after_indices])
    
    # Generate Combined Sine Wave for Extended Indices
    combined_wave_extended = generate_combined_sine_wave(sine_waves, extended_indices, set_negatives_zero=args.set_negatives_zero)
    
    # Generate new dates
    if avg_timespan is not None:
        # Generate dates before the first date
        first_date = dates.iloc[0]
        before_dates = [first_date - pd.Timedelta(days=avg_timespan * (predict_before_steps - i)) for i in range(predict_before_steps, 0, -1)]
        
        # Generate dates after the last date
        last_date = dates.iloc[-1]
        after_dates = [last_date + pd.Timedelta(days=avg_timespan * (i + 1)) for i in range(predict_after_steps)]
    else:
        # If avg_timespan is not available, default to 1 day steps
        logging.warning("Average timespan is not available. Using 1 day steps for predictions.")
        first_date = dates.iloc[0]
        last_date = dates.iloc[-1]
        before_dates = [first_date - pd.Timedelta(days=i) for i in range(predict_before_steps, 0, -1)]
        after_dates = [last_date + pd.Timedelta(days=i) for i in range(1, predict_after_steps + 1)]
    
    # Combine dates with extended indices
    combined_dates = pd.Series(before_dates + list(dates) + after_dates, index=extended_indices)
    
    # Adjusted indices for observed data within the extended indices
    observed_start_idx = len(new_before_indices)
    observed_end_idx = observed_start_idx + len(indices)
    
    # Prepare extended_dates dictionary for plotting
    extended_dates_dict = {}
    if predict_before_steps > 0:
        extended_dates_dict['before'] = {
            'indices': new_before_indices,
            'dates': combined_dates.iloc[:observed_start_idx]
        }
    if predict_after_steps > 0:
        extended_dates_dict['after'] = {
            'indices': new_after_indices,
            'dates': combined_dates.iloc[observed_end_idx:]
        }
    
    # Extract dates corresponding to observed data
    observed_dates = combined_dates.iloc[observed_start_idx:observed_end_idx]
    
    # Log Combined Wave Statistics
    logging.info(f"Combined Wave Statistics:\nMin: {combined_wave_extended.min():.2f}, Max: {combined_wave_extended.max():.2f}, Mean: {combined_wave_extended.mean():.2f}")
    
    # Plotting
    plot_data(observed_dates,
              indices,
              data_values,
              combined_wave_extended,
              extended_dates=extended_dates_dict if extended_dates_dict else None)

if __name__ == '__main__':
    import sys  # Needed for command log entry
    main()
