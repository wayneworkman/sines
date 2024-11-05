import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_observed_data(file_path, value_col='Value'):
    """
    Load the observed data from a CSV file and assign sequential indices.

    Parameters:
        file_path (str): Path to the CSV data file.
        value_col (str): Name of the column containing the observed values.

    Returns:
        indices (np.ndarray): Array of indices.
        data_values (np.ndarray): Array of observed data values.
    """
    try:
        df = pd.read_csv(file_path)
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found in the data.")
        data_values = df[value_col].values
        indices = np.arange(len(data_values))
        return indices, data_values
    except FileNotFoundError:
        print(f"Error: Data file '{file_path}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

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
        print(f"Error: Waves directory '{waves_dir}' does not exist.")
        exit(1)
    
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
                print(f"Warning: Wave file '{filename}' is not a valid JSON. Skipping.")
            except Exception as e:
                print(f"Warning: Could not load wave file '{filename}': {e}. Skipping.")
    if not sine_waves:
        print(f"Error: No valid sine wave JSON files found in '{waves_dir}'.")
        exit(1)
    return sine_waves

def generate_combined_sine_wave(sine_waves, indices, set_negatives_zero=False):
    """
    Generate and combine multiple sine waves based on their parameters.

    Parameters:
        sine_waves (list): List of dictionaries with sine wave parameters.
        indices (np.ndarray): Array of indices.
        set_negatives_zero (bool): If True, set negative sine values to zero.

    Returns:
        combined_wave (np.ndarray): Combined sine wave values.
    """
    combined_wave = np.zeros_like(indices, dtype=np.float64)
    for idx, wave in enumerate(sine_waves, start=1):
        amplitude = wave['amplitude']
        frequency = wave['frequency']
        phase_shift = wave['phase_shift']
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * indices + phase_shift)
        combined_wave += sine_wave
        print(f"Added Wave {idx}: Amplitude={amplitude}, Frequency={frequency}, Phase Shift={phase_shift}")
    
    if set_negatives_zero:
        combined_wave = np.maximum(combined_wave, 0)
    
    return combined_wave

def plot_data(indices, data_values, combined_wave):
    """
    Plot the observed data and the combined sine wave.

    Parameters:
        indices (np.ndarray): Array of indices.
        data_values (np.ndarray): Array of observed data values.
        combined_wave (np.ndarray): Combined sine wave values.
    """
    plt.figure(figsize=(14, 7))
    
    # Plot Observed Data
    plt.plot(indices, data_values, label='Observed Data', color='blue', marker='o', linestyle='-', markersize=4)
    
    # Plot Combined Sine Waves
    plt.plot(indices, combined_wave, label='Combined Sine Waves', color='red', linestyle='--')
    
    plt.title('Observed Data and Combined Sine Waves')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Extrapolator: Combine Sine Waves with Observed Data')
    parser.add_argument('--data-file', type=str, required=True, help='Path to the observed data CSV file')
    parser.add_argument('--waves-dir', type=str, default='waves', help='Directory containing sine wave JSON files (default: waves)')
    parser.add_argument('--value-col', type=str, default='Value', help='Name of the column containing observed values (default: Value)')
    parser.add_argument('--set-negatives-zero', action='store_true', help='Set negative sine wave values to zero')
    
    args = parser.parse_args()
    
    # Load Observed Data
    indices, data_values = load_observed_data(args.data_file, value_col=args.value_col)
    print(f"Loaded Observed Data: {len(indices)} data points.")
    
    # Load Sine Waves
    sine_waves = load_sine_waves(args.waves_dir)
    print(f"Loaded {len(sine_waves)} sine wave(s) from '{args.waves_dir}'.")
    
    # Generate Combined Sine Wave
    combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero=args.set_negatives_zero)
    print(f"Combined Wave Statistics:\nMin: {combined_wave.min():.2f}, Max: {combined_wave.max():.2f}, Mean: {combined_wave.mean():.2f}")
    
    # Plotting
    plot_data(indices, data_values, combined_wave)

if __name__ == '__main__':
    main()
