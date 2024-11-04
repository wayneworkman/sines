import numpy as np
import pandas as pd
import argparse
import random
import os
import json

def create_sine_wave(start_date, end_date, amplitude, frequency, phase_shift, noise_std=1, set_negatives_zero=False):
    """
    Create a sine wave with optional noise and conditional zeroing of negative values.

    Parameters:
    - start_date (str): Start date for the sine wave.
    - end_date (str): End date for the sine wave.
    - amplitude (float): Amplitude of the sine wave.
    - frequency (float): Frequency of the sine wave.
    - phase_shift (float): Phase shift of the sine wave.
    - noise_std (float): Standard deviation of the Gaussian noise to add.
    - set_negatives_zero (bool): If True, sets sine wave values below zero to zero.

    Returns:
    - date_range (pd.DatetimeIndex): Date range for the sine wave.
    - sine_wave (np.ndarray): Generated sine wave values.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_points = len(date_range)
    t = np.arange(num_points)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    noise = np.random.normal(0, noise_std, num_points)  # Adding some noise
    sine_wave += noise  # Apply noise to sine wave
    if set_negatives_zero:
        sine_wave = np.maximum(sine_wave, 0)  # Ensure no negative values
    return date_range, sine_wave

def generate_sample_data(output_file, waves_dir, start_date, end_date, num_waves, max_amplitude, max_frequency, noise_std=1, set_negatives_zero=False):
    """
    Generate sample time series data by combining multiple sine waves.

    Parameters:
    - output_file (str): Path to save the generated sample data CSV.
    - waves_dir (str): Directory to store individual wave parameters.
    - start_date (str): Start date for the generated data in YYYY-MM-DD format.
    - end_date (str): End date for the generated data in YYYY-MM-DD format.
    - num_waves (int): Number of sine waves to combine.
    - max_amplitude (float): Maximum amplitude for the generated sine waves.
    - max_frequency (float): Maximum frequency for the generated sine waves.
    - noise_std (float): Standard deviation of noise added to the sine waves.
    - set_negatives_zero (bool): If True, sets sine wave values below zero to zero.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_points = len(date_range)
    combined_wave = np.zeros(num_points)

    # Ensure the waves directory exists
    if not os.path.exists(waves_dir):
        os.makedirs(waves_dir)
        print(f"Created waves directory at {waves_dir}")

    for wave_id in range(1, num_waves + 1):
        amplitude = random.uniform(1, max_amplitude)
        frequency = random.uniform(0.00001, max_frequency)
        phase_shift = random.uniform(0, 2 * np.pi)
        _, wave = create_sine_wave(
            start_date, end_date, amplitude, frequency, phase_shift, noise_std, set_negatives_zero
        )
        combined_wave += wave

        # Save individual sine wave parameters to the waves directory
        wave_params = {
            "amplitude": amplitude,
            "frequency": frequency,
            "phase_shift": phase_shift
        }
        wave_filename = os.path.join(waves_dir, f"wave_{wave_id}.json")
        with open(wave_filename, "w") as f:
            json.dump(wave_params, f, indent=4)
        print(f"Saved wave {wave_id} parameters to {wave_filename}")

    # Create a DataFrame with date and combined wave data
    data = pd.DataFrame({
        'date': date_range,
        'value': combined_wave
    })

    # Save to CSV
    data.to_csv(output_file, index=False)
    print(f"\nSample data with {num_waves} waves saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sample Time Series Data with Individual Wave Parameters")
    parser.add_argument('--output_file', type=str, default="sample_data.csv", help="Path to save the generated sample data")
    parser.add_argument('--waves_dir', type=str, default="waves", help="Directory to store individual wave parameters")
    parser.add_argument('--start_date', type=str, default="1818-01-01", help="Start date for the generated data in YYYY-MM-DD format")
    parser.add_argument('--end_date', type=str, default="2021-01-01", help="End date for the generated data in YYYY-MM-DD format")
    parser.add_argument('--num_waves', type=int, default=5, help="Number of sine waves to combine in the generated data")
    parser.add_argument('--max_amplitude', type=float, default=150, help="Maximum amplitude for the generated sine waves")
    parser.add_argument('--max_frequency', type=float, default=0.001, help="Maximum frequency for the generated sine waves")
    parser.add_argument('--noise_std', type=float, default=1, help="Standard deviation of noise added to the sine waves")
    parser.add_argument('--set-negatives-zero', action='store_true', help="Set sine wave values below zero to zero")
    
    args = parser.parse_args()

    generate_sample_data(
        output_file=args.output_file,
        waves_dir=args.waves_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        num_waves=args.num_waves,
        max_amplitude=args.max_amplitude,
        max_frequency=args.max_frequency,
        noise_std=args.noise_std,
        set_negatives_zero=args.set_negatives_zero
    )
