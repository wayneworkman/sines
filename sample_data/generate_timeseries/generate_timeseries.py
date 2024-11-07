import numpy as np
import pandas as pd
import argparse
import random
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_sine_wave(t, amplitude, frequency, phase_shift):
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    return sine_wave

def generate_wave_parameters(waves_dir, num_waves, max_amplitude, max_frequency):
    wave_params_list = []
    
    try:
        if not os.path.exists(waves_dir):
            os.makedirs(waves_dir)
            logger.info(f"Created waves directory at {waves_dir}")
        else:
            logger.info(f"Waves directory already exists at {waves_dir}")
    except Exception as e:
        logger.error(f"Failed to create waves directory at {waves_dir}: {e}")
        raise
    
    for wave_id in range(1, num_waves + 1):
        try:
            amplitude = random.uniform(1, max_amplitude)
            frequency = random.uniform(0.00001, max_frequency)
            phase_shift = random.uniform(0, 2 * np.pi)
            
            wave_params = {
                "amplitude": amplitude,
                "frequency": frequency,
                "phase_shift": phase_shift
            }
            wave_params_list.append(wave_params)
            
            wave_filename = os.path.join(waves_dir, f"wave_{wave_id}.json")
            with open(wave_filename, "w") as f:
                json.dump(wave_params, f, indent=4)
            logger.info(f"Saved wave {wave_id} parameters to {wave_filename}")
        except Exception as e:
            logger.error(f"Failed to generate/save parameters for wave {wave_id}: {e}")
            raise
    
    return wave_params_list

def generate_combined_wave(date_range, wave_params_list, noise_std=1, set_negatives_zero='none'):
    num_points = len(date_range)
    t = np.arange(num_points)
    combined_wave = np.zeros(num_points)
    
    try:
        if set_negatives_zero == 'per_wave':
            for params in wave_params_list:
                wave = create_sine_wave(t, params["amplitude"], params["frequency"], params["phase_shift"])
                wave = np.maximum(wave, 0)  # Set negative values to zero per wave
                combined_wave += wave
        else:
            for params in wave_params_list:
                wave = create_sine_wave(t, params["amplitude"], params["frequency"], params["phase_shift"])
                combined_wave += wave
        
        noise = np.random.normal(0, noise_std, num_points)
        combined_wave += noise
        
        if set_negatives_zero == 'after_sum':
            combined_wave = np.maximum(combined_wave, 0)  # Set negative values to zero after summing
    except Exception as e:
        logger.error(f"Failed to generate combined wave: {e}")
        raise
    
    return combined_wave

def save_run_parameters(params, parameters_dir):
    try:
        if not os.path.exists(parameters_dir):
            os.makedirs(parameters_dir)
            logger.info(f"Created parameters directory at {parameters_dir}")
        else:
            logger.info(f"Parameters directory already exists at {parameters_dir}")
    except Exception as e:
        logger.error(f"Failed to create parameters directory at {parameters_dir}: {e}")
        raise
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parameters_filename = os.path.join(parameters_dir, f"run_parameters_{timestamp}.txt")
        
        with open(parameters_filename, "w") as f:
            json.dump(params, f, indent=4, default=str)
        
        logger.info(f"Saved run parameters to {parameters_filename}")
    except Exception as e:
        logger.error(f"Failed to save run parameters to {parameters_filename}: {e}")
        raise

def split_and_save_data(combined_df, training_range, testing_range, training_output, testing_output):
    try:
        training_start, training_end = training_range
        testing_start, testing_end = testing_range
        
        training_df = combined_df[(combined_df['date'] >= training_start) & (combined_df['date'] <= training_end)].copy()
        testing_df = combined_df[(combined_df['date'] >= testing_start) & (combined_df['date'] <= testing_end)].copy()
        
        training_df.to_csv(training_output, index=False)
        logger.info(f"Training data saved to {training_output}")
        
        testing_df.to_csv(testing_output, index=False)
        logger.info(f"Testing data saved to {testing_output}")
    except Exception as e:
        logger.error(f"Failed to split and save data: {e}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Training and Testing Time Series Data with Individual Wave Parameters")
    
    # Output files
    parser.add_argument('--training-output-file', type=str, default="training_data.csv", help="Path to save the generated training data CSV")
    parser.add_argument('--testing-output-file', type=str, default="testing_data.csv", help="Path to save the generated testing data CSV")
    
    # Date ranges
    parser.add_argument('--training-start-date', type=str, default="2000-01-01", help="Start date for the training data in YYYY-MM-DD format")
    parser.add_argument('--training-end-date', type=str, default="2005-01-01", help="End date for the training data in YYYY-MM-DD format")
    parser.add_argument('--testing-start-date', type=str, default="1990-01-01", help="Start date for the testing data in YYYY-MM-DD format")
    parser.add_argument('--testing-end-date', type=str, default="2015-01-01", help="End date for the testing data in YYYY-MM-DD format")
    
    # Wave generation parameters
    parser.add_argument('--waves-dir', type=str, default="waves", help="Directory to store individual wave parameters")
    parser.add_argument('--num-waves', type=int, default=5, help="Number of sine waves to combine in the generated data")
    parser.add_argument('--max-amplitude', type=float, default=150, help="Maximum amplitude for the generated sine waves")
    parser.add_argument('--max-frequency', type=float, default=0.001, help="Maximum frequency for the generated sine waves")
    parser.add_argument('--noise-std', type=float, default=1, help="Standard deviation of noise added to the sine waves")
    
    # Updated Argument: Handling Negative Values
    parser.add_argument('--set-negatives-zero', type=str, choices=['after_sum', 'per_wave', 'none'], default='none',
                        help="How to handle negative sine wave values: 'after_sum', 'per_wave', or 'none' (default)")
    
    # Parameters logging
    parser.add_argument('--parameters-dir', type=str, default="run_parameters", help="Directory to store run parameters")
    
    args = parser.parse_args()
    
    # Additional Validation
    try:
        training_start = pd.to_datetime(args.training_start_date)
        training_end = pd.to_datetime(args.training_end_date)
        testing_start = pd.to_datetime(args.testing_start_date)
        testing_end = pd.to_datetime(args.testing_end_date)
        
        if training_start > training_end:
            parser.error("Training start date must be earlier than or equal to training end date.")
        if testing_start > testing_end:
            parser.error("Testing start date must be earlier than or equal to testing end date.")
    except Exception as e:
        parser.error(f"Date parsing error: {e}")
    
    if args.num_waves < 0:
        parser.error("Number of waves (--num-waves) must be non-negative.")
    if args.max_amplitude <= 0:
        parser.error("Maximum amplitude (--max-amplitude) must be positive.")
    if args.max_frequency <= 0:
        parser.error("Maximum frequency (--max-frequency) must be positive.")
    if args.noise_std < 0:
        parser.error("Noise standard deviation (--noise-std) must be non-negative.")
    
    return args

def main():
    args = parse_arguments()
    
    run_parameters = vars(args)
    save_run_parameters(run_parameters, args.parameters_dir)
    
    training_start = pd.to_datetime(args.training_start_date)
    training_end = pd.to_datetime(args.training_end_date)
    testing_start = pd.to_datetime(args.testing_start_date)
    testing_end = pd.to_datetime(args.testing_end_date)
    
    overall_start_date = min(training_start, testing_start)
    overall_end_date = max(training_end, testing_end)
    
    logger.info(f"\nOverall date range for wave generation: {overall_start_date.date()} to {overall_end_date.date()}")
    
    try:
        wave_params_list = generate_wave_parameters(
            waves_dir=args.waves_dir,
            num_waves=args.num_waves,
            max_amplitude=args.max_amplitude,
            max_frequency=args.max_frequency
        )
    except Exception as e:
        logger.error(f"Error generating wave parameters: {e}")
        return
    
    overall_date_range = pd.date_range(start=overall_start_date, end=overall_end_date, freq='D')
    
    try:
        combined_wave = generate_combined_wave(
            date_range=overall_date_range,
            wave_params_list=wave_params_list,
            noise_std=args.noise_std,
            set_negatives_zero=args.set_negatives_zero
        )
    except Exception as e:
        logger.error(f"Error generating combined wave: {e}")
        return
    
    combined_df = pd.DataFrame({
        'date': overall_date_range,
        'value': combined_wave
    })
    
    try:
        split_and_save_data(
            combined_df=combined_df,
            training_range=(args.training_start_date, args.training_end_date),
            testing_range=(args.testing_start_date, args.testing_end_date),
            training_output=args.training_output_file,
            testing_output=args.testing_output_file
        )
    except Exception as e:
        logger.error(f"Error splitting and saving data: {e}")
        return

if __name__ == "__main__":
    main()
