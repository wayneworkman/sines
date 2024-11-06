"""
Running multi_sines basically requires a super computer.
It is designed to search for multiple sine waves concurrently, because the interplay between multiple sine waves being 
discovered simultaniously will lead to better fitting models.
I don't know if this script actually works or not because I've never waited for it to finish processing.
It does seem to chug along without error, though. This is provided free of charge. :) 
"""


import pyopencl as cl
import numpy as np
import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from itertools import product
import psutil
import gc  # Import garbage collector

# Constants
LOCAL_WORK_SIZE = 256  # Work group size for OpenCL, must be compatible with GPU

# Global STEP_SIZES initialized as empty
STEP_SIZES = {}

# -------------------- Refinement Step Sizes Configuration -------------------- #

REFINEMENT_STEP_SIZES_BASE = {
    'ultrafine': {
        'amplitude_step_ratio': 0.01,
        'frequency_step': 0.00000005,
        'phase_shift_step': 0.0005
    },
    'fine': {
        'amplitude_step_ratio': 0.02,
        'frequency_step': 0.0000001,
        'phase_shift_step': 0.001
    },
    'normal': {
        'amplitude_step_ratio': 0.05,
        'frequency_step': 0.0000005,
        'phase_shift_step': 0.002
    },
    'fast': {
        'amplitude_step_ratio': 0.1,
        'frequency_step': 0.000001,
        'phase_shift_step': 0.005
    }
}

# -------------------- OpenCL Kernel Definition -------------------- #

KERNEL_CODE = """
__kernel void calculate_fitness(
    __global const float *observed,
    __global const float *combined,
    __global const float *amplitudes,
    __global const float *frequencies,
    __global const float *phase_shifts,
    __global float *scores,
    int num_points,
    int num_waves,
    int zero_mode  // 0: no zeroing, 1: per_wave, 2: after_sum
) {
    int gid = get_global_id(0);
    float score = 0.0f;

    for (int i = 0; i < num_points; i++) {
        float sum = 0.0f;
        for (int j = 0; j < num_waves; j++) {
            int idx = gid * num_waves + j;
            float sine_value = amplitudes[idx] * sin(2.0f * 3.141592653589793f * frequencies[idx] * i + phase_shifts[idx]);

            if (zero_mode == 1 && sine_value < 0) {
                sine_value = 0;  // per_wave zeroing
            }

            sum += sine_value;
        }

        float combined_value = combined[i] + sum;

        if (zero_mode == 2 && combined_value < 0) {
            combined_value = 0;  // after_sum zeroing
        }

        float diff = observed[i] - combined_value;
        score += fabs(diff);
    }
    scores[gid] = score;
}
"""

# -------------------- Helper Functions -------------------- #

def get_available_memory():
    """Returns available memory in bytes."""
    mem = psutil.virtual_memory()
    # Reserve some memory (e.g., 10%) to prevent system from running out
    available = mem.available * 0.9
    return available

def calculate_chunk_size(num_waves, bytes_per_combination=28, desired_memory_MB=10):
    """
    Calculate the chunk size based on desired memory usage and number of waves.
    bytes_per_combination = 3 floats per wave * 4 bytes + 4 bytes for score
    """
    available_bytes = get_available_memory()
    desired_memory_bytes = desired_memory_MB * 1024 * 1024
    # Ensure we don't exceed available memory
    desired_memory_bytes = min(desired_memory_bytes, available_bytes)
    chunk_size = desired_memory_bytes // bytes_per_combination
    # Avoid extremely large or small chunk sizes
    chunk_size = max(1000, min(chunk_size, 1000000))
    return chunk_size

# -------------------- Refinement Phase Implementation -------------------- #

def refine_candidates(top_candidates, observed_data, combined_wave, context, queue, ax, wave_count,
                     desired_refinement_step_size='fast', set_negatives_zero=False, max_observed=1.0,
                     num_waves=1, top_n=3):
    logging.info(f"Wave(s) {', '.join(map(str, range(1, num_waves+1)))}: Starting refinement phase.")

    refined_best_score = np.inf
    refined_best_params = None

    # Determine zero_mode based on set_negatives_zero
    if set_negatives_zero == 'per_wave':
        zero_mode = 1
    elif set_negatives_zero == 'after_sum':
        zero_mode = 2
    else:
        zero_mode = 0

    amplitude_step = REFINEMENT_STEP_SIZES_BASE[desired_refinement_step_size]["amplitude_step_ratio"] * max_observed
    frequency_step = REFINEMENT_STEP_SIZES_BASE[desired_refinement_step_size]["frequency_step"]
    phase_shift_step = REFINEMENT_STEP_SIZES_BASE[desired_refinement_step_size]["phase_shift_step"]

    program = cl.Program(context, KERNEL_CODE).build()
    mf = cl.mem_flags
    observed_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=observed_data)
    combined_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=combined_wave)

    new_top_candidates = []

    for candidate_idx, candidate in enumerate(top_candidates):
        # Each candidate has 'waves': list of wave parameters
        param_grids = []
        for wave_idx in range(num_waves):
            params = candidate['waves'][wave_idx]
            amplitude_min = max(params["amplitude"] - 0.5 * max_observed, 0.1)
            amplitude_max = params["amplitude"] + 0.5 * max_observed
            amplitude_range = np.arange(amplitude_min, amplitude_max, amplitude_step)

            frequency_min = max(params["frequency"] - 0.000005, 0.000001)
            frequency_max = min(params["frequency"] + 0.000005, 0.01)
            frequency_range = np.arange(frequency_min, frequency_max, frequency_step)

            phase_shift_min = (params["phase_shift"] - 0.1) % (2 * np.pi)
            phase_shift_max = (params["phase_shift"] + 0.1) % (2 * np.pi)
            if phase_shift_min < phase_shift_max:
                phase_shift_range = np.arange(phase_shift_min, phase_shift_max, phase_shift_step)
            else:
                phase_shift_range = np.concatenate((
                    np.arange(phase_shift_min, 2 * np.pi, phase_shift_step),
                    np.arange(0, phase_shift_max, phase_shift_step)
                ))

            # Validate that ranges are not empty
            if len(amplitude_range) == 0 or len(frequency_range) == 0 or len(phase_shift_range) == 0:
                logging.warning(f"Empty parameter range detected for candidate {candidate_idx + 1}, wave {wave_idx + 1}. Skipping this candidate.")
                param_grids = []
                break

            param_grids.append((amplitude_range, frequency_range, phase_shift_range))

        if not param_grids:
            continue  # Skip candidates with invalid parameter ranges

        # Generate all combinations for N waves
        parameter_combinations = product(*[product(*grid) for grid in param_grids])

        for combo in parameter_combinations:
            amplitudes = []
            frequencies = []
            phase_shifts = []

            for wave_params in combo:
                # Ensure wave_params is a tuple or list with 3 elements
                if not isinstance(wave_params, (tuple, list)) or len(wave_params) != 3:
                    logging.error(f"Invalid wave_params structure: {wave_params}. Expected a tuple/list with 3 elements.")
                    continue
                amplitudes.append(wave_params[0])
                frequencies.append(wave_params[1])
                phase_shifts.append(wave_params[2])

            amplitudes_np = np.array(amplitudes, dtype=np.float32)
            frequencies_np = np.array(frequencies, dtype=np.float32)
            phase_shifts_np = np.array(phase_shifts, dtype=np.float32)
            scores_np = np.empty((1,), dtype=np.float32)

            amplitudes_buf_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=amplitudes_np)
            frequencies_buf_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frequencies_np)
            phase_shifts_buf_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phase_shifts_np)
            scores_buf_cl = cl.Buffer(context, mf.WRITE_ONLY, size=scores_np.nbytes)

            kernel = program.calculate_fitness
            kernel.set_args(
                observed_buf, combined_buf, amplitudes_buf_cl, frequencies_buf_cl,
                phase_shifts_buf_cl, scores_buf_cl, np.int32(len(observed_data)),
                np.int32(num_waves), np.int32(zero_mode)
            )

            global_work_size = (1,)
            local_work_size_final = None  # Let OpenCL decide

            try:
                cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size_final)
                queue.finish()
            except cl.RuntimeError as e:
                logging.error(f"OpenCL kernel execution failed: {e}")
                continue

            try:
                cl.enqueue_copy(queue, scores_np, scores_buf_cl)
                queue.finish()
            except cl.RuntimeError as e:
                logging.error(f"OpenCL buffer copy failed: {e}")
                continue

            score = scores_np[0]
            candidate_params = []
            for w in range(num_waves):
                candidate_params.append({
                    "amplitude": float(amplitudes_np[w]),
                    "frequency": float(frequencies_np[w]),
                    "phase_shift": float(phase_shifts_np[w])
                })

            new_top_candidates.append({"waves": candidate_params, "score": score})

            if score < refined_best_score:
                refined_best_score = score
                refined_best_params = candidate_params

            # Maintain only top_n candidates
            if len(new_top_candidates) > top_n:
                new_top_candidates = sorted(new_top_candidates, key=lambda x: x["score"])[:top_n]

            # Optional plotting for refinement
            if ax is not None and refined_best_params:
                combined_temp = combined_wave.copy()
                for params in refined_best_params:
                    sine_wave = generate_sine_wave(params, len(observed_data),
                                                  set_negatives_zero=(set_negatives_zero == 'per_wave'))
                    combined_temp += sine_wave
                if set_negatives_zero == 'after_sum':
                    combined_temp = np.maximum(combined_temp, 0)
                ax.clear()
                ax.plot(observed_data, label="Observed Data", color="blue")
                ax.plot(combined_temp, label="Combined Sine Waves", color="orange")
                wave_labels = ", ".join([f"W{w+1}" for w in range(num_waves)])
                ax.set_title(f"Refinement Progress - Waves {wave_labels}")
                ax.legend()
                plt.pause(0.01)

        # -------------------- Chunk Cleanup -------------------- #
        # Explicitly delete buffers to free GPU memory
        del observed_buf
        del combined_buf
        del amplitudes_buf_cl
        del frequencies_buf_cl
        del phase_shifts_buf_cl
        del scores_buf_cl
        del amplitudes_np
        del frequencies_np
        del phase_shifts_np
        del scores_np
        gc.collect()  # Force garbage collection

        return new_top_candidates

# -------------------- Brute-Force Search Implementation -------------------- #

def brute_force_sine_wave_search(observed_data, combined_wave, context, queue, ax, wave_count,
                                 desired_step_size='fast', set_negatives_zero=False, max_observed=1.0,
                                 top_n=3, num_waves=1):
    amplitude_range = STEP_SIZES[desired_step_size]["amplitude"]
    frequency_range = STEP_SIZES[desired_step_size]["frequency"]
    phase_shift_range = STEP_SIZES[desired_step_size]["phase_shift"]

    param_grids = []
    for _ in range(num_waves):
        param_grids.append((amplitude_range, frequency_range, phase_shift_range))

    # Estimate total combinations
    total_combinations_estimate = 1
    for grid in param_grids:
        total_combinations_estimate *= len(grid[0]) * len(grid[1]) * len(grid[2])
    logging.info(f"Wave(s) {', '.join(map(str, range(1, num_waves+1)))}: Starting brute-force search with approximately {total_combinations_estimate} combinations.")

    # Determine zero_mode based on set_negatives_zero
    if set_negatives_zero == 'per_wave':
        zero_mode = 1
    elif set_negatives_zero == 'after_sum':
        zero_mode = 2
    else:
        zero_mode = 0

    program = cl.Program(context, KERNEL_CODE).build()
    mf = cl.mem_flags
    observed_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=observed_data)
    combined_buf_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=combined_wave)

    top_candidates = []
    best_score_so_far = np.inf
    processed_combinations = 0

    # Generate all combinations as an iterator
    parameter_combinations = product(*[product(*grid) for grid in param_grids])

    # Calculate dynamic chunk size
    chunk_size = calculate_chunk_size(num_waves)
    logging.info(f"Dynamic chunk size set to: {chunk_size}")

    for chunk_idx, chunk in enumerate(chunked(parameter_combinations, chunk_size)):
        current_chunk_size = len(chunk)
        amplitudes = []
        frequencies = []
        phase_shifts = []

        for combo in chunk:
            for wave_params in combo:
                # Ensure wave_params is a tuple or list with 3 elements
                if not isinstance(wave_params, (tuple, list)) or len(wave_params) != 3:
                    logging.error(f"Invalid wave_params structure: {wave_params}. Expected a tuple/list with 3 elements.")
                    continue
                amplitudes.append(wave_params[0])
                frequencies.append(wave_params[1])
                phase_shifts.append(wave_params[2])

        if not amplitudes:
            logging.warning(f"Wave(s) {', '.join(map(str, range(1, num_waves+1)))}: No valid combinations in this chunk.")
            continue

        amplitudes_np = np.array(amplitudes, dtype=np.float32)
        frequencies_np = np.array(frequencies, dtype=np.float32)
        phase_shifts_np = np.array(phase_shifts, dtype=np.float32)
        scores_np = np.empty((current_chunk_size,), dtype=np.float32)

        amplitudes_buf_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=amplitudes_np)
        frequencies_buf_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frequencies_np)
        phase_shifts_buf_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phase_shifts_np)
        scores_buf_cl = cl.Buffer(context, mf.WRITE_ONLY, size=scores_np.nbytes)

        kernel = program.calculate_fitness
        kernel.set_args(
            observed_buf, combined_buf_cl, amplitudes_buf_cl, frequencies_buf_cl,
            phase_shifts_buf_cl, scores_buf_cl, np.int32(len(observed_data)),
            np.int32(num_waves), np.int32(zero_mode)
        )

        global_work_size = (current_chunk_size,)
        local_work_size_final = None  # Let OpenCL decide

        try:
            cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size_final)
            queue.finish()
        except cl.RuntimeError as e:
            logging.error(f"OpenCL kernel execution failed: {e}")
            continue

        try:
            cl.enqueue_copy(queue, scores_np, scores_buf_cl)
            queue.finish()
        except cl.RuntimeError as e:
            logging.error(f"OpenCL buffer copy failed: {e}")
            continue

        for i in range(current_chunk_size):
            score = scores_np[i]
            params = []
            for w in range(num_waves):
                idx = i * num_waves + w
                if idx >= len(amplitudes_np):
                    logging.error(f"Index out of bounds: {idx} for amplitudes array of length {len(amplitudes_np)}.")
                    continue
                params.append({
                    "amplitude": float(amplitudes_np[idx]),
                    "frequency": float(frequencies_np[idx]),
                    "phase_shift": float(phase_shifts_np[idx])
                })
            if len(params) != num_waves:
                logging.error(f"Incorrect number of parameters for wave(s): expected {num_waves}, got {len(params)}.")
                continue
            candidate = {"waves": params, "score": score}
            top_candidates.append(candidate)

            if score < best_score_so_far:
                best_score_so_far = score
                best_candidate = candidate

            # Maintain only top_n candidates
            if len(top_candidates) > top_n:
                top_candidates = sorted(top_candidates, key=lambda x: x["score"])[:top_n]

        processed_combinations += current_chunk_size
        logging.info(f"Wave(s) {', '.join(map(str, range(1, num_waves+1)))}: Processed {processed_combinations} combinations.")

        if ax is not None and 'best_candidate' in locals():
            combined_temp = combined_wave.copy()
            for params in best_candidate['waves']:
                sine_wave = generate_sine_wave(params, len(observed_data),
                                              set_negatives_zero=(set_negatives_zero == 'per_wave'))
                combined_temp += sine_wave
            if set_negatives_zero == 'after_sum':
                combined_temp = np.maximum(combined_temp, 0)
            ax.clear()
            ax.plot(observed_data, label="Observed Data", color="blue")
            ax.plot(combined_temp, label="Combined Sine Waves", color="orange")
            wave_labels = ", ".join([f"W{w+1}" for w in range(num_waves)])
            ax.set_title(f"Real-Time Fitting Progress - Waves {wave_labels}")
            ax.legend()
            plt.pause(0.01)

        # Log progress every 10 chunks
        if chunk_idx % 10 == 0 and chunk_idx != 0:
            progress = (processed_combinations / total_combinations_estimate) * 100
            logging.info(f"Wave(s) {', '.join(map(str, range(1, num_waves+1)))}: Progress: {progress:.2f}%, Best Score So Far: {best_score_so_far}")

        # -------------------- Chunk Cleanup -------------------- #
        # Explicitly delete buffers to free GPU memory
        del amplitudes_buf_cl
        del frequencies_buf_cl
        del phase_shifts_buf_cl
        del scores_buf_cl
        del amplitudes_np
        del frequencies_np
        del phase_shifts_np
        del scores_np
        gc.collect()  # Force garbage collection

    logging.info(f"Wave(s) {', '.join(map(str, range(1, num_waves+1)))}: Completed brute-force search with best score: {best_score_so_far}")
    return top_candidates

# -------------------- Chunking Helper Function -------------------- #

def chunked(iterable, chunk_size):
    """Yield successive chunk_size-sized chunks from iterable."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

# -------------------- Setup OpenCL -------------------- #

def setup_opencl():
    platforms = cl.get_platforms()
    nvidia_platform = next((p for p in platforms if 'NVIDIA' in p.name), None)
    if not nvidia_platform:
        raise RuntimeError("NVIDIA platform not found.")
    devices = nvidia_platform.get_devices(device_type=cl.device_type.GPU)
    if not devices:
        raise RuntimeError("No GPU devices found on NVIDIA platform.")

    device = devices[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    logging.info(f"Using device: {device.name}")
    logging.info(f"Max work group size: {device.max_work_group_size}")

    return context, queue

# -------------------- Data Loading -------------------- #

def load_data(file_path, date_col="date", value_col="value", moving_average=None):
    try:
        with open(file_path, 'r') as f:
            df = pd.read_json(f)
        df[date_col] = pd.to_datetime(df[date_col])
        logging.info(f"Loaded data from JSON file: {file_path}")
    except (ValueError, json.JSONDecodeError):
        df = pd.read_csv(file_path, parse_dates=[date_col])
        logging.info(f"Loaded data from CSV file: {file_path}")

    df = df.sort_values(by=date_col)

    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in data.")

    if moving_average:
        df[value_col] = df[value_col].rolling(window=moving_average, min_periods=1).mean()

    # Log information about the loaded data file
    num_data_points = len(df)
    max_value = df[value_col].max()
    min_value = df[value_col].min()
    mean_value = df[value_col].mean()
    start_date = df[date_col].min()
    end_date = df[date_col].max()

    logging.info("Loaded data statistics:")
    logging.info(f"  Number of data points: {num_data_points}")
    logging.info(f"  Max value: {max_value}")
    logging.info(f"  Min value: {min_value}")
    logging.info(f"  Mean value: {mean_value}")
    logging.info(f"  Start date: {start_date}")
    logging.info(f"  End date: {end_date}")

    return df[value_col].values.astype(np.float32)

# -------------------- Sine Wave Generation -------------------- #

def generate_sine_wave(params, num_points, set_negatives_zero=False):
    amplitude, frequency, phase_shift = params["amplitude"], params["frequency"], params["phase_shift"]
    t = np.arange(num_points, dtype=np.float32)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    if set_negatives_zero:
        np.maximum(sine_wave, 0, out=sine_wave)
    return sine_wave

# -------------------- Load Previous Waves -------------------- #

def load_previous_waves(num_points, output_dir, set_negatives_zero=False):
    """
    Load all previously saved waves and compute the true cumulative sum without applying zeroing.
    Zeroing (if required) is handled only during fitness evaluation.
    """
    combined_wave = np.zeros(num_points, dtype=np.float32)
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(output_dir, filename), "r") as f:
                    wave_params = json.load(f)
                    if isinstance(wave_params, list):
                        for params in wave_params:
                            sine_wave = generate_sine_wave(params, num_points, set_negatives_zero)
                            combined_wave += sine_wave
                    else:
                        sine_wave = generate_sine_wave(wave_params, num_points, set_negatives_zero)
                        combined_wave += sine_wave
            except json.JSONDecodeError:
                logging.warning(f"Error loading wave file {filename}. Skipping corrupted file.")
    return combined_wave

# -------------------- Main Function -------------------- #

def main():
    parser = argparse.ArgumentParser(description="Time Series Modeling with Sine Waves")
    parser.add_argument('--data-file', type=str, required=True, help="Path to the data file")
    parser.add_argument('--date-col', type=str, default="date", help="Name of the date column in the data")
    parser.add_argument('--value-col', type=str, default="value", help="Name of the value column in the data")
    parser.add_argument('--moving-average', type=int, help="Optional moving average window for smoothing the data")
    parser.add_argument('--waves-dir', type=str, default="waves", help="Directory to store generated wave parameters")
    parser.add_argument('--log-dir', type=str, default="logs", help="Directory to store log files")
    parser.add_argument('--desired-step-size', type=str, choices=['fine', 'normal', 'fast'], default="fast",
                        help="Desired step size mode for brute-force search phase")
    parser.add_argument('--desired-refinement-step-size', type=str, choices=['fine', 'normal', 'fast', 'skip'], default="skip",
                        help="Desired step size mode for refinement phase. Use 'skip' to skip refinement.")
    parser.add_argument('--no-plot', action='store_true', help="Disable real-time plotting")
    parser.add_argument('--wave-count', type=int, default=50, help="Number of waves to generate before exiting. Use 0 for infinite.")
    parser.add_argument('--top-candidates', type=int, default=3, help="Number of top candidates to keep. Default is 3.")
    parser.add_argument('--num-waves', type=int, default=1, help="Number of sine waves to discover at a time. Default is 1.")
    parser.add_argument('--progressive-step-sizes', action='store_true', default=True,
                        help="Dynamically choose step size based on observed and combined wave differences")
    parser.add_argument('--set-negatives-zero', type=str, choices=['after_sum', 'per_wave', 'none'], default='none',
                        help="How to handle negative sine wave values: 'after_sum', 'per_wave', or 'none' (default)")
    parser.add_argument('--desired-memory-mb', type=int, default=10, help="Desired memory per chunk in MB. Lower values reduce memory usage.")

    args = parser.parse_args()

    # Setup logging after parsing arguments
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_filename = os.path.join(args.log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

    logging.info("Sines is Starting")

    context, queue = setup_opencl()
    observed_data = load_data(
        args.data_file,
        date_col=args.date_col,
        value_col=args.value_col,
        moving_average=args.moving_average
    )

    if not os.path.exists(args.waves_dir):
        os.makedirs(args.waves_dir)

    # Use maximum absolute value instead of maximum positive value
    max_observed = np.max(np.abs(observed_data)) if len(observed_data) > 0 else 1.0
    scaling_factor = 1.5
    amplitude_upper_limit = max_observed * scaling_factor

    # Dynamically define STEP_SIZES based on max_observed
    global STEP_SIZES
    STEP_SIZES = {
        'ultrafine': {
            'amplitude': np.arange(0.1, amplitude_upper_limit, 0.01 * max_observed),
            'frequency': np.arange(0.00001, 0.001, 0.0000075),
            'phase_shift': np.arange(0, 2 * np.pi, 0.025)
        },
        'fine': {
            'amplitude': np.arange(0.1, amplitude_upper_limit, 0.02 * max_observed),
            'frequency': np.arange(0.00001, 0.001, 0.000015),
            'phase_shift': np.arange(0, 2 * np.pi, 0.05)
        },
        'normal': {
            'amplitude': np.arange(0.1, amplitude_upper_limit, 0.05 * max_observed),
            'frequency': np.arange(0.00001, 0.001, 0.00003),
            'phase_shift': np.arange(0, 2 * np.pi, 0.15)
        },
        'fast': {
            'amplitude': np.arange(0.1, amplitude_upper_limit, 0.1 * max_observed),
            'frequency': np.arange(0.00001, 0.001, 0.00006),
            'phase_shift': np.arange(0, 2 * np.pi, 0.3)
        }
    }

    # Validate that STEP_SIZES are not empty
    for step_size, grids in STEP_SIZES.items():
        for param, values in grids.items():
            if len(values) == 0:
                logging.warning(f"Step size '{step_size}' has an empty range for parameter '{param}'.")
    
    # Determine if after_sum_zeroing is needed
    after_sum_zeroing = args.set_negatives_zero == 'after_sum'
    # Load previous waves without applying after_sum_zeroing here
    combined_wave = load_previous_waves(len(observed_data), args.waves_dir, set_negatives_zero=(args.set_negatives_zero == 'per_wave'))

    # Initialize plotting if not disabled
    if not args.no_plot:
        plt.ion()
        fig, ax = plt.subplots()
    else:
        ax = None  # No plotting

    wave_count = 0

    while True:
        # Check wave_count limit
        if args.wave_count != 0 and wave_count >= args.wave_count:
            logging.info(f"Reached the specified wave count of {args.wave_count}. Exiting.")
            break

        wave_count += args.num_waves
        wave_start = wave_count - args.num_waves + 1
        wave_end = wave_count
        wave_labels = ", ".join([str(w) for w in range(wave_start, wave_end + 1)])
        logging.info(f"Starting discovery of wave(s) {wave_labels}")

        # Dynamic step size selection based on average difference
        if args.progressive_step_sizes:
            difference = np.mean(np.abs(observed_data - combined_wave))
            if difference > 1000:
                step_size = 'fast'
            elif difference > 100:
                step_size = 'normal'
            elif difference > 20:
                step_size = 'fine'
            else:
                step_size = 'ultrafine'
            logging.info(f"Using {step_size} step size based on difference: {difference:.2f}")
        else:
            step_size = args.desired_step_size
            logging.info(f"Using {step_size} step size as per user configuration.")

        top_candidates = brute_force_sine_wave_search(
            observed_data, combined_wave, context, queue, ax, wave_count,
            desired_step_size=step_size,
            set_negatives_zero=args.set_negatives_zero,
            max_observed=max_observed,
            top_n=args.top_candidates,
            num_waves=args.num_waves
        )

        if args.desired_refinement_step_size.lower() != 'skip':
            refine_candidates(
                top_candidates, observed_data, combined_wave, context, queue, ax, wave_count,
                desired_refinement_step_size=args.desired_refinement_step_size,
                set_negatives_zero=args.set_negatives_zero,
                max_observed=max_observed,
                num_waves=args.num_waves,
                top_n=args.top_candidates
            )
            # After refinement, find the best candidate
            best_candidate = min(top_candidates, key=lambda x: x["score"])
            best_params = best_candidate["waves"]
            best_score = best_candidate["score"]
        else:
            if top_candidates:
                best_params = []
                for w in range(args.num_waves):
                    best_params.append(top_candidates[0]['waves'][w])
                best_score = top_candidates[0]['score']
                logging.info(f"Wave(s) {wave_labels}: Refinement phase skipped. Using top candidate from brute-force search.")
            else:
                best_params, best_score = None, np.inf
                logging.info(f"Wave(s) {wave_labels}: Refinement phase skipped. No candidates available from brute-force search.")

        if best_params is not None:
            # If multiple waves, save them as a list
            if args.num_waves > 1:
                best_params = [{k: float(v) for k, v in wave.items()} for wave in best_params]
            else:
                best_params = {k: float(v) for k, v in best_params.items()}
            wave_id = len([f for f in os.listdir(args.waves_dir) if f.endswith(".json")]) + 1
            with open(os.path.join(args.waves_dir, f"wave_{wave_id}.json"), "w") as f:
                json.dump(best_params, f)

            # Generate and add the new wave(s) to the combined_wave
            if args.num_waves > 1:
                for params in best_params:
                    new_wave = generate_sine_wave(params, len(observed_data), set_negatives_zero=(args.set_negatives_zero == 'per_wave'))
                    combined_wave += new_wave
            else:
                new_wave = generate_sine_wave(best_params, len(observed_data), set_negatives_zero=(args.set_negatives_zero == 'per_wave'))
                combined_wave += new_wave

            # No need to reload and zero the combined_wave here
            # Zeroing is handled during fitness evaluation and plotting

            if not args.no_plot and ax is not None:
                # For plotting, apply zeroing if 'after_sum' is specified
                if args.set_negatives_zero == 'after_sum':
                    combined_plus_sine = np.maximum(combined_wave, 0)
                else:
                    combined_plus_sine = combined_wave.copy()
                ax.clear()
                ax.plot(observed_data, label="Observed Data", color="blue")
                ax.plot(combined_plus_sine, label="Combined Sine Waves", color="orange")
                ax.set_title(f"Real-Time Fitting Progress - Waves {wave_labels}")
                ax.legend()
                plt.pause(0.01)

            logging.info(f"Wave(s) {wave_labels}: Best score: {best_score}")

        else:
            logging.warning("No valid sine wave parameters were found in the refinement phase.")

    if not args.no_plot:
        plt.ioff()
        plt.close(fig)

if __name__ == "__main__":
    main()
