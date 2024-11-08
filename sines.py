import pyopencl as cl
import numpy as np
import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import shlex

# New imports for FFT
from scipy.signal import find_peaks
import numpy.fft as fft
import itertools  # Import itertools for efficient iteration

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

# -------------------- OpenCL Kernel Definitions -------------------- #

KERNEL_CODE_SINGLE_WAVE = """
__kernel void calculate_fitness(
    __global const float *observed,
    __global const float *combined,
    __global const float *amplitudes,
    __global const float *frequencies,
    __global const float *phase_shifts,
    __global float *scores,
    int num_points,
    int zero_mode  // 0: no zeroing, 1: per_wave, 2: after_sum
) {
    int gid = get_global_id(0);
    float amplitude = amplitudes[gid];
    float frequency = frequencies[gid];
    float phase_shift = phase_shifts[gid];
    float score = 0.0f;

    for (int i = 0; i < num_points; i++) {
        float sine_value = amplitude * sin(2.0f * 3.141592653589793f * frequency * i + phase_shift);
        
        if (zero_mode == 1 && sine_value < 0) {
            sine_value = 0;  // per_wave zeroing
        }
        
        float combined_value = combined[i] + sine_value;
        
        if (zero_mode == 2 && combined_value < 0) {
            combined_value = 0;  // after_sum zeroing
        }
        
        float diff = observed[i] - combined_value;
        score += fabs(diff);
    }
    scores[gid] = score;
}
"""

KERNEL_CODE_DOUBLE_WAVE = """
__kernel void calculate_fitness_two_waves(
    __global const float *observed,
    __global const float *combined,
    __global const float *amplitudes1,
    __global const float *frequencies1,
    __global const float *phase_shifts1,
    __global const float *amplitudes2,
    __global const float *frequencies2,
    __global const float *phase_shifts2,
    __global float *scores,
    int num_points,
    int zero_mode  // 0: no zeroing, 1: per_wave, 2: after_sum
) {
    int gid = get_global_id(0);
    float amplitude1 = amplitudes1[gid];
    float frequency1 = frequencies1[gid];
    float phase_shift1 = phase_shifts1[gid];
    float amplitude2 = amplitudes2[gid];
    float frequency2 = frequencies2[gid];
    float phase_shift2 = phase_shifts2[gid];
    float score = 0.0f;

    for (int i = 0; i < num_points; i++) {
        float sine_value1 = amplitude1 * sin(2.0f * 3.141592653589793f * frequency1 * i + phase_shift1);
        float sine_value2 = amplitude2 * sin(2.0f * 3.141592653589793f * frequency2 * i + phase_shift2);
        
        if (zero_mode == 1) {
            if (sine_value1 < 0) sine_value1 = 0;
            if (sine_value2 < 0) sine_value2 = 0;
        }
        
        float combined_value = combined[i] + sine_value1 + sine_value2;
        
        if (zero_mode == 2 && combined_value < 0) {
            combined_value = 0;  // after_sum zeroing
        }
        
        float diff = observed[i] - combined_value;
        score += fabs(diff);
    }
    scores[gid] = score;
}
"""

# -------------------- FFT-Based Initial Frequency Estimation -------------------- #

def estimate_initial_frequencies(residual, sampling_rate=1.0):
    n = len(residual)
    freq_spectrum = fft.fft(residual)
    freq = fft.fftfreq(n, d=sampling_rate)
    magnitude = np.abs(freq_spectrum)
    # Only consider positive frequencies
    pos_mask = freq > 0
    freq = freq[pos_mask]
    magnitude = magnitude[pos_mask]
    # Identify peaks in the frequency spectrum
    peaks, _ = find_peaks(magnitude)
    peak_freqs = freq[peaks]
    peak_magnitudes = magnitude[peaks]
    # Sort peaks by magnitude
    sorted_indices = np.argsort(peak_magnitudes)[::-1]
    peak_freqs = peak_freqs[sorted_indices]
    return peak_freqs

# -------------------- Chunk Size Determination -------------------- #

def determine_chunk_size(total_combinations, max_work_group_size, max_mem_alloc_size, num_parameters=3):
    # Estimate the memory required per parameter combination
    # Each combination has num_parameters floats and score (1 float)
    bytes_per_combination = (num_parameters + 1) * 4  # floats * 4 bytes each

    # Calculate the maximum number of combinations that fit into the max memory allocation
    max_combinations_memory = max_mem_alloc_size // bytes_per_combination

    # Ensure chunk size is a multiple of max_work_group_size
    max_combinations_memory = (max_combinations_memory // max_work_group_size) * max_work_group_size

    # Set a reasonable upper limit if the calculated size is too large
    reasonable_upper_limit = 1 << 16  # 65,536
    chunk_size = min(max_combinations_memory, reasonable_upper_limit)

    # Ensure chunk_size does not exceed total_combinations
    chunk_size = min(chunk_size, total_combinations)

    # Ensure chunk_size is at least LOCAL_WORK_SIZE
    chunk_size = max(LOCAL_WORK_SIZE, chunk_size)

    return chunk_size

# -------------------- Brute-Force Search with Chunked Processing -------------------- #

def brute_force_sine_wave_search(observed_data, combined_wave, context, queue, ax, wave_count, desired_step_size='fast', set_negatives_zero=False, max_observed=1.0, max_work_group_size=LOCAL_WORK_SIZE, max_mem_alloc_size=134217728, num_top=5, initial_frequencies=None, optimize_two_waves=False):
    amplitude_range = STEP_SIZES[desired_step_size]["amplitude"]
    frequency_range = STEP_SIZES[desired_step_size]["frequency"]
    phase_shift_range = STEP_SIZES[desired_step_size]["phase_shift"]

    # Determine zero_mode based on set_negatives_zero
    if set_negatives_zero == 'per_wave':
        zero_mode = 1
    elif set_negatives_zero == 'after_sum':
        zero_mode = 2
    else:
        zero_mode = 0

    if optimize_two_waves:
        # [Code for optimizing two waves remains unchanged]
        pass
    else:
        logging.info(f"Wave {wave_count}: Starting brute-force search with one sine wave.")

        # Adjust frequency ranges based on initial frequencies if provided
        if initial_frequencies is not None and len(initial_frequencies) >= 1:
            delta_freq = frequency_range[1] - frequency_range[0] if len(frequency_range) > 1 else 0.0001
            # Expand frequency_range to include initial frequency if necessary
            start_freq = initial_frequencies[0] - delta_freq * 5
            end_freq = initial_frequencies[0] + delta_freq * 5
            # Ensure frequencies are within a reasonable range
            start_freq = max(start_freq, 0.000001)
            end_freq = min(end_freq, 0.5)  # Adjust 0.5 as per your data's maximum reasonable frequency
            if start_freq >= end_freq:
                logging.warning(f"Adjusted frequency range is invalid for initial frequency {initial_frequencies[0]:.6f}. Using default frequency range.")
                start_freq = frequency_range[0]
                end_freq = frequency_range[-1]
            frequency_range = np.arange(start_freq, end_freq, delta_freq)
            if len(frequency_range) == 0:
                logging.warning(f"Frequency range is empty after adjustment. Using default frequency range.")
                frequency_range = STEP_SIZES[desired_step_size]["frequency"]
        param_iter = itertools.product(
            amplitude_range, frequency_range, phase_shift_range
        )
        total_combinations = len(amplitude_range) * len(frequency_range) * len(phase_shift_range)

        num_parameters = 3  # Since we're optimizing one wave
        KERNEL_CODE = KERNEL_CODE_SINGLE_WAVE

    logging.info(f"Total combinations to evaluate: {total_combinations}")

    # Determine chunk size
    chunk_size = determine_chunk_size(total_combinations, max_work_group_size, max_mem_alloc_size, num_parameters)
    num_chunks = int(np.ceil(total_combinations / chunk_size))
    logging.info(f"Processing parameter combinations in {num_chunks} chunks of size {chunk_size}.")

    program = cl.Program(context, KERNEL_CODE).build()
    mf = cl.mem_flags
    observed_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=observed_data)
    combined_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=combined_wave)

    top_candidates = []
    best_score_so_far = np.inf

    # Process parameter combinations in chunks
    chunk_idx = 0
    while True:
        # Prepare parameter lists for the current chunk
        parameters_chunk = list(itertools.islice(param_iter, chunk_size))
        if not parameters_chunk:
            break  # No more combinations to process

        current_chunk_size = len(parameters_chunk)

        if optimize_two_waves:
            amplitude_chunk1 = np.array([p[0] for p in parameters_chunk], dtype=np.float32)
            frequency_chunk1 = np.array([p[1] for p in parameters_chunk], dtype=np.float32)
            phase_shift_chunk1 = np.array([p[2] for p in parameters_chunk], dtype=np.float32)
            amplitude_chunk2 = np.array([p[3] for p in parameters_chunk], dtype=np.float32)
            frequency_chunk2 = np.array([p[4] for p in parameters_chunk], dtype=np.float32)
            phase_shift_chunk2 = np.array([p[5] for p in parameters_chunk], dtype=np.float32)
            scores_chunk = np.empty(current_chunk_size, dtype=np.float32)

            amplitudes_buf1 = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=amplitude_chunk1)
            frequencies_buf1 = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frequency_chunk1)
            phase_shifts_buf1 = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phase_shift_chunk1)
            amplitudes_buf2 = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=amplitude_chunk2)
            frequencies_buf2 = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frequency_chunk2)
            phase_shifts_buf2 = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phase_shift_chunk2)
            scores_buf = cl.Buffer(context, mf.WRITE_ONLY, size=scores_chunk.nbytes)

            kernel = program.calculate_fitness_two_waves
            kernel.set_args(
                observed_buf, combined_buf, amplitudes_buf1, frequencies_buf1, phase_shifts_buf1,
                amplitudes_buf2, frequencies_buf2, phase_shifts_buf2,
                scores_buf, np.int32(len(observed_data)), np.int32(zero_mode)
            )
        else:
            amplitude_chunk = np.array([p[0] for p in parameters_chunk], dtype=np.float32)
            frequency_chunk = np.array([p[1] for p in parameters_chunk], dtype=np.float32)
            phase_shift_chunk = np.array([p[2] for p in parameters_chunk], dtype=np.float32)
            scores_chunk = np.empty(current_chunk_size, dtype=np.float32)

            amplitudes_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=amplitude_chunk)
            frequencies_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frequency_chunk)
            phase_shifts_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phase_shift_chunk)
            scores_buf = cl.Buffer(context, mf.WRITE_ONLY, size=scores_chunk.nbytes)

            kernel = program.calculate_fitness
            kernel.set_args(
                observed_buf, combined_buf, amplitudes_buf, frequencies_buf,
                phase_shifts_buf, scores_buf, np.int32(len(observed_data)), np.int32(zero_mode)
            )

        # Determine local work size
        if current_chunk_size >= max_work_group_size:
            if current_chunk_size % max_work_group_size == 0:
                local_work_size = (max_work_group_size,)
            else:
                for lws in range(max_work_group_size, 0, -1):
                    if current_chunk_size % lws == 0:
                        local_work_size = (lws,)
                        break
                else:
                    local_work_size = None
        else:
            local_work_size = (current_chunk_size,)

        # Enqueue kernel execution
        cl.enqueue_nd_range_kernel(queue, kernel, (current_chunk_size,), local_work_size)
        queue.finish()

        # Retrieve scores
        cl.enqueue_copy(queue, scores_chunk, scores_buf)
        queue.finish()

        # Select top `num_top` candidates from the current chunk
        indices = np.argsort(scores_chunk)[:num_top]
        for idx in indices:
            if optimize_two_waves:
                params = {
                    "amplitude1": amplitude_chunk1[idx],
                    "frequency1": frequency_chunk1[idx],
                    "phase_shift1": phase_shift_chunk1[idx],
                    "amplitude2": amplitude_chunk2[idx],
                    "frequency2": frequency_chunk2[idx],
                    "phase_shift2": phase_shift_chunk2[idx]
                }
            else:
                params = {
                    "amplitude": amplitude_chunk[idx],
                    "frequency": frequency_chunk[idx],
                    "phase_shift": phase_shift_chunk[idx]
                }
            score = scores_chunk[idx]
            top_candidates.append((params, score))
            if score < best_score_so_far:
                best_score_so_far = score

        # Keep only the top `num_top` candidates overall
        top_candidates = sorted(top_candidates, key=lambda x: x[1])[:num_top]

        if ax is not None:
            if chunk_idx % 5 == 0 or chunk_idx == num_chunks - 1:
                if top_candidates:
                    best_params = top_candidates[0][0]
                    if optimize_two_waves:
                        sine_wave1_params = {
                            "amplitude": best_params["amplitude1"],
                            "frequency": best_params["frequency1"],
                            "phase_shift": best_params["phase_shift1"]
                        }
                        sine_wave2_params = {
                            "amplitude": best_params["amplitude2"],
                            "frequency": best_params["frequency2"],
                            "phase_shift": best_params["phase_shift2"]
                        }
                        sine_wave1 = generate_sine_wave(sine_wave1_params, len(observed_data), set_negatives_zero=(set_negatives_zero == 'per_wave'))
                        sine_wave2 = generate_sine_wave(sine_wave2_params, len(observed_data), set_negatives_zero=(set_negatives_zero == 'per_wave'))
                        combined_plus_sine = combined_wave + sine_wave1 + sine_wave2
                    else:
                        sine_wave = generate_sine_wave(best_params, len(observed_data), set_negatives_zero=(set_negatives_zero == 'per_wave'))
                        combined_plus_sine = combined_wave + sine_wave

                    if set_negatives_zero == 'after_sum':
                        combined_plus_sine = np.maximum(combined_plus_sine, 0)
                    ax.clear()
                    ax.plot(observed_data, label="Observed Data", color="blue")
                    ax.plot(combined_plus_sine, label="Combined Sine Waves", color="orange")
                    if optimize_two_waves:
                        ax.set_title(f"Real-Time Fitting Progress - Wave {wave_count}, {wave_count + 1}")
                    else:
                        ax.set_title(f"Real-Time Fitting Progress - Wave {wave_count}")
                    ax.legend()
                    plt.pause(0.01)

        progress = min((chunk_idx + 1) * chunk_size, total_combinations) / total_combinations * 100
        if chunk_idx % 10 == 0:
            if optimize_two_waves:
                logging.info(f"Wave {wave_count}, {wave_count + 1}: Progress: {progress:.2f}%, Best Score So Far: {best_score_so_far}")
            else:
                logging.info(f"Wave {wave_count}: Progress: {progress:.2f}%, Best Score So Far: {best_score_so_far}")

        chunk_idx += 1

    if optimize_two_waves:
        logging.info(f"Wave {wave_count}, {wave_count + 1}: Completed brute-force search with best score: {best_score_so_far}")
    else:
        logging.info(f"Wave {wave_count}: Completed brute-force search with best score: {best_score_so_far}")

    return top_candidates

# -------------------- Refinement Phase Implementation -------------------- #

def refine_candidates(top_candidates, observed_data, combined_wave, context, queue, ax, wave_count, desired_refinement_step_size='fast', set_negatives_zero=False, max_observed=1.0, max_work_group_size=LOCAL_WORK_SIZE, max_mem_alloc_size=134217728):
    logging.info(f"Wave {wave_count}: Starting refinement phase.")

    refined_best_score = np.inf
    refined_best_params = None

    # Determine zero_mode based on set_negatives_zero
    if set_negatives_zero == 'per_wave':
        zero_mode = 1
    elif set_negatives_zero == 'after_sum':
        zero_mode = 2
    else:
        zero_mode = 0

    # Define the number of steps for each parameter
    num_amplitude_steps = 20
    num_frequency_steps = 20
    num_phase_shift_steps = 20

    # Define mf, observed_buf, combined_buf, and program
    mf = cl.mem_flags

    observed_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=observed_data)
    combined_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=combined_wave)

    program = cl.Program(context, KERNEL_CODE_SINGLE_WAVE).build()

    for candidate_idx, (params, _) in enumerate(top_candidates):
        amplitude_min = max(params["amplitude"] - 0.5 * max_observed, 0.1)
        amplitude_max = params["amplitude"] + 0.5 * max_observed
        amplitude_range = np.linspace(amplitude_min, amplitude_max, num_amplitude_steps)

        frequency_min = max(params["frequency"] - 0.000005, 0.000001)
        frequency_max = min(params["frequency"] + 0.000005, 0.5)  # Adjust upper limit as needed
        frequency_range = np.linspace(frequency_min, frequency_max, num_frequency_steps)

        phase_shift_min = (params["phase_shift"] - 0.1) % (2 * np.pi)
        phase_shift_max = (params["phase_shift"] + 0.1) % (2 * np.pi)
        if phase_shift_min < phase_shift_max:
            phase_shift_range = np.linspace(phase_shift_min, phase_shift_max, num_phase_shift_steps)
        else:
            # Handle wrapping around 2π
            range1 = np.linspace(phase_shift_min, 2 * np.pi, num=int(num_phase_shift_steps / 2))
            range2 = np.linspace(0, phase_shift_max, num=int(num_phase_shift_steps / 2))
            phase_shift_range = np.concatenate((range1, range2))

        amplitude_grid, frequency_grid, phase_shift_grid = np.meshgrid(
            amplitude_range, frequency_range, phase_shift_range, indexing='ij'
        )
        parameter_combinations = np.stack([
            amplitude_grid.ravel(),
            frequency_grid.ravel(),
            phase_shift_grid.ravel()
        ], axis=-1)
        total_combinations = parameter_combinations.shape[0]

        chunk_size = determine_chunk_size(total_combinations, max_work_group_size, max_mem_alloc_size)
        num_chunks = int(np.ceil(total_combinations / chunk_size))
        logging.info(f"Wave {wave_count}: Refinement candidate {candidate_idx + 1}/{len(top_candidates)} with {total_combinations} combinations.")

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total_combinations)
            current_chunk_size = end - start

            amplitude_chunk = parameter_combinations[start:end, 0].astype(np.float32)
            frequency_chunk = parameter_combinations[start:end, 1].astype(np.float32)
            phase_shift_chunk = parameter_combinations[start:end, 2].astype(np.float32)
            scores_chunk = np.empty(current_chunk_size, dtype=np.float32)

            amplitudes_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=amplitude_chunk)
            frequencies_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frequency_chunk)
            phase_shifts_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phase_shift_chunk)
            scores_buf = cl.Buffer(context, mf.WRITE_ONLY, size=scores_chunk.nbytes)

            kernel = program.calculate_fitness
            kernel.set_args(
                observed_buf, combined_buf, amplitudes_buf, frequencies_buf,
                phase_shifts_buf, scores_buf, np.int32(len(observed_data)), np.int32(zero_mode)
            )

            if current_chunk_size >= max_work_group_size:
                if current_chunk_size % max_work_group_size == 0:
                    local_work_size = (max_work_group_size,)
                else:
                    for lws in range(max_work_group_size, 0, -1):
                        if current_chunk_size % lws == 0:
                            local_work_size = (lws,)
                            break
                    else:
                        local_work_size = None
            else:
                local_work_size = (current_chunk_size,)

            cl.enqueue_nd_range_kernel(queue, kernel, (current_chunk_size,), local_work_size)
            queue.finish()

            cl.enqueue_copy(queue, scores_chunk, scores_buf)
            queue.finish()

            min_idx = np.argmin(scores_chunk)
            if scores_chunk[min_idx] < refined_best_score:
                refined_best_score = scores_chunk[min_idx]
                refined_best_params = {
                    "amplitude": amplitude_chunk[min_idx],
                    "frequency": frequency_chunk[min_idx],
                    "phase_shift": phase_shift_chunk[min_idx]
                }

            if ax is not None:
                if chunk_idx % 5 == 0 or chunk_idx == num_chunks - 1:
                    sine_wave = generate_sine_wave(refined_best_params, len(observed_data), set_negatives_zero=(set_negatives_zero == 'per_wave'))
                    combined_plus_sine = combined_wave + sine_wave
                    if set_negatives_zero == 'after_sum':
                        combined_plus_sine = np.maximum(combined_plus_sine, 0)
                    ax.clear()
                    ax.plot(observed_data, label="Observed Data", color="blue")
                    ax.plot(combined_plus_sine, label="Combined Sine Waves", color="orange")
                    ax.set_title(f"Refinement Progress - Wave {wave_count}")
                    ax.legend()
                    plt.pause(0.01)

            if chunk_idx % 10 == 0:
                progress = (chunk_idx + 1) / num_chunks * 100
                logging.info(f"Wave {wave_count}: Refinement Progress: {progress:.2f}%, Best Score So Far: {refined_best_score}")

    logging.info(f"Wave {wave_count}: Completed refinement phase with best score: {refined_best_score}")
    return refined_best_params, refined_best_score


# -------------------- End of Refinement Phase -------------------- #

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

    # Retrieve device-specific information for chunk size determination
    max_work_group_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    max_mem_alloc_size = device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
    logging.info(f"Device Max Work Group Size: {max_work_group_size}")
    logging.info(f"Device Max Memory Allocation Size: {max_mem_alloc_size} bytes")

    return context, queue, max_work_group_size, max_mem_alloc_size

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

def generate_sine_wave(params, num_points, set_negatives_zero=False):
    amplitude, frequency, phase_shift = params["amplitude"], params["frequency"], params["phase_shift"]
    t = np.arange(num_points, dtype=np.float32)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    if set_negatives_zero:
        np.maximum(sine_wave, 0, out=sine_wave)
    return sine_wave

def load_previous_waves(num_points, waves_dir, set_negatives_zero=False):
    """
    Load all previously saved waves and compute the true cumulative sum without applying zeroing.
    Zeroing (if required) is handled only during fitness evaluation.
    """
    combined_wave = np.zeros(num_points, dtype=np.float32)
    for filename in sorted(os.listdir(waves_dir)):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(waves_dir, filename), "r") as f:
                    wave_params = json.load(f)
                    sine_wave = generate_sine_wave(wave_params, num_points, set_negatives_zero)
                    combined_wave += sine_wave
            except json.JSONDecodeError:
                logging.warning(f"Error loading wave file {filename}. Skipping corrupted file.")
    return combined_wave

def main():
    parser = argparse.ArgumentParser(description="Time Series Modeling with Sine Waves")
    parser.add_argument('--data-file', type=str, required=True, help="Path to the data file")
    parser.add_argument('--date-col', type=str, default="date", help="Name of the date column in the data")
    parser.add_argument('--value-col', type=str, default="value", help="Name of the value column in the data")
    parser.add_argument('--project-dir', type=str, required=True, help="Directory to store project data including waves and logs")
    parser.add_argument('--moving-average', type=int, help="Optional moving average window for smoothing the data")
    parser.add_argument('--desired-step-size', type=str, choices=['fine', 'normal', 'fast'], default="normal",
                        help="Desired step size mode for brute-force search phase")
    parser.add_argument('--desired-refinement-step-size', type=str, choices=['fine', 'normal', 'fast', 'skip'], default="normal",
                        help="Desired step size mode for refinement phase. Use 'skip' to skip refinement.")
    parser.add_argument('--no-plot', action='store_true', help="Disable real-time plotting")
    parser.add_argument('--wave-count', type=int, default=50, help="Number of waves to generate before exiting. Use 0 for infinite.")

    parser.add_argument('--progressive-step-sizes', action='store_true', default=True,
                        help="Dynamically choose step size based on observed and combined wave differences")
    parser.add_argument('--set-negatives-zero', type=str, choices=['after_sum', 'per_wave', 'none'], default='none',
                        help="How to handle negative sine wave values: 'after_sum', 'per_wave', or 'none' (default)")

    # New Arguments
    parser.add_argument('--top-candidates', type=int, default=5, help="Number of top candidates to consider. Default is 5.")
    parser.add_argument('--use-fft-initialization', action='store_true',
                        help="Enable FFT-based initial frequency estimation")
    parser.add_argument('--optimize-two-waves', action='store_true',
                        help="Optimize two sine waves simultaneously")

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

    # Special Command Log Entry
    command_log_path = os.path.join(log_dir, "command_log.log")
    with open(command_log_path, "a") as cmd_log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        command = 'python3 ' + 'sines.py ' + ' '.join(shlex.quote(arg) for arg in sys.argv[1:])
        cmd_log_file.write(f"{timestamp} {command}\n")

    logging.info("Sines is Starting")

    context, queue, max_work_group_size, max_mem_alloc_size = setup_opencl()
    observed_data = load_data(
        args.data_file,
        date_col=args.date_col,
        value_col=args.value_col,
        moving_average=args.moving_average
    )

    if not os.path.exists(waves_dir):
        os.makedirs(waves_dir)

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

    # Determine if after_sum_zeroing is needed
    after_sum_zeroing = args.set_negatives_zero == 'after_sum'
    combined_wave = load_previous_waves(len(observed_data), waves_dir, set_negatives_zero=(args.set_negatives_zero == 'per_wave'))

    # Initialize plotting if not disabled
    if not args.no_plot:
        plt.ion()
        fig, ax = plt.subplots()
    else:
        ax = None  # No plotting

    wave_count = len([f for f in os.listdir(waves_dir) if f.endswith(".json")])
    while True:
        # Check wave_count limit
        if args.wave_count != 0 and wave_count >= args.wave_count:
            logging.info(f"Reached the specified wave count of {args.wave_count}. Exiting.")
            break

        if args.optimize_two_waves:
            logging.info(f"Starting discovery of waves {wave_count + 1}, {wave_count + 2}")
        else:
            logging.info(f"Starting discovery of wave {wave_count + 1}")

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

        # Use FFT-based initial frequency estimation if enabled
        if args.use_fft_initialization:
            residual = observed_data - combined_wave
            initial_frequencies = estimate_initial_frequencies(residual)
            logging.info(f"Initial frequency estimates from FFT: {initial_frequencies[:2]}")
        else:
            initial_frequencies = None

        # Pass `args.top_candidates` to brute_force_sine_wave_search
        top_candidates = brute_force_sine_wave_search(
            observed_data, combined_wave, context, queue, ax, wave_count + 1,
            desired_step_size=step_size,
            set_negatives_zero=args.set_negatives_zero,
            max_observed=max_observed,
            max_work_group_size=max_work_group_size,
            max_mem_alloc_size=max_mem_alloc_size,
            num_top=args.top_candidates,  # Configurable top candidates
            initial_frequencies=initial_frequencies,
            optimize_two_waves=args.optimize_two_waves
        )

        if args.desired_refinement_step_size.lower() != 'skip' and not args.optimize_two_waves:
            best_params, best_score = refine_candidates(
                top_candidates, observed_data, combined_wave, context, queue, ax, wave_count + 1,
                desired_refinement_step_size=args.desired_refinement_step_size,
                set_negatives_zero=args.set_negatives_zero,
                max_observed=max_observed,
                max_work_group_size=max_work_group_size,
                max_mem_alloc_size=max_mem_alloc_size
            )
        else:
            if top_candidates:
                best_params, best_score = top_candidates[0][0], top_candidates[0][1]
                if args.optimize_two_waves:
                    logging.info(f"Wave {wave_count + 1}, {wave_count + 2}: Refinement phase skipped. Using top candidate from brute-force search.")
                else:
                    logging.info(f"Wave {wave_count + 1}: Refinement phase skipped. Using top candidate from brute-force search.")
            else:
                best_params, best_score = None, np.inf
                if args.optimize_two_waves:
                    logging.info(f"Wave {wave_count + 1}, {wave_count + 2}: Refinement phase skipped. No candidates available from brute-force search.")
                else:
                    logging.info(f"Wave {wave_count + 1}: Refinement phase skipped. No candidates available from brute-force search.")

        if best_params is not None:
            if args.optimize_two_waves:
                sine_wave1_params = {
                    "amplitude": float(best_params["amplitude1"]),
                    "frequency": float(best_params["frequency1"]),
                    "phase_shift": float(best_params["phase_shift1"])
                }
                sine_wave2_params = {
                    "amplitude": float(best_params["amplitude2"]),
                    "frequency": float(best_params["frequency2"]),
                    "phase_shift": float(best_params["phase_shift2"])
                }
                wave_id1 = wave_count + 1
                wave_id2 = wave_count + 2
                with open(os.path.join(waves_dir, f"wave_{wave_id1}.json"), "w") as f:
                    json.dump(sine_wave1_params, f)
                with open(os.path.join(waves_dir, f"wave_{wave_id2}.json"), "w") as f:
                    json.dump(sine_wave2_params, f)

                new_wave1 = generate_sine_wave(sine_wave1_params, len(observed_data), set_negatives_zero=(args.set_negatives_zero == 'per_wave'))
                new_wave2 = generate_sine_wave(sine_wave2_params, len(observed_data), set_negatives_zero=(args.set_negatives_zero == 'per_wave'))
                combined_wave += new_wave1 + new_wave2

                if not args.no_plot and ax is not None:
                    if args.set_negatives_zero == 'after_sum':
                        combined_plus_sine = np.maximum(combined_wave, 0)
                    else:
                        combined_plus_sine = combined_wave.copy()
                    ax.clear()
                    ax.plot(observed_data, label="Observed Data", color="blue")
                    ax.plot(combined_plus_sine, label="Combined Sine Waves", color="orange")
                    ax.set_title(f"Real-Time Fitting Progress - {wave_count + 2} Waves")
                    ax.legend()
                    plt.pause(0.01)

                logging.info(f"Waves {wave_count + 1}, {wave_count + 2}: Best score: {best_score}")
                wave_count += 2
            else:
                best_params = {k: float(v) for k, v in best_params.items()}
                wave_id = wave_count + 1
                with open(os.path.join(waves_dir, f"wave_{wave_id}.json"), "w") as f:
                    json.dump(best_params, f)

                new_wave = generate_sine_wave(best_params, len(observed_data), set_negatives_zero=(args.set_negatives_zero == 'per_wave'))
                combined_wave += new_wave

                if not args.no_plot and ax is not None:
                    if args.set_negatives_zero == 'after_sum':
                        combined_plus_sine = np.maximum(combined_wave, 0)
                    else:
                        combined_plus_sine = combined_wave.copy()
                    ax.clear()
                    ax.plot(observed_data, label="Observed Data", color="blue")
                    ax.plot(combined_plus_sine, label="Combined Sine Waves", color="orange")
                    ax.set_title(f"Real-Time Fitting Progress - {wave_count + 1} Waves")
                    ax.legend()
                    plt.pause(0.01)

                logging.info(f"Wave {wave_count + 1}: Best score: {best_score}")
                wave_count += 1
        else:
            logging.warning("No valid sine wave parameters were found in the refinement phase.")

    if not args.no_plot:
        plt.ioff()
        plt.close(fig)

if __name__ == "__main__":
    import sys  # Needed for command log entry
    main()
