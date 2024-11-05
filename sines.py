import pyopencl as cl
import numpy as np
import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from json import dumps
from sys import exit

# Constants
LOCAL_WORK_SIZE = 256  # Work group size for OpenCL, must be compatible with GPU

# Configuration constants for step sizes

STEP_SIZES = {
    'ultrafine': {
        'amplitude': np.arange(1, 20000, 1),
        'frequency': np.arange(0.00001, 0.001, 0.0000075),
        'phase_shift': np.arange(0, 2 * np.pi, 0.025)
    },
    'fine': {
        'amplitude': np.arange(1, 20000, 2),
        'frequency': np.arange(0.00001, 0.001, 0.000015),
        'phase_shift': np.arange(0, 2 * np.pi, 0.05)
    },
    'normal': {
        'amplitude': np.arange(1, 20000, 4),
        'frequency': np.arange(0.00001, 0.001, 0.00003),
        'phase_shift': np.arange(0, 2 * np.pi, 0.15)
    },
    'fast': {
        'amplitude': np.arange(1, 20000, 8),
        'frequency': np.arange(0.00001, 0.001, 0.00006),
        'phase_shift': np.arange(0, 2 * np.pi, 0.3)
    }
}


# Refinement step sizes configuration with smaller steps

REFINEMENT_STEP_SIZES = {
    'ultrafine': {
        'amplitude_step': 0.01,
        'frequency_step': 0.00000005,
        'phase_shift_step': 0.0005
    },
    'fine': {
        'amplitude_step': 0.02,
        'frequency_step': 0.0000001,
        'phase_shift_step': 0.001
    },
    'normal': {
        'amplitude_step': 0.05,
        'frequency_step': 0.0000005,
        'phase_shift_step': 0.002
    },
    'fast': {
        'amplitude_step': 0.1,
        'frequency_step': 0.000001,
        'phase_shift_step': 0.005
    }
}


# -------------------- Refinement Phase Implementation -------------------- #

def refine_candidates(top_candidates, observed_data, combined_wave, context, queue, ax, wave_count, desired_refinement_step_size='fast', set_negatives_zero=False):
    logging.info(f"Wave {wave_count}: Starting refinement phase.")


    # This logic will activate when combined_wave contains something other than all zeros,
    # which should be the case for wave 2 discovery.
    # if np.all(combined_wave != 0):
    #     # Extract numerical values from top_candidates
    #     # Each candidate is a tuple: (params_dict, score)
    #     # We will extract amplitude, frequency, phase_shift, and score
    #     top_candidates_numeric = [
    #         [candidate[0]['amplitude'], candidate[0]['frequency'], candidate[0]['phase_shift'], candidate[1]]
    #         for candidate in top_candidates
    #     ]

    #     # Save top_candidates_numeric as a human-readable text file
    #     np.savetxt(
    #         'test_data_top_candidates.txt',
    #         top_candidates_numeric,
    #         delimiter=',',
    #         header='amplitude,frequency,phase_shift,score',
    #         comments=''
    #     )

    #     # Save observed_data and combined_wave as before
    #     np.savetxt('test_data_observed_data.txt', observed_data, delimiter=',')
    #     np.savetxt('test_data_combined_wave.txt', combined_wave, delimiter=',')

    #     exit()

    
    refined_best_score = np.inf
    refined_best_params = None

    # Additional parameter for kernel to control negative value handling
    non_negative = 1 if set_negatives_zero else 0

    kernel_code = """
    __kernel void calculate_fitness(
        __global const float *observed,
        __global const float *combined,
        __global const float *amplitudes,
        __global const float *frequencies,
        __global const float *phase_shifts,
        __global float *scores,
        int num_points,
        int non_negative
    ) {
        int gid = get_global_id(0);
        float amplitude = amplitudes[gid];
        float frequency = frequencies[gid];
        float phase_shift = phase_shifts[gid];
        float score = 0.0f;

        for (int i = 0; i < num_points; i++) {
            float sine_value = amplitude * sin(2.0f * 3.141592653589793f * frequency * i + phase_shift);
            if (non_negative && sine_value < 0) sine_value = 0;
            float diff = observed[i] - (combined[i] + sine_value);
            score += fabs(diff);
        }
        scores[gid] = score;
    }
    """
    program = cl.Program(context, kernel_code).build()
    mf = cl.mem_flags
    observed_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=observed_data)
    combined_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=combined_wave)

    for candidate_idx, (params, _) in enumerate(top_candidates):
        amplitude_step = REFINEMENT_STEP_SIZES[desired_refinement_step_size]["amplitude_step"]
        frequency_step = REFINEMENT_STEP_SIZES[desired_refinement_step_size]["frequency_step"]
        phase_shift_step = REFINEMENT_STEP_SIZES[desired_refinement_step_size]["phase_shift_step"]

        amplitude_min = max(params["amplitude"] - 0.5, 0.1)
        amplitude_max = params["amplitude"] + 0.5
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

        amplitude_grid, frequency_grid, phase_shift_grid = np.meshgrid(
            amplitude_range, frequency_range, phase_shift_range, indexing='ij'
        )
        parameter_combinations = np.stack([
            amplitude_grid.ravel(),
            frequency_grid.ravel(),
            phase_shift_grid.ravel()
        ], axis=-1)
        total_combinations = parameter_combinations.shape[0]

        chunk_size = 1024
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
                phase_shifts_buf, scores_buf, np.int32(len(observed_data)), np.int32(non_negative)
            )

            if current_chunk_size >= LOCAL_WORK_SIZE:
                if current_chunk_size % LOCAL_WORK_SIZE == 0:
                    local_work_size = (LOCAL_WORK_SIZE,)
                else:
                    for lws in range(LOCAL_WORK_SIZE, 0, -1):
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
                    sine_wave = generate_sine_wave(refined_best_params, len(observed_data), set_negatives_zero)
                    ax.clear()
                    ax.plot(observed_data, label="Observed Data", color="blue")
                    ax.plot(combined_wave + sine_wave, label="Combined Sine Waves", color="orange")
                    ax.set_title(f"Refinement Progress - Wave {wave_count}")
                    ax.legend()
                    plt.pause(0.01)

            if chunk_idx % 10 == 0:
                elapsed_time = chunk_idx * chunk_size / (total_combinations / 100)
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

    return context, queue

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
    
    return df[value_col].values.astype(np.float32)

def generate_sine_wave(params, num_points, set_negatives_zero=False):
    amplitude, frequency, phase_shift = params["amplitude"], params["frequency"], params["phase_shift"]
    t = np.arange(num_points, dtype=np.float32)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    if set_negatives_zero:
        np.maximum(sine_wave, 0, out=sine_wave)
    return sine_wave

def load_previous_waves(num_points, output_dir, set_negatives_zero=False):
    combined_wave = np.zeros(num_points, dtype=np.float32)
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(output_dir, filename), "r") as f:
                    wave_params = json.load(f)
                    combined_wave += generate_sine_wave(wave_params, num_points, set_negatives_zero)
            except json.JSONDecodeError:
                logging.warning(f"Error loading wave file {filename}. Skipping corrupted file.")
    return combined_wave

def brute_force_sine_wave_search(observed_data, combined_wave, context, queue, ax, wave_count, desired_step_size='fast', set_negatives_zero=False):
    amplitude_range = STEP_SIZES[desired_step_size]["amplitude"]
    frequency_range = STEP_SIZES[desired_step_size]["frequency"]
    phase_shift_range = STEP_SIZES[desired_step_size]["phase_shift"]

    amplitude_grid, frequency_grid, phase_shift_grid = np.meshgrid(
        amplitude_range, frequency_range, phase_shift_range, indexing='ij'
    )
    parameter_combinations = np.stack([
        amplitude_grid.ravel(),
        frequency_grid.ravel(),
        phase_shift_grid.ravel()
    ], axis=-1)
    total_combinations = parameter_combinations.shape[0]

    logging.info(f"Wave {wave_count}: Starting brute-force search with {total_combinations} combinations.")

    # Additional parameter for kernel to control negative value handling
    non_negative = 1 if set_negatives_zero else 0

    kernel_code = """
    __kernel void calculate_fitness(
        __global const float *observed,
        __global const float *combined,
        __global const float *amplitudes,
        __global const float *frequencies,
        __global const float *phase_shifts,
        __global float *scores,
        int num_points,
        int non_negative
    ) {
        int gid = get_global_id(0);
        float amplitude = amplitudes[gid];
        float frequency = frequencies[gid];
        float phase_shift = phase_shifts[gid];
        float score = 0.0f;

        for (int i = 0; i < num_points; i++) {
            float sine_value = amplitude * sin(2.0f * 3.141592653589793f * frequency * i + phase_shift);
            if (non_negative && sine_value < 0) sine_value = 0;
            float diff = observed[i] - (combined[i] + sine_value);
            score += fabs(diff);
        }
        scores[gid] = score;
    }
    """
    program = cl.Program(context, kernel_code).build()
    mf = cl.mem_flags
    observed_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=observed_data)
    combined_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=combined_wave)

    chunk_size = 1024
    num_chunks = int(np.ceil(total_combinations / chunk_size))
    logging.info(f"Wave {wave_count}: Processing parameter combinations in {num_chunks} chunks of size {chunk_size}.")

    top_candidates = []
    best_score_so_far = np.inf

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
            phase_shifts_buf, scores_buf, np.int32(len(observed_data)), np.int32(non_negative)
        )

        if current_chunk_size >= LOCAL_WORK_SIZE:
            if current_chunk_size % LOCAL_WORK_SIZE == 0:
                local_work_size = (LOCAL_WORK_SIZE,)
            else:
                for lws in range(LOCAL_WORK_SIZE, 0, -1):
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

        num_top = 20
        indices = np.argsort(scores_chunk)[:num_top]
        for idx in indices:
            params = {
                "amplitude": amplitude_chunk[idx],
                "frequency": frequency_chunk[idx],
                "phase_shift": phase_shift_chunk[idx]
            }
            score = scores_chunk[idx]
            top_candidates.append((params, score))
            if score < best_score_so_far:
                best_score_so_far = score

        top_candidates = sorted(top_candidates, key=lambda x: x[1])[:num_top]

        if ax is not None:
            if chunk_idx % 5 == 0 or chunk_idx == num_chunks - 1:
                best_params = top_candidates[0][0]
                sine_wave = generate_sine_wave(best_params, len(observed_data), set_negatives_zero)
                ax.clear()
                ax.plot(observed_data, label="Observed Data", color="blue")
                ax.plot(combined_wave + sine_wave, label="Combined Sine Waves", color="orange")
                ax.set_title(f"Real-Time Fitting Progress - Wave {wave_count}")
                ax.legend()
                plt.pause(0.01)

        if chunk_idx % 10 == 0:
            elapsed_time = chunk_idx * chunk_size / (total_combinations / 100)
            progress = (chunk_idx + 1) / num_chunks * 100
            logging.info(f"Wave {wave_count}: Progress: {progress:.2f}%, Best Score So Far: {best_score_so_far}")

    logging.info(f"Wave {wave_count}: Completed brute-force search with best score: {best_score_so_far}")
    return top_candidates

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

    parser.add_argument('--progressive-step-sizes', action='store_true', default=True,
                        help="Dynamically choose step size based on observed and combined wave differences")
    parser.add_argument('--set-negatives-zero', action='store_true', help="Set sine wave values below zero to zero")
    args = parser.parse_args()

    # Setup logging after parsing arguments
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_filename = os.path.join(args.log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    
    logging.info("Sunspot Prediction Model Starting")

    context, queue = setup_opencl()
    observed_data = load_data(
        args.data_file,
        date_col=args.date_col,
        value_col=args.value_col,
        moving_average=args.moving_average
    )

    if not os.path.exists(args.waves_dir):
        os.makedirs(args.waves_dir)

    combined_wave = load_previous_waves(len(observed_data), args.waves_dir, set_negatives_zero=args.set_negatives_zero)

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

        wave_count += 1
        logging.info(f"Starting discovery of wave {wave_count}")


        # Dynamic step size selection based on average difference
        if args.progressive_step_sizes:
            difference = np.mean(np.abs(observed_data - combined_wave))
            if difference > 150:
                step_size = 'fast'
            elif difference > 50:
                step_size = 'normal'
            elif difference > 10:
                step_size = 'fine'
            else:
                step_size = 'ultrafine'
            logging.info(f"Using {step_size} step size based on difference: {difference:.2f}")
        else:
            step_size = args.desired_step_size
        top_candidates = brute_force_sine_wave_search(
            observed_data, combined_wave, context, queue, ax, wave_count,
            desired_step_size=args.desired_step_size,
            set_negatives_zero=args.set_negatives_zero
        )

        if args.desired_refinement_step_size.lower() != 'skip':
            best_params, best_score = refine_candidates(
                top_candidates, observed_data, combined_wave, context, queue, ax, wave_count,
                desired_refinement_step_size=args.desired_refinement_step_size,
                set_negatives_zero=args.set_negatives_zero
            )
        else:
            best_params, best_score = top_candidates[0][0], top_candidates[0][1]
            logging.info(f"Wave {wave_count}: Refinement phase skipped. Using top candidate from brute-force search.")

        if best_params is not None:
            best_params = {k: float(v) for k, v in best_params.items()}
            wave_id = len([f for f in os.listdir(args.waves_dir) if f.endswith(".json")]) + 1
            with open(os.path.join(args.waves_dir, f"wave_{wave_id}.json"), "w") as f:
                json.dump(best_params, f)

            new_wave = generate_sine_wave(best_params, len(observed_data), set_negatives_zero=args.set_negatives_zero)
            combined_wave += new_wave

            if not args.no_plot and ax is not None:
                ax.clear()
                ax.plot(observed_data, label="Observed Data", color="blue")
                ax.plot(combined_wave, label="Combined Sine Waves", color="orange")
                ax.set_title(f"Real-Time Fitting Progress - {wave_count} Waves")
                ax.legend()
                plt.pause(0.01)

            logging.info(f"Wave {wave_count}: Best score: {best_score}")

        else:
            logging.warning("No valid sine wave parameters were found in the refinement phase.")

    if not args.no_plot:
        plt.ioff()
        plt.close(fig)

if __name__ == "__main__":
    main()