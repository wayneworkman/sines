import pyopencl as cl
import numpy as np
import json
import os
import requests
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import time

# Configuration constants
WAVES_DIR = "waves"  # Directory to store best-fit wave parameters
LOGS_DIR = "logs"    # Directory to store log files
LOCAL_WORK_SIZE = 256  # Work group size for OpenCL, must be compatible with GPU

# Desired step size mode for Phase 1
DESIRED_STEP_SIZE = "fast"  # Options: 'fine', 'normal', 'fast'

# Desired step size mode for Phase 2 refinement
DESIRED_REFINEMENT_STEP_SIZE = "fast"  # Options: 'fine', 'normal', 'fast'

# Phase 1 step sizes configuration
STEP_SIZES = {
    'fine': {
        'amplitude': np.arange(1, 101, 1.0),  # Small step for precision
        'frequency': np.arange(0.00001, 0.001, 0.000015),
        'phase_shift': np.arange(0, 2 * np.pi, 0.05)
    },
    'normal': {
        'amplitude': np.arange(1, 101, 1.5),
        'frequency': np.arange(0.00001, 0.001, 0.00003),
        'phase_shift': np.arange(0, 2 * np.pi, 0.15)
    },
    'fast': {
        'amplitude': np.arange(1, 101, 3.0),
        'frequency': np.arange(0.00001, 0.001, 0.00006),
        'phase_shift': np.arange(0, 2 * np.pi, 0.3)
    }
}

# Phase 2 refinement step sizes configuration
REFINEMENT_STEP_SIZES = {
    'fine': {
        'amplitude_step': 0.05,    # Small step for precision
        'frequency_step': 0.0000005,
        'phase_shift_step': 0.005
    },
    'normal': {
        'amplitude_step': 0.1,
        'frequency_step': 0.000001,
        'phase_shift_step': 0.01
    },
    'fast': {
        'amplitude_step': 0.2,
        'frequency_step': 0.000005,
        'phase_shift_step': 0.02
    }
}

# Logging setup
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
log_filename = os.path.join(LOGS_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

logging.info("Sunspot Prediction Model Starting")

# Ensure necessary directories exist
if not os.path.exists(WAVES_DIR):
    os.makedirs(WAVES_DIR)

# GPU setup function
def setup_opencl():
    """Configures the OpenCL environment and selects the NVIDIA GPU device."""
    platforms = cl.get_platforms()
    nvidia_platform = next((p for p in platforms if 'NVIDIA' in p.name), None)
    if not nvidia_platform:
        raise RuntimeError("NVIDIA platform not found.")
    devices = nvidia_platform.get_devices(device_type=cl.device_type.GPU)
    if not devices:
        raise RuntimeError("No GPU devices found on NVIDIA platform.")

    # Select and configure the GPU
    device = devices[0]  # Use the first available GPU
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    logging.info(f"Using device: {device.name}")
    logging.info(f"Max work group size: {device.max_work_group_size}")

    global MAX_LOCAL_WORK_SIZE
    MAX_LOCAL_WORK_SIZE = device.max_work_group_size
    logging.info(f"Max local work size: {MAX_LOCAL_WORK_SIZE}")

    return context, queue

# Download and preprocess sunspot data
def download_and_process_sunspot_data(url):
    """Downloads, processes, and applies a 27-day moving average to the sunspot data."""
    logging.info("Downloading sunspot data.")
    response = requests.get(url)
    data = response.text.strip().splitlines()
    sunspot_data = []

    for line in data:
        fields = line.split(';')
        if len(fields) < 5:
            continue
        try:
            year, month, day = int(fields[0]), int(fields[1]), int(fields[2])
            sn_value = int(float(fields[4].strip()))
            sunspot_data.append(sn_value)
        except (ValueError, IndexError):
            continue

    # Convert to numpy array for easy manipulation
    sunspot_data = np.array(sunspot_data, dtype=np.float32)

    # Apply a 27-day moving average
    averaged_sunspot_data = np.convolve(sunspot_data, np.ones(27) / 27, mode='valid')

    logging.info("Sunspot data downloaded, processed, and 27-day moving average applied.")
    return averaged_sunspot_data

# Load existing sine waves from previous runs
def load_previous_waves(num_points):
    """Loads sine waves from previous best-fit parameters and combines them."""
    combined_wave = np.zeros(num_points, dtype=np.float32)
    for filename in sorted(os.listdir(WAVES_DIR)):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(WAVES_DIR, filename), "r") as f:
                    wave_params = json.load(f)
                    if not all(k in wave_params for k in ["amplitude", "frequency", "phase_shift"]):
                        raise ValueError(f"Invalid parameters in {filename}")
                    combined_wave += generate_sine_wave(wave_params, num_points)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logging.warning(f"Skipping invalid wave file {filename}: {e}")
    return combined_wave

# Generate a sine wave based on parameters
def generate_sine_wave(params, num_points):
    """Generates a sine wave given amplitude, frequency, and phase shift."""
    amplitude, frequency, phase_shift = params["amplitude"], params["frequency"], params["phase_shift"]
    t = np.arange(num_points, dtype=np.float32)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    np.maximum(sine_wave, 0, out=sine_wave)  # Enforce non-negative values
    return sine_wave

# Brute-force search phase
def brute_force_sine_wave_search(observed_data, combined_wave, context, queue, ax, wave_count):
    """Performs a brute-force search over sine wave parameters to find the best fit."""
    logging.info(f"Starting Phase 1: Brute-Force Search for Wave {wave_count}")

    # Select step sizes based on the desired mode
    amplitude_range = STEP_SIZES[DESIRED_STEP_SIZE]["amplitude"]
    frequency_range = STEP_SIZES[DESIRED_STEP_SIZE]["frequency"]
    phase_shift_range = STEP_SIZES[DESIRED_STEP_SIZE]["phase_shift"]

    # Create grid of parameter combinations
    amplitude_grid, frequency_grid, phase_shift_grid = np.meshgrid(
        amplitude_range, frequency_range, phase_shift_range, indexing='ij'
    )
    parameter_combinations = np.stack([
        amplitude_grid.ravel(),
        frequency_grid.ravel(),
        phase_shift_grid.ravel()
    ], axis=-1)
    total_combinations = parameter_combinations.shape[0]
    logging.info(f"Total parameter combinations: {total_combinations}")

    # OpenCL kernel for fitness calculation
    kernel_code = """
    __kernel void calculate_fitness(
        __global const float *observed,
        __global const float *combined,
        __global const float *amplitudes,
        __global const float *frequencies,
        __global const float *phase_shifts,
        __global float *scores,
        int num_points
    ) {
        int gid = get_global_id(0);
        float amplitude = amplitudes[gid];
        float frequency = frequencies[gid];
        float phase_shift = phase_shifts[gid];
        float score = 0.0f;

        for (int i = 0; i < num_points; i++) {
            float sine_value = amplitude * sin(2.0f * 3.141592653589793f * frequency * i + phase_shift);
            if (sine_value < 0) sine_value = 0;
            float diff = observed[i] - (combined[i] + sine_value);
            score += fabs(diff);
        }
        scores[gid] = score;
    }
    """
    program = cl.Program(context, kernel_code).build()

    # Transfer constant data to GPU memory once
    mf = cl.mem_flags
    observed_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=observed_data)
    combined_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=combined_wave)

    # Prepare to process combinations in chunks
    chunk_size = 1024  # You can adjust this based on your GPU's memory capacity
    num_chunks = int(np.ceil(total_combinations / chunk_size))
    logging.info(f"Processing parameter combinations in {num_chunks} chunks of size {chunk_size}")

    top_candidates = []
    start_time = time.time()
    last_log_time = start_time  # For controlling log frequency
    best_score_so_far = np.inf  # Track the best score across all chunks

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_combinations)
        current_chunk_size = end - start

        amplitude_chunk = parameter_combinations[start:end, 0].astype(np.float32)
        frequency_chunk = parameter_combinations[start:end, 1].astype(np.float32)
        phase_shift_chunk = parameter_combinations[start:end, 2].astype(np.float32)
        scores_chunk = np.empty(current_chunk_size, dtype=np.float32)

        # Allocate buffers for parameters and scores
        amplitudes_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=amplitude_chunk)
        frequencies_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frequency_chunk)
        phase_shifts_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phase_shift_chunk)
        scores_buf = cl.Buffer(context, mf.WRITE_ONLY, size=scores_chunk.nbytes)

        # Set kernel arguments and execute
        kernel = program.calculate_fitness
        kernel.set_args(
            observed_buf, combined_buf, amplitudes_buf, frequencies_buf,
            phase_shifts_buf, scores_buf, np.int32(len(observed_data))
        )

        # Adjust local work size dynamically
        if current_chunk_size >= LOCAL_WORK_SIZE:
            if current_chunk_size % LOCAL_WORK_SIZE == 0:
                local_work_size = (LOCAL_WORK_SIZE,)
            else:
                # Adjust local work size to the greatest divisor of current_chunk_size less than LOCAL_WORK_SIZE
                for lws in range(LOCAL_WORK_SIZE, 0, -1):
                    if current_chunk_size % lws == 0:
                        local_work_size = (lws,)
                        break
                else:
                    local_work_size = None
        else:
            local_work_size = (current_chunk_size,)

        # Enqueue kernel
        cl.enqueue_nd_range_kernel(queue, kernel, (current_chunk_size,), local_work_size)
        queue.finish()

        # Retrieve scores
        cl.enqueue_copy(queue, scores_chunk, scores_buf)
        queue.finish()

        # Find top candidates in this chunk
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
            # Update best score if found
            if score < best_score_so_far:
                best_score_so_far = score

        # Keep only the overall top candidates
        top_candidates = sorted(top_candidates, key=lambda x: x[1])[:num_top]

        # Real-time plotting and less frequent logging
        if chunk_idx % 5 == 0 or chunk_idx == num_chunks - 1:
            best_params = top_candidates[0][0]
            sine_wave = generate_sine_wave(best_params, len(observed_data))
            ax.clear()
            ax.plot(observed_data, label="Observed Data", color="blue")
            ax.plot(combined_wave + sine_wave, label="Combined Sine Waves", color="orange")
            ax.set_title("Real-Time Fitting Progress")
            ax.legend()
            plt.pause(0.01)

        # Log only every 10 seconds
        current_time = time.time()
        if current_time - last_log_time >= 10:
            elapsed_time = current_time - start_time
            progress = (chunk_idx + 1) / num_chunks * 100
            logging.info(f"Wave {wave_count}: Progress: {progress:.2f}%, Elapsed Time: {elapsed_time / 60:.2f} minutes, Best Score So Far: {best_score_so_far}")
            last_log_time = current_time

    logging.info(f"Completed Phase 1 for Wave {wave_count}")
    return top_candidates

# Refinement phase for top candidates
def refine_candidates(top_candidates, observed_data, combined_wave, context, queue, ax, wave_count):
    """Refines the top candidates from Phase 1 by fine-tuning parameters for a more precise fit."""
    logging.info(f"Starting Phase 2: Refinement Phase for Wave {wave_count}")

    refined_best_score = np.inf
    refined_best_params = None

    # OpenCL kernel for fitness calculation (same as Phase 1)
    kernel_code = """
    __kernel void calculate_fitness(
        __global const float *observed,
        __global const float *combined,
        __global const float *amplitudes,
        __global const float *frequencies,
        __global const float *phase_shifts,
        __global float *scores,
        int num_points
    ) {
        int gid = get_global_id(0);
        float amplitude = amplitudes[gid];
        float frequency = frequencies[gid];
        float phase_shift = phase_shifts[gid];
        float score = 0.0f;

        for (int i = 0; i < num_points; i++) {
            float sine_value = amplitude * sin(2.0f * 3.141592653589793f * frequency * i + phase_shift);
            if (sine_value < 0) sine_value = 0;
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

    start_time = time.time()
    last_log_time = start_time  # For controlling log frequency

    for params, _ in top_candidates:
        # Retrieve step sizes based on desired refinement mode
        amplitude_step = REFINEMENT_STEP_SIZES[DESIRED_REFINEMENT_STEP_SIZE]["amplitude_step"]
        frequency_step = REFINEMENT_STEP_SIZES[DESIRED_REFINEMENT_STEP_SIZE]["frequency_step"]
        phase_shift_step = REFINEMENT_STEP_SIZES[DESIRED_REFINEMENT_STEP_SIZE]["phase_shift_step"]

        # Define parameter ranges with boundary checks
        amplitude_min = max(params["amplitude"] - 0.5, 0.1)  # Amplitude should be positive
        amplitude_max = params["amplitude"] + 0.5
        amplitude_range = np.arange(amplitude_min, amplitude_max, amplitude_step)

        frequency_min = max(params["frequency"] - 0.000005, 0.000001)  # Avoid negative frequencies
        frequency_max = min(params["frequency"] + 0.000005, 0.01)  # Assuming upper frequency limit
        frequency_range = np.arange(frequency_min, frequency_max, frequency_step)

        phase_shift_min = params["phase_shift"] - 0.1
        phase_shift_max = params["phase_shift"] + 0.1
        # Ensure phase_shift remains within [0, 2π]
        phase_shift_min = phase_shift_min % (2 * np.pi)
        phase_shift_max = phase_shift_max % (2 * np.pi)
        if phase_shift_min < phase_shift_max:
            phase_shift_range = np.arange(phase_shift_min, phase_shift_max, phase_shift_step)
        else:
            # Handle wrap-around
            phase_shift_range = np.concatenate((
                np.arange(phase_shift_min, 2 * np.pi, phase_shift_step),
                np.arange(0, phase_shift_max, phase_shift_step)
            ))

        # Create grid of refined parameter combinations
        amplitude_grid, frequency_grid, phase_shift_grid = np.meshgrid(
            amplitude_range, frequency_range, phase_shift_range, indexing='ij'
        )
        parameter_combinations = np.stack([
            amplitude_grid.ravel(),
            frequency_grid.ravel(),
            phase_shift_grid.ravel()
        ], axis=-1)
        total_combinations = parameter_combinations.shape[0]

        # Process combinations in chunks
        chunk_size = 1024  # Adjust based on GPU capacity
        num_chunks = int(np.ceil(total_combinations / chunk_size))
        logging.info(f"Processing refinement phase in {num_chunks} chunks of size {chunk_size}")

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total_combinations)
            current_chunk_size = end - start

            amplitude_chunk = parameter_combinations[start:end, 0].astype(np.float32)
            frequency_chunk = parameter_combinations[start:end, 1].astype(np.float32)
            phase_shift_chunk = parameter_combinations[start:end, 2].astype(np.float32)
            scores_chunk = np.empty(current_chunk_size, dtype=np.float32)

            # Allocate buffers for parameters and scores
            amplitudes_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=amplitude_chunk)
            frequencies_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frequency_chunk)
            phase_shifts_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phase_shift_chunk)
            scores_buf = cl.Buffer(context, mf.WRITE_ONLY, size=scores_chunk.nbytes)

            # Set kernel arguments and execute
            kernel = program.calculate_fitness
            kernel.set_args(
                observed_buf, combined_buf, amplitudes_buf, frequencies_buf,
                phase_shifts_buf, scores_buf, np.int32(len(observed_data))
            )

            # Adjust local work size dynamically
            if current_chunk_size >= LOCAL_WORK_SIZE:
                if current_chunk_size % LOCAL_WORK_SIZE == 0:
                    local_work_size = (LOCAL_WORK_SIZE,)
                else:
                    # Adjust local work size to the greatest divisor of current_chunk_size less than LOCAL_WORK_SIZE
                    for lws in range(LOCAL_WORK_SIZE, 0, -1):
                        if current_chunk_size % lws == 0:
                            local_work_size = (lws,)
                            break
                    else:
                        local_work_size = None
            else:
                local_work_size = (current_chunk_size,)

            # Enqueue kernel
            cl.enqueue_nd_range_kernel(queue, kernel, (current_chunk_size,), local_work_size)
            queue.finish()

            # Retrieve scores
            cl.enqueue_copy(queue, scores_chunk, scores_buf)
            queue.finish()

            # Find the best score and parameters in this chunk
            min_idx = np.argmin(scores_chunk)
            if scores_chunk[min_idx] < refined_best_score:
                refined_best_score = scores_chunk[min_idx]
                refined_best_params = {
                    "amplitude": amplitude_chunk[min_idx],
                    "frequency": frequency_chunk[min_idx],
                    "phase_shift": phase_shift_chunk[min_idx]
                }

            # Log high-level progress every 10 seconds
            current_time = time.time()
            if current_time - last_log_time >= 10:
                elapsed_time = current_time - start_time
                logging.info(f"Refinement for Wave {wave_count}: Best Score So Far: {refined_best_score}, Elapsed Time: {elapsed_time / 60:.2f} minutes")
                last_log_time = current_time

    logging.info(f"Completed Phase 2 for Wave {wave_count}")
    return refined_best_params, refined_best_score

# Main function
def main():
    # Initialize OpenCL environment
    context, queue = setup_opencl()

    # Load and preprocess sunspot data
    url = "http://www.sidc.be/silso/INFO/sndtotcsv.php"
    observed_data = download_and_process_sunspot_data(url)
    observed_data = np.array(observed_data, dtype=np.float32)

    # Initialize combined_wave
    combined_wave = np.zeros_like(observed_data, dtype=np.float32)

    # Load previously discovered waves to build the base model
    combined_wave += load_previous_waves(len(observed_data))

    # Initialize interactive plot for real-time updates
    plt.ion()
    fig, ax = plt.subplots()

    wave_count = 0  # Keep track of the number of waves discovered

    # Start the wave discovery loop
    while True:
        wave_count += 1
        logging.info(f"Starting discovery of wave {wave_count}")

        # Phase 1: Brute-force search to find initial best-fit sine waves
        top_candidates = brute_force_sine_wave_search(observed_data, combined_wave, context, queue, ax, wave_count)

        # Phase 2: Refinement of the top candidates from Phase 1
        best_params, best_score = refine_candidates(top_candidates, observed_data, combined_wave, context, queue, ax, wave_count)

        # Save best-fit parameters and update combined_wave
        if best_params is not None:
            # Convert best_params to native Python floats for JSON serialization
            best_params = {k: float(v) for k, v in best_params.items()}
            wave_id = len([f for f in os.listdir(WAVES_DIR) if f.endswith(".json")]) + 1
            with open(os.path.join(WAVES_DIR, f"wave_{wave_id}.json"), "w") as f:
                json.dump(best_params, f)

            # Update combined_wave with the new wave
            new_wave = generate_sine_wave(best_params, len(observed_data))
            combined_wave += new_wave

            # Update the live plot
            ax.clear()
            ax.plot(observed_data, label="Observed Data", color="blue")
            ax.plot(combined_wave, label="Combined Sine Waves", color="orange")
            ax.set_title(f"Real-Time Fitting Progress - {wave_count} Waves")
            ax.legend()
            plt.pause(0.01)

            # Optionally, implement a stopping criterion here
            # For example, stop if the best_score is below a threshold
            # if best_score < desired_threshold:
            #     logging.info(f"Desired accuracy achieved. Stopping at wave {wave_count}.")
            #     break

        else:
            logging.warning("No valid sine wave parameters were found in the refinement phase.")
            # Optionally, implement a stopping condition if no valid parameters are found
            # break

    # Close the plot when done (unreachable in this infinite loop)
    plt.ioff()
    plt.close(fig)

if __name__ == "__main__":
    main()
