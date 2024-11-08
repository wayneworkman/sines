# Sines Project

![Sines Project Logo](./sines.svg)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Intent](#intent)
- [How It Works](#how-it-works)
  - [Phase One: Brute-Force Search](#phase-one-brute-force-search)
  - [Phase Two: Refinement](#phase-two-refinement)
- [Usage](#usage)
  - [Generating Sine Waves (`sines.py`)](#generating-sine-waves-sinespy)
  - [Extrapolating Data (`extrapolator.py`)](#extrapolating-data-extrapolatorpy)
  - [Testing OpenCL Support (`test_OpenCL_support.py`)](#testing-opencl-support-test_opencl_supportpy)
  - [Running Unit Tests (`tests.py`)](#running-unit-tests-testspy)
  - [Sample Data Scripts](#sample-data-scripts)
- [Testing](#testing)
  - [Test Suite: `tests.py`](#test-suite-testspy)
- [Performance](#performance)
- [Known Limitations](#known-limitations)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The **Sines Project** is a powerful toolset designed for modeling and extrapolating time series data using sine wave decomposition. By breaking down complex time series into constituent sine waves, the project facilitates accurate predictions and analyses of historical and future data points. Sines was produced through collaborative work between a human and an A.I. Read the story inside of "the_journey.txt"

## Features

- **Automated Sine Wave Generation**: Utilizes brute-force and refinement techniques to identify optimal sine wave parameters that best fit your data.
- **GPU Acceleration**: Leverages GPU computational power through OpenCL for intensive calculations.
- **FFT-Based Initialization**: Employs Fast Fourier Transform (FFT) for initial frequency estimation, improving convergence and accuracy.
- **Simultaneous Dual Wave Optimization**: Capable of optimizing two sine waves simultaneously to capture complex patterns in the data.
- **Real-Time Visualization**: Monitors the fitting progress with dynamic plotting capabilities.
- **Data Extrapolation**: Reconstructs and extrapolates data using identified sine waves.
- **Logging**: Tracks the process with detailed log files.
- **Comprehensive Testing**: Includes unit tests to ensure reliability and correctness.

## Intent

The **Sines Project** provides a framework for decomposing and analyzing time series data. By modeling data as a sum of sine waves, users can uncover underlying periodicities and predict trends in complex systems.

## How It Works

The project operates in two primary phases: **Brute-Force Search** and **Refinement**.

### Phase One: Brute-Force Search

1. **Parameter Grid Generation**: Generates a grid of possible sine wave parameters (amplitude, frequency, phase shift). With the `--optimize-two-waves` flag, the grid encompasses combinations of two sine waves simultaneously.
2. **FFT-Based Frequency Estimation**: When enabled via `--use-fft-initialization`, the script uses the Fast Fourier Transform to estimate initial frequencies, focusing the search on the most promising frequency ranges.
3. **Fitness Calculation**: For each parameter combination, calculates a fitness score by comparing it against the observed data.
4. **Top Candidates Selection**: Retains the top-performing parameter combinations with the lowest fitness scores, configurable via `--top-candidates`.
5. **Real-Time Visualization**: Continuously updates plots to visualize the fitting progress.

### Phase Two: Refinement

1. **Focused Parameter Search**: Performs a finer search around top candidates from the brute-force phase. This phase can be skipped with `--desired-refinement-step-size skip`.
2. **Fitness Recalculation**: Recomputes fitness scores for refined parameter combinations.
3. **Best Parameters Identification**: Selects parameters that yield the best fitness scores.
4. **Optional Refinement**: Users can skip this phase for faster results.

## Usage

### Generating Sine Waves (`sines.py`)

The `sines.py` script processes time series data to identify sine waves that model the data accurately.

**Example Usage**:
```
python3 sines.py --data-file sample_data/sunspots/SN_d_tot_V2.0.csv --project-dir ~/sunspots --date-col date --value-col sunspot --desired-step-size fast --desired-refinement-step-size fast --wave-count 5 --set-negatives-zero after_sum --use-fft-initialization --optimize-two-waves
```

#### Arguments
- `--data-file`: **(Required)** Path to the input data file (JSON or CSV).
- `--project-dir`: **(Required)** Directory to store project data including waves and logs.
- `--date-col`: Name of the date column in the input data (default: `date`).
- `--value-col`: Name of the value column in the input data (default: `value`).
- `--moving-average`: Optional window size for smoothing data with a moving average.
- `--desired-step-size`: Step size mode for brute-force search (`fine`, `normal`, `fast`; default: `normal`).
- `--desired-refinement-step-size`: Step size mode for refinement phase (`fine`, `normal`, `fast`, `skip`; default: `normal`).
- `--progressive-step-sizes`: **(Flag)** Dynamically adjust step sizes based on observed and combined wave differences (default: enabled).
- `--set-negatives-zero`: How to handle negative sine wave values (`after_sum`, `per_wave`, `none`; default: `none`).
- `--no-plot`: **(Flag)** Disable real-time plotting.
- `--wave-count`: Specify the maximum number of waves to discover before the script stops. If set to `0`, the script will continue indefinitely (default: `50`).
- `--top-candidates`: Number of top candidates to consider during the brute-force search phase (default: `5`).
- `--use-fft-initialization`: **(Flag)** Enable FFT-based initial frequency estimation to focus the search on dominant frequencies.
- `--optimize-two-waves`: **(Flag)** Optimize two sine waves simultaneously during the brute-force search phase.

#### Detailed Explanation of `--set-negatives-zero`

The `--set-negatives-zero` argument controls how negative values in the sine waves are handled during the sine wave discovery and combination phases:

- **`after_sum`**:
  - **Fitting Process**: During the fitting (wave discovery) phase, each candidate sine wave is added to the existing combined wave. Immediately after summing, any negative values resulting from the combination are set to zero **before** the fitness evaluation. This ensures that the combined wave remains non-negative **only during evaluation and visualization**, while the true cumulative sum retains all sine wave contributions, including negative ones. This approach allows the fitting algorithm to assess the impact of each new wave accurately without permanently altering the cumulative waveform.
  - **Combination Phase**: Once a sine wave is selected and added, the combined wave used for evaluation has had its negative values zeroed **only** for that evaluation step, maintaining the integrity of the overall cumulative sum for future wave additions.
  - **Visualization**: The combined wave displayed in real-time visualization reflects the zeroed state during fitness evaluations, providing a clear and non-negative representation of the fitting progress.

- **`per_wave`**:
  - **Fitting Process**: Each candidate sine wave being evaluated has its negative values set to zero **before** being added to the combined wave. This ensures that only non-negative values from the candidate wave contribute to the combined wave, maintaining a strictly non-negative combined waveform throughout the entire fitting and combination processes.
  - **Combination Phase**: As each sine wave is added, the combined wave remains free of negative values since all contributing sine waves are non-negative.
  - **Visualization**: The combined wave never displays negative values at any point during the fitting and combination processes, providing a consistently non-negative representation.

- **`none`**:
  - **Fitting Process**: No zeroing is applied. Both individual sine waves and the combined waveform retain their negative values throughout the fitting and combination processes. This allows the algorithm to utilize the full oscillatory behavior of sine waves, potentially leading to a more accurate fit but resulting in a combined waveform that includes negative values.
  - **Combination Phase**: The cumulative sum of sine waves includes all positive and negative contributions, reflecting the true nature of the fitted model.
  - **Visualization**: The combined wave displays both positive and negative values, providing an unaltered view of the fitting progress.

**Choosing Between Options**:

- **`after_sum`**: Ideal when you want to maintain the full dynamics of sine wave interactions during fitting while ensuring non-negative representations in evaluations and visualizations.
- **`per_wave`**: Suitable for scenarios where a strictly non-negative combined waveform is required throughout the entire process.
- **`none`**: Best when the complete oscillatory behavior of sine waves is essential for accurate modeling without any constraints on the combined waveform's sign.

### Extrapolating Data (`extrapolator.py`)

The `extrapolator.py` script reconstructs and extrapolates the data using generated sine waves.

**Example Usage**:
```
python3 extrapolator.py --data-file sample_data/sunspots/SN_d_tot_V2.0.csv --project-dir ~/sunspots --date-col date --value-col sunspot --set-negatives-zero after_sum
```

#### Arguments
- `--data-file`: **(Required)** Path to the observed data CSV file.
- `--project-dir`: **(Required)** Directory containing project data including waves and logs.
- `--date-col`: Name of the column containing date information (default: `Timestamp`).
- `--value-col`: Name of the column containing observed values (default: `Value`).
- `--set-negatives-zero`: How to handle negative sine wave values (`after_sum`, `per_wave`, `none`; default: `none`).
- `--predict-before`: Percentage of data points to predict before the observed data (default: `5.0`).
- `--predict-after`: Percentage of data points to predict after the observed data (default: `5.0`).
- `--moving-average`: Apply a moving average filter to smooth the data (default: `None`).

#### Detailed Explanation of `--set-negatives-zero`

The `--set-negatives-zero` argument in `extrapolator.py` functions similarly to its counterpart in `sines.py`, determining how negative values in the combined sine waves are handled during data reconstruction and extrapolation:

- **`after_sum`**:
  - **Extrapolation Process**: After all sine waves are combined to form the extrapolated waveform, any negative values resulting from the sum are set to zero. This ensures that the extrapolated data does not contain negative values, which can be essential for datasets where negative values are not meaningful (e.g., sales figures).
  - **Impact on Data**: Maintains the integrity of individual sine waves during combination while ensuring the final extrapolated data is non-negative.

- **`per_wave`**:
  - **Extrapolation Process**: Each sine wave is processed individually to set negative values to zero **before** being added to the combined waveform. This ensures that the combined waveform remains non-negative throughout the entire extrapolation process.
  - **Impact on Data**: Guarantees that the extrapolated data remains non-negative at all stages, potentially simplifying interpretations where negative values are invalid.

- **`none`**:
  - **Extrapolation Process**: No zeroing is applied. Both individual sine waves and the combined extrapolated waveform retain their negative values.
  - **Impact on Data**: Allows for a complete representation of the sine wave contributions, including negative values, which may be necessary for datasets where negative values are valid and meaningful.

**Choosing Between Options**:

- **`after_sum`**: Suitable when you want the flexibility of negative sine wave contributions during combination but require the final extrapolated data to be non-negative.
- **`per_wave`**: Ideal for scenarios where maintaining non-negative values throughout the extrapolation process is crucial.
- **`none`**: Best when the dataset naturally includes negative values and you want to preserve the full dynamics of the combined sine waves without any alterations.

### Testing OpenCL Support (`test_OpenCL_support.py`)

The `test_OpenCL_support.py` script verifies OpenCL support and GPU functionality.

**Example Usage**:
```
python3 test_OpenCL_support.py
```

**Description**:
This script performs the following actions:
1. Lists all available OpenCL platforms and devices.
2. Executes a simple vector addition on the GPU to confirm OpenCL functionality.
3. Outputs the results of the vector addition to verify correctness.

**Expected Output**:
- Details of available OpenCL platforms and devices.
- Confirmation of successful vector addition, displaying input vectors and the resulting vector.
- A success message indicating that the OpenCL test was completed successfully.

### Running Unit Tests (`tests.py`)

The `tests.py` script contains a comprehensive suite of unit tests to validate various functionalities within the **Sines Project**.

**Example Usage**:
```
python3 tests.py
```

**Description**:
This test suite covers:
- Sine wave generation, including edge cases.
- Data loading from CSV and JSON files, including handling of malformed data.
- Wave parameter refinement processes.
- FFT-based frequency estimation and its integration.
- Optimization of two sine waves simultaneously.
- OpenCL platform and device detection.
- Integration tests simulating full workflows.
- Performance tests with varying dataset sizes and parameters.

## Performance

**Notes**:

- **High Amplitude Impact**: Datasets with higher maximum amplitudes expand the search space, leading to longer processing times during sine wave discovery.
- **Data Point Volume**: Larger numbers of data points increase computational load.

**Optimization Tips**:

- Utilize `--progressive-step-sizes` to dynamically adjust step sizes, potentially reducing search times.
- Consider skipping the refinement phase with `--desired-refinement-step-size skip` for faster, albeit less precise, results.
- Enable FFT-based frequency estimation with `--use-fft-initialization` to focus the search on dominant frequencies.
- Utilize the `--optimize-two-waves` flag to capture complex patterns more efficiently.
- Utilize GPU acceleration effectively by ensuring optimal OpenCL setup and driver configurations.

## Known Limitations

- **Date Range**: The date range for extrapolated data is constrained by the datetime library and Pandas limitations:
  - **Start Date**: 1677-09-22
  - **End Date**: 2262-04-10
- **Extrapolator Date Approximation**: Because `sines.py` and `extrapolator.py` both use data index rather than date to calculate data values for sine waves, the dates displayed within extrapolator are a very close approximation but not exactly correct.
- **Extrapolator data alignment**: Some observed data don't align correctly with combined sine wave values. This is a known issue and will hopefully be fixed in a future version.
- **Performance on Large Datasets**: High-amplitude datasets with extensive data points significantly increase processing times due to larger search spaces.
- **Step Size Configuration**: Improper step size configurations can lead to suboptimal sine wave discoveries or excessively long computation times.
- **Dependency on GPU**: Optimal performance relies on GPU availability and proper OpenCL setup. Systems without compatible GPUs may experience degraded performance.
- **FFT Limitations**: The FFT-based frequency estimation assumes that the dominant frequencies are present and may not capture all significant frequencies, especially in noisy data.
- **Simultaneous Wave Optimization Complexity**: Using `--optimize-two-waves` increases computational complexity and may lead to longer processing times.



## Logging

Both `sines.py` and `extrapolator.py` generate detailed logs.

- **Log Directory**: Defined by `--project-dir/logs`.
- **Log Contents**:
  - **Progress Updates**: Percentage completion of brute-force search phases.
  - **Fitness Scores**: Tracking the best and current fitness scores during searches.
  - **Parameter Selections**: Details of sine wave parameters being evaluated.
  - **Warnings and Error Messages**: Notifications about data loading issues, OpenCL errors, and other runtime warnings.

Additionally, each execution appends a log entry to `command_log.log` within the `logs` directory. This entry includes a timestamp and the full command with all arguments, facilitating easy reproduction of any project run.

Logs are instrumental for monitoring the process and diagnosing issues during wave discovery and data extrapolation. They are especially useful for long-running processes or when running scripts in non-interactive environments.

## Contributing

Contributions are welcome! Please follow these steps to contribute to the **Sines Project**:

1. **Fork the Repository**: Click the "Fork" button at the top-right corner of the repository page.
2. **Clone Your Fork**:  
```
git clone https://github.com/wayneworkman/sines.git
```
3. **Create a New Branch**:  
```
git checkout -b feature/your-feature-name
```
4. **Make Your Changes**: Implement your feature or fix.
5. **Execute, Update, and Add to the Tests**:
```
python3 tests.py
```
6. **Commit Your Changes**:  
```
git commit -m "Add feature: your feature description"
```
7. **Push to Your Fork**:  
```
git push origin feature/your-feature-name
```
8. **Open a Pull Request**: Navigate to the original repository and click "New Pull Request."

**Disclaimer**: This project is provided "as is" without warranty. Use at your own risk.

## License

This project is released under the MIT License. You are free to use, modify, and distribute it. However, portions of the sample_data provided are licensed differently. See the README in each of those directories as well as any licenses in those directories.
