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
  - [Test Script: `test_sines.py`](#test-script-test_sinespy)
  - [Test Suite: `tests.py`](#test-suite-testspy)
- [Performance](#performance)
- [Known Limitations](#known-limitations)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The **Sines Project** is a powerful toolset designed for modeling and extrapolating time series data using sine wave decomposition. By breaking down complex time series into constituent sine waves, the project facilitates accurate predictions and analyses of historical and future data points.

## Features

- **Automated Sine Wave Generation**: Utilizes brute-force and refinement techniques to identify optimal sine wave parameters that best fit your data.
- **GPU Acceleration**: Leverages GPU computational power through OpenCL for intensive calculations.
- **Real-Time Visualization**: Monitors the fitting progress with dynamic plotting capabilities.
- **Data Extrapolation**: Reconstructs and extrapolates data using identified sine waves.
- **Logging**: Tracks the process with detailed log files.
- **Comprehensive Testing**: Includes unit tests to ensure reliability and correctness.

## Intent

The **Sines Project** provides a framework for decomposing and analyzing time series data. By modeling data as a sum of sine waves, users can uncover underlying periodicities and predict trends in complex systems.

## How It Works

The project operates in two primary phases: **Brute-Force Search** and **Refinement**.

### Phase One: Brute-Force Search

1. **Parameter Grid Generation**: Generates a grid of possible sine wave parameters (amplitude, frequency, phase shift).
2. **Fitness Calculation**: For each parameter combination, calculates a fitness score by comparing it against the observed data.
3. **Top Candidates Selection**: Retains the top-performing parameter combinations with the lowest fitness scores.
4. **Real-Time Visualization**: Continuously updates plots to visualize the fitting progress.

### Phase Two: Refinement

1. **Focused Parameter Search**: Performs a finer search around top candidates from the brute-force phase.
2. **Fitness Recalculation**: Recomputes fitness scores for refined parameter combinations.
3. **Best Parameters Identification**: Selects parameters that yield the best fitness scores.
4. **Optional Refinement**: Users can skip this phase for faster results.

## Usage

### Generating Sine Waves (`sines.py`)

The `sines.py` script processes time series data to identify sine waves that model the data accurately.

**Example Usage**:
```
python3 sines.py --data-file sample_data/sunspots/SN_d_tot_V2.0.csv --date-col date --value-col sunspot --desired-step-size fast --desired-refinement-step-size fast --wave-count 5 --set-negatives-zero after_sum
```

#### Arguments
- `--data-file`: **(Required)** Path to the input data file (JSON or CSV).
- `--date-col`: Name of the date column in the input data (default: `date`).
- `--value-col`: Name of the value column in the input data (default: `value`).
- `--moving-average`: Optional window size for smoothing data with a moving average.
- `--waves-dir`: Directory to store generated sine wave parameters (default: `waves`).
- `--log-dir`: Directory to store log files (default: `logs`).
- `--desired-step-size`: Step size mode for brute-force search (`fine`, `normal`, `fast`; default: `fast`).
- `--desired-refinement-step-size`: Step size mode for refinement phase (`fine`, `normal`, `fast`, `skip`; default: `skip`).
- `--progressive-step-sizes`: **(Flag)** Dynamically adjust step sizes based on observed and combined wave differences (default: enabled).
- `--set-negatives-zero`: How to handle negative sine wave values (`after_sum`, `per_wave`; default: `after_sum`).
  - **`after_sum`**: Apply zeroing to the **combined sine waves** after summing all individual waves. This means that after each new sine wave is added to the cumulative sum, any resulting negative values in the combined waveform are set to zero **only** for the purpose of fitness evaluation and visualization. The actual cumulative sum retains its true value, including negative contributions, ensuring that all wave interactions are accurately considered during the fitting process.
  - **`per_wave`**: Apply zeroing to **each individual sine wave** before adding it to the combined wave. This ensures that each sine wave contributes only non-negative values to the combined wave, preventing any negative values from appearing in the combined waveform throughout the entire fitting and combination processes.
- `--no-plot`: **(Flag)** Disable real-time plotting.
- `--wave-count`: Specify the maximum number of waves to discover before the script stops. If set to `0`, the script will continue indefinitely (default: `50`).

#### Detailed Explanation of `--set-negatives-zero`

The `--set-negatives-zero` argument controls how negative values in the sine waves are handled during the sine wave discovery and combination phases:

- **`after_sum` (Default)**:
  - **Fitting Process**: During the fitting (wave discovery) phase, each candidate sine wave is added to the existing combined wave. Immediately after summing, any negative values resulting from the combination are set to zero **before** the fitness evaluation. This ensures that the combined wave remains non-negative **only during evaluation and visualization**, while the true cumulative sum retains all sine wave contributions, including negative ones. This approach allows the fitting algorithm to assess the impact of each new wave accurately without permanently altering the cumulative waveform.
  - **Combination Phase**: Once a sine wave is selected and added, the combined wave used for evaluation has had its negative values zeroed **only** for that evaluation step, maintaining the integrity of the overall cumulative sum for future wave additions.
  - **Visualization**: The combined wave displayed in real-time visualization reflects the zeroed state during fitness evaluations, providing a clear and non-negative representation of the fitting progress.

- **`per_wave`**:
  - **Fitting Process**: Each candidate sine wave being evaluated has its negative values set to zero **before** being added to the combined wave. This ensures that only non-negative values from the candidate wave contribute to the combined wave, maintaining a strictly non-negative combined waveform throughout the entire fitting and combination processes.
  - **Combination Phase**: As each sine wave is added, the combined wave remains free of negative values since all contributing sine waves are non-negative.
  - **Visualization**: The combined wave never displays negative values at any point during the fitting and combination processes, providing a consistently non-negative representation.

**Choosing Between `after_sum` and `per_wave`**:

- Use `--set-negatives-zero after_sum` when you want the fitting process to consider the full oscillatory behavior of sine waves, including negative values, for a more accurate fit. This option ensures that the combined wave remains non-negative by zeroing negative sums **only** during fitness evaluations, preserving the true cumulative sum of all waves for comprehensive fitting.

- Use `--set-negatives-zero per_wave` when you require that the combined wave remains non-negative throughout the entire process, including during fitting. This option restricts each sine wave to contribute only non-negative values, preventing any negative values in the combined wave at all times and ensuring a consistently non-negative waveform.

### Extrapolating Data (`extrapolator.py`)

The `extrapolator.py` script reconstructs and extrapolates the data using generated sine waves.

**Example Usage**:
```
python3 extrapolator.py --data-file sample_data/sunspots/SN_d_tot_V2.0.csv --date-col date --value-col sunspot
```

#### Arguments
- `--data-file`: **(Required)** Path to the observed data CSV file.
- `--waves-dir`: Directory containing sine wave JSON files (default: `waves`).
- `--date-col`: Name of the column containing date information (default: `Timestamp`).
- `--value-col`: Name of the column containing observed values (default: `Value`).
- `--set-negatives-zero`: How to handle negative sine wave values (`after_sum`, `per_wave`; default: `after_sum`).
- `--predict-before`: Percentage of data points to predict before the observed data (default: `5.0`).
- `--predict-after`: Percentage of data points to predict after the observed data (default: `5.0`).
- `--moving-average`: Apply a moving average filter to smooth the data (default: `None`).

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
- OpenCL platform and device detection.
- Integration tests simulating full workflows.
- Performance tests with varying dataset sizes and parameters.

## Performance

The **Sines Project** has been optimized for GPU resources, enhancing processing speed. 

**Performance Benchmarks**:

- **Processor**: Intel Core i5-2400 @ 3.10 GHz (4 cores)
- **RAM**: 16 GB
- **GPU**: Nvidia Quadro 6000
- **OS**: Ubuntu 20.04.6 LTS

**Dataset Examples**:

- **Solar Sunspot Data**:
  - **Data Points**: 75,546
  - **Maximum Amplitude**: ~375
  - **Performance**: Each brute-force search phase takes approximately one minute per wave discovery.

- **M4 Test Data**:
  - **Data Points**: 470
  - **Maximum Amplitude**: ~60,000
  - **Performance**: Each brute-force search phase takes about one second.

**Notes**:
- **High Amplitude Impact**: Datasets with higher maximum amplitudes expand the search space, leading to longer processing times during sine wave discovery.
- **Data Point Volume**: Larger numbers of data points increase computational load.
- **Optimization Tips**:
  - Utilize `--progressive-step-sizes` to dynamically adjust step sizes, potentially reducing search times.
  - Consider skipping the refinement phase with `--desired-refinement-step-size skip` for faster, albeit less precise, results.
  - Utilize GPU acceleration effectively by ensuring optimal OpenCL setup and driver configurations.

## Known Limitations

- **Date Range**: The date range for extrapolated data is constrained by the datetime library and Pandas limitations:
  - **Start Date**: 1677-09-22
  - **End Date**: 2262-04-10
- **Date Approximation**: Because `sines.py` and `extrapolator.py` both use data index rather than date to calculate data values for sine waves, the dates displayed within extrapolator are a very close approximation but not exactly correct.
- **Performance on Large Datasets**: High-amplitude datasets with extensive data points significantly increase processing times due to larger search spaces.
- **Step Size Configuration**: Improper step size configurations can lead to suboptimal sine wave discoveries or excessively long computation times.
- **Dependency on GPU**: Optimal performance relies on GPU availability and proper OpenCL setup. Systems without compatible GPUs may experience degraded performance.


## Logging

Both `sines.py` and `extrapolator.py` generate detailed logs.

- **Log Directory**: Defined by `--log-dir` in `sines.py`.
- **Log Contents**:
  - **Progress Updates**: Percentage completion of brute-force search phases.
  - **Fitness Scores**: Tracking the best and current fitness scores during searches.
  - **Parameter Selections**: Details of sine wave parameters being evaluated.
  - **Warnings and Error Messages**: Notifications about data loading issues, OpenCL errors, and other runtime warnings.

Logs are instrumental for monitoring the process and diagnosing issues during wave discovery and data extrapolation. They are especially useful for long-running processes or when running scripts in non-interactive environments.

## Contributing

Contributions are welcome! Please follow these steps to contribute to the **Sines Project**:

1. **Fork the Repository**: Click the "Fork" button at the top-right corner of the repository page.
1. **Clone Your Fork**:  
   ```
   git clone https://github.com/wayneworkman/sines.git
   ```
1. **Create a New Branch**:  
   ```
   git checkout -b feature/your-feature-name
   ```
1. **Make Your Changes**: Implement your feature or fix.
1. **Execute, Update, and Add to the Tests**:
   ```
   python3 tests.py
   ```
1. **Commit Your Changes**:  
   ```
   git commit -m "Add feature: your feature description"
   ```
1. **Push to Your Fork**:  
   ```
   git push origin feature/your-feature-name
   ```
1. **Open a Pull Request**: Navigate to the original repository and click "New Pull Request."

**Disclaimer**: This project is provided "as is" without warranty. Use at your own risk.
