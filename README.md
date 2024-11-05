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
```bash
python3 sines.py --data-file sample_data/sunspots/SN_d_tot_V2.0.csv --date-col date --value-col sunspot --desired-step-size fast --desired-refinement-step-size fast --wave-count 5 --set-negatives-zero after_sum
```

#### Arguments
- `--data-file`: Path to the input data file (JSON or CSV).
- `--date-col`: Name of the date column in the input data (default: `date`).
- `--value-col`: Name of the value column in the input data (default: `value`).
- `--moving-average`: Optional window size for smoothing data with a moving average.
- `--waves-dir`: Directory to store generated sine wave parameters (default: `waves`).
- `--log-dir`: Directory to store log files (default: `logs`).
- `--desired-step-size`: Step size mode for brute-force search (`fine`, `normal`, `fast`; default: `fast`).
- `--desired-refinement-step-size`: Step size mode for refinement phase (`fine`, `normal`, `fast`, `skip`; default: `skip`).
- `--no-plot`: Disable real-time plotting.
- `--wave-count`: Specify the maximum number of waves to discover before the script stops. If set to `0`, the script will continue indefinitely (default: `50`).

### Extrapolating Data (`extrapolator.py`)

The `extrapolator.py` script reconstructs and extrapolates the data using generated sine waves.

**Usage**:
```bash
python3 extrapolator.py --data-file sample_data/sunspots/SN_d_tot_V2.0.csv --date-col date --value-col sunspot
```

This command will load all sine wave parameters, combine them, and plot the reconstructed time series data alongside the actual data.

### Testing OpenCL Support (`test_OpenCL_support.py`)

The `test_OpenCL_support.py` script verifies OpenCL support and GPU functionality.

**Usage**:
```bash
python3 test_OpenCL_support.py
```

Expected output includes details of available platforms and devices, as well as a sample calculation result.

### Running Unit Tests (`tests.py`)

The `tests.py` script contains a comprehensive suite of unit tests to validate various functionalities within the **Sines Project**.

**Usage**:
```bash
python -m unittest tests.py
```

### Sample Data Scripts

The `sample_data` directory contains scripts designed to pull, update, or generate test data for the **Sines Project**. These scripts have their own README files, which provide detailed instructions on their usage and functionalities.

## Testing

### Test Script: `test_sines.py`

The `test_sines.py` script is a comprehensive suite of unit tests that validate various functionalities within the **Sines Project**. It tests key components such as sine wave generation, data loading, wave parameter refinement, and OpenCL support for GPU acceleration.

**Highlights of the Test Cases**:
- **Sine Wave Generation**: Tests basic and edge cases for sine wave generation.
- **Data Loading**: Validates data loading from CSV and JSON, handling missing or malformed data, and testing moving averages.
- **Wave Parameter Refinement**: Ensures that the refinement function can handle edge cases, such as empty candidate lists.
- **Integration Tests**: Simulates full integration, from loading data to searching and refining wave parameters.
- **OpenCL Support**: Mocks and tests for OpenCL platform and device detection, ensuring compatibility with NVIDIA GPUs.

Run the test suite with:
```bash
python -m unittest test_sines.py
```

### Test Suite: `tests.py`

The `tests.py` script extends the testing framework with additional test cases, ensuring comprehensive coverage of all functionalities.

Run the test suite with:
```bash
python -m unittest tests.py
```

## Performance

The **Sines Project** has been optimized for GPU resources, enhancing processing speed. 

**Performance Benchmarks**:

- **Processor**: Intel Core i5-2400 @ 3.10 GHz (4 cores)
- **RAM**: 16 GB
- **GPU**: Nvidia Quadro 6000
- **OS**: Ubuntu 20.04.6 LTS
- **Dataset Examples**:
  - **Solar Sunspot Data**: 75,546 data points with a maximum amplitude of ~375. Each brute-force search phase takes approximately one minute per wave discovery.
  - **M4 Test Data**: 470 data points with a maximum amplitude nearing 60,000. Each wave discovery with this data takes about one second.

**Note**: Datasets with higher maximum amplitudes increase the search space, resulting in longer processing times during sine wave discovery. However, a smaller number of data points decreases the number of computations needed to evaluate each sine wave within the search space.

## Known Limitations

- **Date Range**: The date range for extrapolated data is constrained by the datetime library and Pandas limitations:
  - **Start Date**: 1677-09-22
  - **End Date**: 2262-04-10
- **Date-Free Sines Processing**: The `sines.py` script generates sine waves based on data indices, without a date context. Thus, the date range limitation within `extrapolator.py` is not present within `sines.py`.
- **Performance on Large Datasets**: High-amplitude datasets with extensive data points significantly increase processing times due to larger search spaces.

## Logging

Both `sines.py` and `extrapolator.py` generate detailed logs.

- **Log Directory**: Defined by `--log-dir` in `sines.py`.
- **Log Contents**:
  - Progress updates
  - Fitness scores
  - Parameter selections
  - Warnings and error messages

Logs are instrumental for monitoring the process and diagnosing issues during wave discovery and data extrapolation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

**Disclaimer**: This project is provided "as is" without warranty. Use at your own risk.
