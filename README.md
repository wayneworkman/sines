
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
  - [Generating Sine Waves](#generating-sine-waves)
  - [Extrapolating Data](#extrapolating-data)
- [Performance](#performance)
- [Known Limitations](#known-limitations)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The **Sines Project** is a powerful toolset designed for modeling and extrapolating time series data using sine wave decomposition. By breaking down complex time series into constituent sine waves, the project facilitates accurate predictions and analyses of historical and future data points.

## Features

- **Automated Sine Wave Generation**: Uses brute-force and refinement techniques to identify optimal sine wave parameters that best fit your data.
- **GPU Acceleration**: Leverages GPU computational power for intensive calculations.
- **Real-Time Visualization**: Monitors the fitting progress with dynamic plotting capabilities.
- **Data Extrapolation**: Reconstructs and extrapolates data using identified sine waves.
- **Logging**: Tracks the process with detailed log files.

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
python3 sines.py --data-file sample_data/sunspots/SN_d_tot_V2.0.csv --date-col date --value-col sunspot --desired-step-size fast --desired-refinement-step-size fast --wave-count 5 --set-negatives-zero
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

## Performance

The **Sines Project** has been optimized for GPU resources, enhancing processing speed. Benchmarks:

- **Processor**: Intel Core i5-2400 @ 3.10 GHz (4 cores)
- **RAM**: 16 GB
- **GPU**: Nvidia Quadro 6000
- **OS**: Ubuntu 20.04.6 LTS
- **Dataset**: Solar Sunspot Data with 75,546 data points (CSV format)

On this setup, each brute-force search phase takes approximately 8-10 seconds per wave discovery.

## Known Limitations

- **Date Range**: The date range for extrapolated data is constrained by the datetime library and Pandas limitations:
  - **Start Date**: 1677-09-22
  - **End Date**: 2262-04-10
- **Date-Free Sines Processing**: The `sines.py` script generates sine waves based on data indices, without a date context. Thus, the date range limitation within `extrapolator.py` is not present within `sines.py`.

## How Sine Waves are Mapped to Dates

The sine waves generated by `sines.py` are accurately mapped onto a date-based timeline in the final extrapolated graph (the displayed matplotlib graph (users can manually save the graph via the interface if needed)) using these steps:

1. **Parameter-Based Wave Generation**: `sines.py` produces sine waves as amplitude values indexed by data position.
2. **Cumulative Series Construction**: These sine waves are saved in JSON files (e.g., `wave_1.json`) representing a cumulative series that, when summed, aligns with observed sunspot variations.
3. **Date Mapping with `extrapolator.py`**:
   - The `extrapolator.py` script defines a fixed date range (`START_DATE = 1677-09-22`, `END_DATE = 2262-04-10`) and maps the cumulative sine wave series onto it.
   - Using `pd.date_range`, the script generates a date index to match each reconstructed sine wave value to a calendar date.
4. **Summing Over the Date Range**: The sine wave parameters are re-generated over the full date range, covering historical and future dates. The resulting series is plotted with actual data, creating an accurate time alignment.

This method provides a robust and accurate date-based reconstruction.

## Logging

Both `sines.py` and `extrapolator.py` generate detailed logs.
- **Log Directory**: Defined by `--log-dir` in `sines.py`.
- **Log Contents**:
  - Progress updates
  - Fitness scores
  - Parameter selections
  - Warnings and error messages

## OpenCL Support Test

The repository includes a script, `test_OpenCL_support.py`, for verifying OpenCL support and GPU functionality.

```bash
python test_OpenCL_support.py
```

Expected output includes details of available platforms and devices, as well as a sample calculation result.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

**Disclaimer**: This project is provided "as is" without warranty. Use at your own risk.