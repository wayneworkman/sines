```
# Generate Sample Time Series Data

This utility script generates synthetic time series data based on a combination of sine waves and saves it as a CSV file. The generated data can be used as input for `sines.py`, enabling testing with custom time series data in place of actual data.

## Features
- Generates time series data by combining multiple sine waves with customizable parameters.
- Adds Gaussian noise to each wave to create more realistic data.
- Provides flexible handling of negative sine wave values (`after_sum`, `per_wave`, or `none`).
- Saves data to a CSV file with `date` and `value` columns, which is compatible with `sines.py`.
- Logs detailed information and errors for easier debugging and monitoring.
- Includes comprehensive unit tests to ensure reliability.

## Requirements
- Python 3.x
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `argparse`
  - `logging`
  
You can install the required libraries using:
```
pip install numpy pandas
```

## Usage
Run the script from the command line, providing optional arguments to control various aspects of the generated data.

### Basic Command
```
python generate_timeseries.py
```

### Command-line Arguments
| Argument                   | Type   | Default            | Description                                              |
|----------------------------|--------|--------------------|----------------------------------------------------------|
| `--training-output-file`   | String | `training_data.csv` | Path to save the generated training data CSV.            |
| `--testing-output-file`    | String | `testing_data.csv` | Path to save the generated testing data CSV.             |
| `--training-start-date`    | String | `2000-01-01`      | Start date for the training data in `YYYY-MM-DD` format. |
| `--training-end-date`      | String | `2005-01-01`      | End date for the training data in `YYYY-MM-DD` format.   |
| `--testing-start-date`     | String | `1990-01-01`      | Start date for the testing data in `YYYY-MM-DD` format.  |
| `--testing-end-date`       | String | `2015-01-01`      | End date for the testing data in `YYYY-MM-DD` format.    |
| `--waves-dir`              | String | `waves`           | Directory to store individual wave parameters.           |
| `--num-waves`              | Int    | `5`               | Number of sine waves to combine in the generated data.   |
| `--max-amplitude`          | Float  | `150`             | Maximum amplitude for the generated sine waves.          |
| `--max-frequency`          | Float  | `0.001`           | Maximum frequency for the generated sine waves.          |
| `--noise-std`              | Float  | `1`               | Standard deviation of noise added to the sine waves.     |
| `--set-negatives-zero`     | String | `none`            | How to handle negative sine wave values: `'after_sum'`, `'per_wave'`, or `'none'` (default). |
| `--parameters-dir`         | String | `run_parameters`  | Directory to store run parameters.                       |

### Example Command
```
python generate_timeseries.py --training-output-file="training_data.csv" --testing-output-file="testing_data.csv" --training-start-date="2000-01-01" --training-end-date="2005-01-01" --testing-start-date="1990-01-01" --testing-end-date="2015-01-01" --waves-dir="waves" --num-waves=7 --max-amplitude=100 --max-frequency=0.002 --noise-std=2 --set-negatives-zero="per_wave"
```

This command will generate a synthetic dataset with:
- 7 combined sine waves
- A maximum amplitude of 100
- A maximum frequency of 0.002
- Gaussian noise with a standard deviation of 2
- Negative values in each wave set to zero before summing

The training data will be saved to `training_data.csv`, and testing data to `testing_data.csv`.

## Output Format
The generated CSV file will contain two columns:
- `date`: The date for each data point in `YYYY-MM-DD` format.
- `value`: The combined value of the sine waves at each date, with added noise.

Sample output format:
```
date,value
2000-01-01,23.5
2000-01-02,25.1
2000-01-03,21.8
...
```

## Using the Data with Sines

Example usage
```
# Utilize the training data with sines.
python3 sines.py --data-file sample_data/generate_timeseries/training_data.csv --date-col date --value-col value --wave-count 0

# Utilize the testing data with the extrapolator.
python3 extrapolator.py --data-file sample_data/generate_timeseries/testing_data.csv --value-col value --date-col date
```

## Unit Tests

The unit tests for this script are located in the `tests.py` file within the same directory. You can run the tests using the following command:

```
python -m unittest tests.py
```

## Logging

The script uses Python's `logging` module to provide detailed information and error messages during execution. Logs include:
- Creation of directories for waves and run parameters.
- Saving of individual wave parameters.
- Generation of the combined wave.
- Splitting and saving of training and testing data.
- Any errors encountered during execution.

Logs are output to the console with timestamps and log levels (INFO, ERROR).

## Error Handling

Enhanced error handling ensures that:
- Start dates are not after end dates for both training and testing ranges.
- Numerical parameters like `num_waves`, `max_amplitude`, `max_frequency`, and `noise_std` are within valid ranges.
- Directories are created successfully or appropriate errors are logged.
- Missing or invalid wave parameters are handled gracefully.

If invalid arguments are provided, the script will terminate gracefully with informative error messages.

## Notes
- This utility is designed to produce synthetic data for testing purposes and is not based on actual or other observational data.
- The script can be adapted to generate various types of time series data by adjusting the parameters.
- Ensure that all required directories have the appropriate permissions for the script to create and write files.

## License
This script is released under the MIT License. You are free to use, modify, and distribute it for non-commercial purposes.
