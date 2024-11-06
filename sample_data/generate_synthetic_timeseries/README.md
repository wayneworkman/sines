
# Generate Sample Time Series Data

This utility script generates synthetic time series data based on a combination of sine waves and saves it as a CSV file. The generated data can be used as input for `sines.py`, enabling testing with custom time series data in place of actual data.

## Features
- Generates time series data by combining multiple sine waves with customizable parameters.
- Adds Gaussian noise to each wave to create more realistic data.
- Saves data to a CSV file with `date` and `value` columns, which is compatible with `sines.py`.

## Requirements
- Python 3.x
- Required Python libraries:
  - `numpy`
  - `pandas`

You can install the required libraries using:
```bash
pip install numpy pandas
```

## Usage
Run the script from the command line, providing optional arguments to control various aspects of the generated data.

### Basic Command
```bash
python generate_sample_data.py
```

### Command-line Arguments
| Argument         | Type   | Default         | Description                                              |
|------------------|--------|-----------------|----------------------------------------------------------|
| `--output_file`  | String | `sample_data.csv` | Path to save the generated sample data.                    |
| `--start_date`   | String | `1818-01-01`    | Start date for the generated data in `YYYY-MM-DD` format.|
| `--end_date`     | String | `2021-01-01`    | End date for the generated data in `YYYY-MM-DD` format.  |
| `--num_waves`    | Int    | `5`             | Number of sine waves to combine in the generated data.   |
| `--max_amplitude`| Float  | `150`           | Maximum amplitude for the generated sine waves.          |
| `--max_frequency`| Float  | `0.001`         | Maximum frequency for the generated sine waves.          |
| `--noise_std`    | Float  | `1`             | Standard deviation of noise added to the sine waves.     |

### Example Command
```bash
python generate_sample_data.py --output_file="synthetic_data.csv" --start_date="1800-01-01" --end_date="2020-01-01" --num_waves=7 --max_amplitude=100 --max_frequency=0.002 --noise_std=2
```

This command will generate a synthetic dataset with:
- 7 combined sine waves
- A maximum amplitude of 100
- A maximum frequency of 0.002
- Gaussian noise with a standard deviation of 2

The data will be saved to `synthetic_data.csv`.

## Output Format
The generated CSV file will contain two columns:
- `date`: The date for each data point in `YYYY-MM-DD` format.
- `value`: The combined value of the sine waves at each date, with added noise.

Sample output format:
```
date,value
1800-01-01,23.5
1800-01-02,25.1
1800-01-03,21.8
...
```

## Notes
- This utility is designed to produce synthetic data for testing purposes and is not based on actual or other observational data.
- The script can be adapted to generate various types of time series data by adjusting the parameters.

## License
This script is released under the MIT License. You are free to use, modify, and distribute it for non-commercial purposes.