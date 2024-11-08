# M4 Data ETL Script

This script downloads, processes, and structures time series data from the M4 competition's GitHub repository. It is specifically tailored to handle files related to different frequencies such as daily, hourly, monthly, etc., as specified in the metadata (`M4-info.csv`). The script creates timestamps based on each series' start date and frequency, providing structured output in the form of individual CSV files for each series.

## Overview of the Script

- **Data Download**: The script fetches the required files directly from the M4 GitHub repository and places them into a designated `M4Data` directory within the user's `Downloads` folder.
- **Metadata Handling**: The script relies on `M4-info.csv` for series metadata (such as start date and frequency) to reconstruct accurate timestamps.
- **ETL Processing**:
  - **Training Data**: Processes each time series in the training data files, generating a timestamped CSV for each series.
  - **Test Data**: For each test series, the script combines the corresponding training and test data (if available) and produces a continuous timestamped CSV file.
  - **Output**: All processed data is saved to `M4Data_transformed`, where each series has its own CSV file.

### Important Notes on Data Usage

This script does not distribute the M4 competition data. Instead, it directly downloads the data from the original source, the M4 GitHub repository, as the repository does not provide a license. This ensures compliance with distribution policies, as the data is retrieved directly by users and transformed locally on their systems.

## Requirements

- Python 3.x
- Required Python libraries:
  - `requests`
  - `pandas`
  - `argparse`
  - `logging`
  
You can install the required libraries using:
```bash
pip install requests pandas
```

## How to Run the Script

1. **Download the Script**: Place the script in a desired location on your machine.
2. **Execute the Script**:
   ```bash
   python m4_data_etl.py
   ```
3. **Output Location**:
   - Downloaded data will be placed in `~/Downloads/M4Data`.
   - Processed CSV files will be in `~/Downloads/M4Data_transformed`.

## Configuration

- **Series Limit**: The global constant `SERIES_LIMIT` allows limiting the number of time series processed, which is useful for testing or partial runs. Adjust this value at the top of the script as needed.
  ```python
  SERIES_LIMIT = 1  # Adjust this to process more or fewer series
  ```
- **URLs**: Only monthly data URLs are enabled by default. Other URLs are provided in the script but commented out. Uncomment these URLs if you wish to download and process additional datasets.
  ```python
  urls = [
      # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Train/Daily-train.csv",
      # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Test/Daily-test.csv",
      # ...
      "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Train/Monthly-train.csv",
      # ...
      "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Test/Monthly-test.csv",
      # ...
  ]
  ```

### Example File Structure

The `M4Data_transformed` directory will contain files in the following format:
```
~/Downloads/M4Data_transformed/
├── ID_Monthly-train.csv
├── ID_Monthly-test_combined.csv
└── ...
```

Each file includes columns for `Timestamp` and `Value`.

### Logging

The script uses Python's `logging` module to provide detailed information and error messages during execution. Logs include:

- Creation of directories for waves and run parameters.
- Saving of individual wave parameters.
- Generation of the combined wave.
- Splitting and saving of training and testing data.
- Any errors encountered during execution.

To enable detailed logging, uncomment the logging configuration in the script:
```python
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
```

Logs are output to the console with timestamps and log levels (INFO, DEBUG, ERROR).

## Using the Data

After running the script to produce the training and test data, you can use `sines.py` and `extrapolator.py` as follows:

### Using with `sines.py`
```
# Create the first three waves without FFT
python3 sines.py --data-file ~/Downloads/M4Data_transformed/M1_Monthly-train.csv --date-col Timestamp --value-col Value --wave-count 3 --project-dir ~/M1_Monthly

# Create other waves with FFT
python3 sines.py --data-file ~/Downloads/M4Data_transformed/M1_Monthly-train.csv --date-col Timestamp --value-col Value --wave-count 20 --project-dir ~/M1_Monthly --use-fft-initialization

# Create 5 final waves without FFT.
python3 sines.py --data-file ~/Downloads/M4Data_transformed/M1_Monthly-train.csv --date-col Timestamp --value-col Value --wave-count 25 --project-dir ~/M1_Monthly
```

### Using with `extrapolator.py`
After completing sufficient training, you can use the extrapolator with the combined test data:
```
python3 extrapolator.py --data-file ~/Downloads/M4Data_transformed/M1_Monthly-test_combined.csv --date-col Timestamp --value-col Value --predict-before 0 --predict-after 0 --project-dir ~/M1_Monthly
```

With the graph produced, you can visually see how the predicted values align with the actual test data.


## Error Handling

Enhanced error handling ensures that:

- **HTTP Requests**: The script checks the status code of each download request and logs an error if a download fails.
- **Directory Operations**: The script verifies the creation of required directories and logs any failures.
- **Data Processing**: The script handles missing metadata and logs warnings for any inconsistencies.
- **Timestamp Generation**: The script prevents datetime overflow by checking against a maximum allowable date.

If invalid arguments or unexpected scenarios are encountered, the script will log appropriate error messages and terminate gracefully.

## Disclaimer

Please note that data downloaded from the M4 GitHub repository is used under the terms specified by the repository owners. Since the M4 repository lacks a specific license, we ensure compliance by not redistributing the data and instead downloading it directly as part of this script.

## License

This script is released under the MIT License. You are free to use, modify, and distribute it.

