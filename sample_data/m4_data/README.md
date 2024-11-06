
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

To install the required libraries, run:
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

- **Series Limit**: The global constant `SERIES_LIMIT` allows limiting the number of time series processed, which is useful for testing or partial runs.
- **URLs**: Only monthly data URLs are enabled by default. Other URLs are provided in the script but commented out. Uncomment these URLs if you wish to download and process additional datasets.

### Example File Structure

The `M4Data_transformed` directory will contain files in the following format:
```
~/Downloads/M4Data_transformed/
├── ID_Daily-train.csv
├── ID_Daily-test_combined.csv
└── ...
```

Each file includes columns for `Timestamp` and `Value`.

### Logging

The script includes optional logging to help debug and monitor the processing flow, providing detailed information on each step. To enable the logging, uncomment the logging configuration.

## Using the data

After having run the script to produce the training data and test data, you can use `sines.py` like this with the training data.
```
python3 sines.py --data-file ~/Downloads/M4Data_transformed/M1_Monthly-train.csv --date-col Timestamp --value-col Value --wave-count 0
```

After having comleted a sufficient amount of training, you can stop sines.py and use `extrapolator.py` with the testing data. We can set predict-before and predict-after to zero since we combined the training data with the test data into a test_combined.csv file via the `m4_data_etl.py` script. With the graph produced, you can visually see how the predicted values align with the actual test data.
```
python3 extrapolator.py --data-file ~/Downloads/M4Data_transformed/M1_Monthly-test_combined.csv --date-col Timestamp --value-col Value --predict-before 0 --predict-after 0
```

## Disclaimer

Please note that data downloaded from the M4 GitHub repository is used under the terms specified by the repository owners. Since the M4 repository lacks a specific license, we ensure compliance by not redistributing the data and instead downloading it directly as part of this script.