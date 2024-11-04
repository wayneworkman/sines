import requests
import pandas as pd
from pathlib import Path
from datetime import timedelta
import logging

# Global constant to limit the number of series to process
SERIES_LIMIT = 5  # Adjust this to process more or fewer series

# Configure logging for detailed output
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# URLs to download (only daily files and info file are active, others are commented for future use)
urls = [
    # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Train/Daily-train.csv",
    # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Test/Daily-test.csv",
    # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Train/Hourly-train.csv",
    "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Train/Monthly-train.csv",
    # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Train/Quarterly-train.csv",
    # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Train/Weekly-train.csv",
    # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Train/Yearly-train.csv",
    # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Test/Hourly-test.csv",
    "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Test/Monthly-test.csv",
    # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Test/Quarterly-test.csv",
    # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Test/Weekly-test.csv",
    # "https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/Test/Yearly-test.csv",
]

# Always download the info file. This contains the metadata required to construct the datetime stamps for the data values.
urls.append("https://github.com/Mcompetitions/M4-methods/raw/refs/heads/master/Dataset/M4-info.csv")

# Determine user's Downloads folder cross-platform
downloads_dir = Path.home() / "Downloads"
data_dir = downloads_dir / "M4Data"
transformed_dir = downloads_dir / "M4Data_transformed"
data_dir.mkdir(exist_ok=True)
transformed_dir.mkdir(exist_ok=True)

# Function to download files
def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download {url}")

# Download each file into M4Data
for url in urls:
    filename = url.split("/")[-1]
    download_path = data_dir / filename
    print(f"Downloading {filename}...")
    download_file(url, download_path)

# Load M4-info.csv to retrieve metadata for datetime reconstruction
info_file_path = data_dir / "M4-info.csv"
m4_info_df = pd.read_csv(info_file_path)

# Create a dictionary for quick metadata lookup
m4_info_dict = m4_info_df.set_index("M4id").to_dict(orient="index")

# Function to create a datetime index based on frequency and starting date
def create_datetime_index(start_date_str, frequency, length):
    logging.debug(f"Starting create_datetime_index with start_date_str: {start_date_str}, frequency: {frequency}, length: {length}")
    
    # Parse start date and set maximum allowable date
    start_date = pd.to_datetime(start_date_str, dayfirst=True)
    max_date = pd.Timestamp('2262-04-11')
    dates = []
    current_date = start_date
    
    # Log the maximum allowable date boundary
    logging.debug(f"Max date boundary set to: {max_date}")

    for i in range(length):
        # Check for datetime overflow
        if current_date > max_date:
            logging.warning(f"Reached maximum allowable datetime at iteration {i}. Stopping date generation.")
            break
        
        # Append current date and log each date addition
        dates.append(current_date)
        logging.debug(f"Appended date {current_date} at iteration {i}")
        
        # Increment date based on specified frequency
        if frequency == 1:         # Yearly
            current_date += pd.DateOffset(years=1)
        elif frequency == 4:       # Quarterly
            current_date += pd.DateOffset(months=3)
        elif frequency == 12:      # Monthly
            current_date += pd.DateOffset(months=1)
        elif frequency == 52:      # Weekly
            current_date += timedelta(weeks=1)
        elif frequency == 7:       # Daily
            current_date += timedelta(days=1)
        elif frequency == 24:      # Hourly
            current_date += timedelta(hours=1)
        else:
            # Log the unexpected frequency and raise an error
            logging.error(f"Unexpected frequency encountered: {frequency}.")
            raise ValueError(f"Unexpected frequency: {frequency}")
    
    # Log completion and number of dates generated
    logging.info(f"Generated {len(dates)} dates successfully in create_datetime_index.")
    return dates

# Updated process_file function to use full data for train files and continuous timestamps for combined files
def process_file(file_path, is_test=False):
    df = pd.read_csv(file_path)
    base_name = file_path.stem  # e.g., "Daily-train"
    print(f"Processing file: {file_path.name} ({'Test' if is_test else 'Train'})")
    
    for i, row in df.iterrows():
        if i >= SERIES_LIMIT:
            break  # Stop processing after SERIES_LIMIT entries
        
        series_id = row[0]
        series_data = row[1:].dropna()  # Drop NaN values
        if series_id not in m4_info_dict:
            print(f"  Warning: Metadata not found for series {series_id}")
            continue
        
        # Retrieve metadata, including Horizon for test data
        metadata = m4_info_dict[series_id]
        start_date = metadata["StartingDate"]
        frequency = metadata["Frequency"]
        horizon = metadata["Horizon"]
        
        # Determine appropriate length for datetime index
        if is_test:
            # For test files, use Horizon as length to limit combined range
            datetime_index = create_datetime_index(start_date, frequency, len(series_data))
        else:
            # For train files, use the full series length
            datetime_index = create_datetime_index(start_date, frequency, len(series_data))
        
        # Create DataFrame for series data
        series_df = pd.DataFrame({"Timestamp": datetime_index, "Value": series_data.values[:len(datetime_index)]})
        
        if not is_test:
            # Save full training series with datetime index
            output_file = transformed_dir / f"{series_id}_{base_name}.csv"
            series_df.to_csv(output_file, index=False)
            print(f"  Saved training series: {output_file.name}")
        
        else:
            # For test data, ensure continuous timestamping with train data if available
            train_file = transformed_dir / f"{series_id}_{base_name.replace('-test', '-train')}.csv"
            if train_file.exists():
                train_series = pd.read_csv(train_file)
                combined_length = len(train_series) + len(series_df)
                continuous_datetime_index = create_datetime_index(start_date, frequency, combined_length)
                
                # Concatenate train and test series while using continuous timestamps
                combined_series = pd.concat([train_series, series_df], ignore_index=True)
                combined_series["Timestamp"] = continuous_datetime_index
                output_file = transformed_dir / f"{series_id}_{base_name}_combined.csv"
                combined_series.to_csv(output_file, index=False)
                print(f"  Saved combined series: {output_file.name}")
            else:
                print(f"  Warning: Training file not found for series {series_id}, expected at {train_file}")
    
    print(f"Completed processing for {file_path.name}\n")

# Separate function to process train and test files in two steps
def process_files_in_two_steps():
    # First pass: Process all training files
    print("Processing all training files first...")
    for file_path in data_dir.iterdir():
        if "train" in file_path.name:
            process_file(file_path, is_test=False)

    # Second pass: Process all test files
    print("\nProcessing all test files after training files are ready...")
    for file_path in data_dir.iterdir():
        if "test" in file_path.name:
            process_file(file_path, is_test=True)

# Execute the two-step processing
process_files_in_two_steps()
print("Download and transformation complete.")

