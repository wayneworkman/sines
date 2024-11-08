# SILSO data readme

## Usage

This command will download and transform the sunspot data, and save it within this directory.
```
python3 download.py
```

You can then use sines.py to model this data like such:
```
python3 sines.py --data-file sample_data/sunspots/SN_d_tot_V2.0.csv --date-col date --value-col sunspot --set-negatives-zero after_sum --project-dir ~/sunspots
```

You can use the extrapolator.py to extrapolate the model out into the future or the past.
```
python3 extrapolator.py --data-file sample_data/sunspots/SN_d_tot_V2.0.csv --value-col sunspot --date-col date --set-negatives-zero after_sum --project-dir ~/sunspots --predict-before 50 --predict-after 100
```


## Data Website

https://www.sidc.be/SILSO/home

## Explanation of Changes

1. **Date Format Transformation**
   - **Original Data**: The date in the original data is represented in three separate columns: Year, Month, and Day.
   - **Processed Data**: The processed data combines these columns into a single date column in `YYYY-MM-DD` format, using a standardized date format for easier data handling and analysis.

2. **Exclusion of Unused Columns**
   - **Original Data**: The raw data includes additional columns:
     - Fractional year representation of the date (Column 4).
     - Standard deviation of the daily sunspot number (Column 6).
     - Count of observations used to compute the daily value (Column 7).
     - Definitive/provisional indicator (Column 8).
   - **Processed Data**: These columns are omitted, retaining only the date and daily total sunspot number columns. The exclusion of unused columns simplifies the dataset, focusing solely on the time series values essential for modeling.

3. **Handling of Missing Data (Sunspot Value = -1)**
   - **Original Data**: In the original dataset, a sunspot value of `-1` indicates missing data for that day.
   - **Processed Data**: The processed data does not modify or remove these values. A missing value (`-1`) remains `-1` in the processed dataset, retaining this indicator for consistency. This preserves the integrity of missing data information without any alteration.

4. **Rolling 27-Day Moving Average Applied to Sunspot Counts**
   - **Original Data**: The raw sunspot data presents daily values without any smoothing or aggregation, meaning each day is reported individually based on raw observations.
   - **Processed Data**: A 27-day moving average is applied to the daily sunspot counts. This averaging smooths out daily fluctuations, providing a more stable trend over time. The moving average replaces each sunspot count with the average of that day and the preceding 26 days, resulting in a smoothed dataset that highlights general trends and cycles.

5. **File Format and Structure**
   - **Original Data**: The original data file is in a semicolon-separated (`;`) text format, without headers, which requires manual interpretation of columns.
   - **Processed Data**: The processed data is saved as a CSV file with a comma separator, including column headers ("date" and "sunspot") for clarity. This format change makes the data easier to load into data analysis tools and more user-friendly.

### Summary of Changes
The processed data saved by the script:
- Consolidates date information into a single column with a `YYYY-MM-DD` format.
- Excludes columns not directly relevant to sunspot number counts.
- Applies a 27-day moving average to the sunspot counts to smooth the data.
- Changes the delimiter from semicolons to commas.
- Adds headers to the columns for clarity and ease of use in downstream applications.

These changes are intended to make the dataset more accessible for non-commercial analytical purposes while preserving the integrity and utility of the original data. The processed data should not be used commercially, in accordance with the CC BY-NC4.0 license terms from SILSO.


## Data License

SILSO data is under CC BY-NC4.0 license (https://goo.gl/PXrLYd) which means you can :

Share - copy and redistribute the material in any medium or format
Adapt - remix, transform, and build upon the material

As long as you follow the license terms:

Attribution - You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
NonCommercial - You may not use the material for commercial purposes.
No additional restrictions - You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
No additional restrictions - You may not apply legal terms or technological meas


## Data Credits

Source: WDC-SILSO, Royal Observatory of Belgium, Brussels