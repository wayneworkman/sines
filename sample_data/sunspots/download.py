import requests
import pandas as pd
from datetime import datetime

DATA_URL = "http://www.sidc.be/silso/INFO/sndtotcsv.php"
OUTPUT_FILE = "SN_d_tot_V2.0.csv"

def download_and_process_sunspot_data(url, output_file):
    response = requests.get(url)
    data = response.text.strip().splitlines()
    sunspot_data = []

    for line in data:
        fields = line.split(';')
        if len(fields) < 5:
            continue
        try:
            year, month, day = int(fields[0]), int(fields[1]), int(fields[2])
            sn_value = int(float(fields[4].strip()))
            sunspot_data.append([datetime(year, month, day), sn_value])
        except (ValueError, IndexError):
            continue

    sunspot_df = pd.DataFrame(sunspot_data, columns=['date', 'sunspot'])
    sunspot_df.set_index('date', inplace=True)
    sunspot_df['sunspot'] = sunspot_df['sunspot'].rolling(window=27, min_periods=1).mean()
    sunspot_df.to_csv(output_file)
    print(f"Sunspot data saved to {output_file}")

if __name__ == "__main__":
    download_and_process_sunspot_data(DATA_URL, OUTPUT_FILE)
