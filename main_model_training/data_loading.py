import pandas as pd
import os

# Define the folder containing the datasets
data_folder = os.path.dirname(__file__)

# List of dataset filenames
files = {
    'attractions': 'Attractions.xlsx',
    'hotels': 'Hotel.xlsx',
    'restaurants': 'restaurants.xlsx',
    'transport': 'Transport.xlsx',
    'travel_costs': 'travel_costs.xlsx',
    'rentals_hospitals': 'emergency_services.xlsx',
    'emergency_services': 'emergency_services.xlsx',
}

def load_datasets(folder, files_dict):
    data = {}
    for key, filename in files_dict.items():
        path = os.path.join(folder, filename)
        try:
            data[key] = pd.read_excel(path)
            #print(f"Loaded {filename} with shape {data[key].shape}")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    return data

if __name__ == "__main__":
    datasets = load_datasets(data_folder, files)
    # Example: print first few rows of each dataset
    for name, df in datasets.items():
        #print(f"\n{name.upper()} sample:")
        print(df.head())
