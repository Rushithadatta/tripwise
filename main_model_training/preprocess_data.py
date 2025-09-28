import pandas as pd
import os
from data_loading import load_datasets, files, data_folder

def preprocess_dataframe(df):
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()
    # Standardize column names
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    # Drop completely empty columns
    df = df.dropna(axis=1, how='all')
    # Fill missing values with suitable defaults using .loc
    for col in df.columns:
        if df[col].dtype == 'O':  # Object type (string)
            df.loc[:, col] = df[col].fillna('Unknown')
        else:
            fill_value = df[col].median() if not df[col].isnull().all() else 0
            df.loc[:, col] = df[col].fillna(fill_value)
    return df

def preprocess_all_datasets(datasets):
    processed = {}
    for name, df in datasets.items():
        processed[name] = preprocess_dataframe(df)
        #print(f"Preprocessed {name}: {processed[name].shape} shape")
    return processed

if __name__ == "__main__":
    datasets = load_datasets(data_folder, files)
    processed_datasets = preprocess_all_datasets(datasets)
    # Example: print info for each processed dataset
    for name, df in processed_datasets.items():
        print(f"\n{name.upper()} info:")
        print(df.info())
        print(df.head())
