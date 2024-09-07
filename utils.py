import glob
import os
import shutil
import zipfile

import pandas as pd
from tqdm import tqdm


def unzip_file(file_path, extract_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def delete_contents(extract_path):
    print('Deleting contents...')
    # Check if the directory exists
    if os.path.exists(extract_path):
        # Delete all files and subdirectories in the directory
        for filename in tqdm(os.listdir(extract_path)):
            file_path = os.path.join(extract_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        print(f"The directory '{extract_path}' does not exist.")


def combine_csv_files(extract_path, output_file):
    # Get a list of all CSV files in the subfolders of extract_path
    csv_files = glob.glob(extract_path + '/**/*.csv', recursive=True)
    print(f'Найдено {len(csv_files)} csv-файлов')

    combined_data = pd.DataFrame()
    # Iterate over the CSV files and concatenate them
    for file in tqdm(csv_files):
        df = pd.read_csv(file)
        combined_data = pd.concat([combined_data, df])
        # os.remove(file)
    print(combined_data.shape)

    # Write the combined data to a new CSV file
    print(f'Запись в {output_file}...')
    combined_data.to_csv(output_file, index=False)


def get_percentage_of_missing_values(df):
    df_missing_values = df.isnull().mean() * 100
    df_missing_values = df_missing_values.T.reset_index().T
    df_missing_values.columns = df_missing_values.loc['index', :].values
    df_missing_values.drop('index', axis=0, inplace=True)

    return df_missing_values

def get_percentage_of_zeros(df):
    df_zeros = (df == 0).mean() * 100
    df_zeros = df_zeros.T.reset_index().T
    df_zeros.columns = df_zeros.loc['index', :].values
    df_zeros.drop('index', axis=0, inplace=True)

    return df_zeros