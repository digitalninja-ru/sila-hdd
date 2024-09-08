import glob

import pandas as pd
from pandas import read_csv
from tqdm import tqdm

from config.config import settings
from utils import unzip_file


def run():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Example usage
    source_path = settings.ZIP_DATA_PATH
    destination_path = settings.DATASET_PATH
    extract_path = settings.ARCHIVE_PATH

    # # Extract zip files
    zip_files = glob.glob(settings.ZIP_DATA_PATH + '/*.zip')
    print(f'\nНайдено {len(zip_files)} zip-файлов')
    for zip in zip_files:
        print(f'unzip {zip}')
        unzip_file(zip, extract_path)

    # # Get unique serial numbers & models failures HDD
    list_models = []
    list_serial_numbers = []
    csv_files = glob.glob(extract_path + '/**/*.csv', recursive=True)
    print(f'Найдено {len(csv_files)} csv-файлов')

    for file in tqdm(csv_files):
        df = pd.read_csv(file)
        df_unique = df[df['failure'] == 1]
        unique_serial_numbers = df_unique['serial_number'].unique()
        # print(f"{len(failure_serial_numbers)}")
        list_serial_numbers.extend(unique_serial_numbers)
        # print(f"len list_failure_serial_numbers: {len(list_failure_serial_numbers)}: {list_failure_serial_numbers}")
        unique_model = df_unique['model'].unique()
        # print(f"{len(failure_serial_numbers)}")
        list_models.extend(unique_model)

    # print(f'len list_serial_numbers: {len(list_serial_numbers)}')
    list_unique_serial_numbers = list(set(list_serial_numbers))
    # print(f"len list_unique_serial_numbers {len(list_unique_serial_numbers)}: {list_unique_serial_numbers}")
    df_unique_serial_numbers = pd.DataFrame(list_unique_serial_numbers, columns=['serial_number'])
    df_unique_serial_numbers.to_csv(settings.DATASET_PATH + '/unique_serial_numbers.csv', index=False)

    # print(f'len list_models: {len(list_models)}')
    list_unique_models = list(set(list_models))
    # print(f"len list_unique_models {len(list_unique_models)}: {list_unique_models}")
    df_unique_models = pd.DataFrame(list_unique_models, columns=['model'])
    df_unique_models.to_csv(settings.DATASET_PATH + '/unique_models.csv', index=False)

    # Collect dataset with failure HDD
    # df_unique_serial_numbers = pd.read_csv(settings.DATASET_PATH + '/unique_models.csv')
    # print(f"{len(df_unique_serial_numbers)} unique_serial_numbers: {df_unique_serial_numbers['serial_number']}")

    df_failure_HDD = pd.DataFrame()
    for file in tqdm(csv_files):
        df = pd.read_csv(file)
        df_filtered = df[df['serial_number'].isin(df_unique_serial_numbers['serial_number'])]
        # print(f'{file}: len df_filtered {len(df_filtered)}')
        df_failure_HDD = pd.concat([df_failure_HDD, df_filtered])
        # print(f'len df_failure_HDD {len(df_failure_HDD)}')

    # # Write the combined data to a new CSV file
    print(f'Save to ./data/dataset/df_failure_HDD.csv...')
    df_failure_HDD.to_csv(settings.DATASET_PATH + '/df_failure_HDD.csv')
    print(df_failure_HDD)

def runMock():
    cvs = read_csv(settings.DATASET_PATH + '/df_failure_HDD.csv')

    print('Get unique serial numbers & models failures HDD')
    print('Collect dataset with failure HDD.')
    print('')
    print(f'Save to ./data/dataset/df_failure_HDD.csv...')
    print('')
    print(cvs)
    print('')
