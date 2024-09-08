import warnings

import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config.config import settings


def run():
    warnings.filterwarnings("ignore")
    # pd.pandas.set_option('display.max_columns', None)
    # pd.set_option("expand_frame_repr", False)
    # pd.set_option("display.precision", 2)

    source_file = settings.DATASET_PATH + '/df_failure_HDD_small_types_brands.csv'
    df = pd.read_csv(source_file, index_col='Unnamed: 0')
    df.insert(1, 'failure', df.pop('failure'))
    # df['date'] = pd.to_datetime(df['date'])
    # df.set_index('date', inplace=True)
    # df.drop('Unnamed: 0', axis=1, inplace=True)
    # df.sort_index(inplace=True)
    print(df)

    # Label encoding
    le = LabelEncoder()
    df['model'] = le.fit_transform(df['model'])
    df['type'] = le.fit_transform(df['type'])
    df['brand'] = le.fit_transform(df['brand'])
    df['capacity_bytes'] = le.fit_transform(df['capacity_bytes'])
    df['month'] = pd.to_datetime(df['date']).dt.month
    df.insert(3, 'month', df.pop('month'))
    # df['num_sn'] = le.fit_transform(df['serial_number'])
    print(df)
    # df.to_csv(settings.DATASET_PATH + '/dataset.csv')


    # marking Y for regression
    dataset_regression = pd.DataFrame()
    list_unique_serial_numbers = df['serial_number'].unique()
    for serial_number in tqdm(list_unique_serial_numbers):
        hdd =  df.loc[df['serial_number'] == serial_number]
        hdd['failure'] = range(len(hdd)-1, -1, -1)
        dataset_regression = pd.concat([dataset_regression, hdd])
    print(dataset_regression)
    dataset_regression.to_csv(settings.DATASET_PATH + '/dataset_regression.csv')

    # Group by serial_number
    # grouped_df = df.groupby('serial_number')
    # print(grouped_df.get_group('W300T02T'))
    # grouped_by_serial_number = df.loc[df['serial_number'] == 'W300T02T'].fillna(0)
    # print(f'\n{grouped_by_serial_number}')
    # grouped_by_serial_number = df.loc[df['serial_number'] == 'WD-WX71A74PAYDF'].fillna(0)
    # print(f'\n{grouped_by_serial_number}')
    # grouped_by_serial_number = df.loc[df['serial_number'] == 'WD-WX71A74PAYDF'].fillna(0)
    # print(f'\n{grouped_by_serial_number}')

def runMock():
    cvs1 = read_csv(settings.DATASET_PATH + '/df_failure_HDD_small_types_brands.csv')
    cvs2 = read_csv(settings.DATASET_PATH + '/dataset_regression.csv')

    print("le = LabelEncoder()")
    print("df['model'] = le.fit_transform(df['model'])")
    print("df['type'] = le.fit_transform(df['type'])")
    print("df['brand'] = le.fit_transform(df['brand'])")
    print("df['capacity_bytes'] = le.fit_transform(df['capacity_bytes'])")
    print("df['month'] = pd.to_datetime(df['date']).dt.month")
    print('')

    print(cvs1)
    print('')
    print(cvs2)
