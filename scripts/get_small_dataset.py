
import warnings

import numpy as np
import pandas as pd

from config.config import settings
from utils import get_percentage_of_missing_values, get_percentage_of_zeros


def run():
    warnings.filterwarnings("ignore")
    # pd.pandas.set_option('display.max_columns', None)
    # pd.set_option("expand_frame_repr", False)
    # pd.set_option("display.precision", 2)

    source_file = settings.DATASET_PATH + '/df_failure_HDD.csv'

    df = pd.read_csv(source_file)
    df = df.replace(False, np.nan)
    # df['date'] = pd.to_datetime(df['date'])
    # df.set_index('date', inplace=True)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    # df.sort_index(inplace=True)
    print(df)
    print(f'Get percentage of missing values:\n{get_percentage_of_missing_values(df)}\n')

    # Remove empty columns
    empty_columns = df.columns[df.isnull().all()]
    print("\nEmpty columns:", len(empty_columns))
    df = df.drop(empty_columns, axis=1)
    print(df)
    print(f'Get percentage of missing values:\n{get_percentage_of_missing_values(df)}')

    # msno.matrix(df)
    # plt.show()

    # Remove columns with 0 more than 0.1% missing values
    normal_percentage_of_missing_values = 1
    df.fillna(0, inplace=True)
    columns_to_drop = df.columns[
        df.apply(lambda x: pd.to_numeric(x.iloc[0], errors='coerce') < normal_percentage_of_missing_values)]
    columns_to_drop = columns_to_drop.to_list()
    columns_to_drop.remove('failure')
    # print(f'\nGet percentage of missing values:\n{get_percentage_of_missing_values(df)}')
    print(f'Found {len(columns_to_drop)} columns to drop: {columns_to_drop}\n')


    df.drop(columns_to_drop, axis=1, inplace=True)
    # print(f'\nGet percentage of missing values:\n{get_percentage_of_missing_values(df)}')
    print(df)


    # Add types & brandS
    df_types_brands = pd.read_csv(settings.DATASET_PATH + '/Sila-HDD-with-type.csv')
    print(df_types_brands)
    print()
    df_merged = pd.merge(df, df_types_brands, on='model')
    df_merged.insert(3, 'type', df_merged.pop('type'))
    df_merged.insert(4, 'brand', df_merged.pop('brand'))
    print(df_merged)
    print(f'\nGet percentage of missing values:\n{get_percentage_of_missing_values(df_merged)}')
    zeros_percentage = get_percentage_of_zeros(df_merged)
    print(f'Zeros percentage:\n{zeros_percentage}')
    df_merged.to_csv(settings.DATASET_PATH + '/df_failure_HDD_small_types_brands.csv')

    # msno.matrix(df)
    # plt.show()
