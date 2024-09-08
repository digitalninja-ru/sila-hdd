import os
import time
import joblib
import warnings

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config.config import settings

def run():
    os.environ["OPENBLAS_NUM_THREADS"] = "24"

    warnings.filterwarnings("ignore")
    pd.pandas.set_option('display.max_columns', None)
    pd.set_option("expand_frame_repr", False)
    pd.set_option("display.precision", 2)
    start_time = time.time()


    source_file = settings.DATASET_PATH + '/dataset_regression.csv'
    df = pd.read_csv(source_file, index_col='Unnamed: 0')
    df.sort_index(inplace=True, ascending=True)
    # print(df[df['failure'] == df['failure'].max()])
    # print(df)
    # exit()

    # Group by serial_number
    # hdd = ['W300R8BM', 32]
    hdd = ['ZHZ650PV', 320]
    serial_number = hdd[0]
    observation = hdd[1]
    n_component = 27
    print('Not enough observations to process..') if observation < n_component else None

    # Get sample for predict
    df = df.loc[df['serial_number'] == serial_number].fillna(0)
    new_hdd = df.iloc[:observation, 3:]
    y = df['failure'][observation-1: observation]

    # Scaling
    scaler = StandardScaler()
    new_hdd_normalized = scaler.fit_transform(new_hdd)
    print(f'\nx_train_normalized.shape: {new_hdd_normalized.shape}')

    pca = PCA(n_components=27)  # keep 99% of the variance
    new_hdd_pca = pca.fit_transform(new_hdd_normalized)
    print(f'x_train_pca.shape: {new_hdd_pca.shape}')

    # Загрузка лучшей модели
    best_model = joblib.load(settings.MODELS_PATH + "/best_KNeighborsRegressor.pkl")
    y_new_pred = best_model.predict(new_hdd_pca)
    print(f'\nThe disk {serial_number} will die in : {int(y_new_pred[-1:])} days')
    print(f'The disk {serial_number} died in : {int(y)} days')
