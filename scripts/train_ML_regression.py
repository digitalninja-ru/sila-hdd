import os
import time
import warnings

import joblib
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from config.config import settings

def run():
    os.environ["OPENBLAS_NUM_THREADS"] = "24"

    warnings.filterwarnings("ignore")
    # pd.pandas.set_option('display.max_columns', None)
    # pd.set_option("expand_frame_repr", False)
    # pd.set_option("display.precision", 2)
    start_time = time.time()


    source_file = settings.DATASET_PATH +  '/dataset_regression.csv'
    df = pd.read_csv(source_file, index_col='Unnamed: 0')
    df.sort_index(inplace=True, ascending=True)
    print(df)

    # Scaling
    x_train = df.iloc[:, 3:]
    y_train = df['failure']
    print(f'\nx_train:\n{x_train}')
    print(f'\ny_train:\n{y_train}')
    scaler = StandardScaler()
    x_train_normalized = scaler.fit_transform(x_train)
    print(f'\nx_train_normalized.shape: {x_train_normalized.shape}')

    # Using PCA
    # Find the number of components to keep 95% of the variance
    # pca = PCA()
    # pca.fit(x_train_normalized)
    # cumsum = np.cumsum(pca.explained_variance_ratio_)
    # plt.plot(cumsum)
    # plt.grid()
    # plt.show()
    # plt.plot(cumsum[:30])
    # plt.grid()
    # plt.show()
    # exit()

    pca = PCA(n_components=0.99)  # keep 95% of the variance
    x_train_pca = pca.fit_transform(x_train_normalized)
    print(f'x_train_pca.shape: {x_train_pca.shape}')

    # Шаг 4: Разделение данных на тренировочную и тестовую выборки
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x_train_pca, y_train,
                                                                                test_size=0.2,
                                                                                shuffle=True, random_state=42)
    print(f"x_train_split.shape: {x_train_split.shape}", f"y_train_split.shape: {y_train_split.shape}")
    print(f"x_test_split.shape: {x_test_split.shape}", f"y_test_split.shape: {y_test_split.shape}\n")

    # # Создадим и обучим разные модели регрессии
    # models = {
    #     "KNeighborsRegressor": KNeighborsRegressor(),
    #     # "GradientBoostingRegressor": GradientBoostingRegressor(),
    #     # "LinearRegression": LinearRegression(),
    #     # "Ridge": Ridge(),
    #     # "Lasso": Lasso(),
    #     # "ElasticNet": ElasticNet(),
    #     # # "RandomForestRegressor": RandomForestRegressor(),
    #     # # "SVR": SVR(),
    # }
    #
    # for name, model in models.items():
    #     model.fit(x_train_split, y_train_split)
    #     y_pred = model.predict(x_test_split)
    #     mse = mean_squared_error(y_test_split, y_pred)
    #     print(f"Модель: {name}, Mean Squared Error: {mse}")

    # Tuning KNeighborsRegressor by Optuna
    def objective(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan'])

        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)
        model.fit(x_train_split, y_train_split)
        y_pred = model.predict(x_test_split)
        mse = mean_squared_error(y_test_split, y_pred)

        # Сохраняем модель, если она лучшая
        if trial.number == 0 or mse < study.best_value:
            joblib.dump(model, settings.MODELS_PATH + "/best_KNeighborsRegressor.pkl")

        return mse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    print(f"Лучшие параметры: {study.best_params}")
    print(f"Лучшее MSE: {study.best_value}")


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Script execution time: {round(execution_time/60, 2)} min")
