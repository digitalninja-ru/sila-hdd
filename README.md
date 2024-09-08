# Sila HDD (Предсказание отказов оборудования)

## Запуск проект

Создать папки:
- ./data/archive 
- ./data/dataset 
- ./data/models
- ./data/zipdata

В zipdata помещаем данные из открытых источников (https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)

Запускаем по порядку пункты 1-5

1. Get failure serial numbers
2. Get Small Dataset
3. Preparing dataset
4. Train ML regression
5. Predict ML regression


## Технологии
Python 3.11
Scikit-learn 1.5.1
Optuna 4.0.0