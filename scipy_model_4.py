from sklearn.metrics import mean_squared_error
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import dual_annealing
import time

res = pd.read_excel('ML_manufacture_first_all_10_HYPER_Батончики_Супермаркет.xlsx')

y_col_res = res.iloc[1:, 6:7]

X_col_res = res.iloc[1:, 7:]


def catboost_model_all():
    train = res.sample(frac=0.8, random_state=42).copy()

    valid = res[~res.index.isin(train.index)].copy()
    X_col = res.columns[7:]
    print(X_col)

    y_col = 'Percentage_Sales_rub'
    print(y_col)
    train_pool = Pool(train[X_col], train[y_col])
    valid_pool = Pool(valid[X_col], valid[y_col])

    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        verbose=10,
        random_state=42
    )
    model.fit(train_pool, eval_set=valid_pool)

    comparison = pd.DataFrame({'y_true': res[y_col],
                               'y_predict': model.predict(res[X_col])})
    print(comparison)
    # Предположим, у вас есть массив с истинными значениями y_true и массив с прогнозируемыми значениями y_pred
    mse = mean_squared_error(res[y_col], model.predict(res[X_col]))
    print(f'Среднеквадратичная ошибка результата между y_true и y_predict: {mse}')
    print(model.get_feature_importance(prettified=True))
    test_all(model=model)


# Определяем функцию для минимизации

def test_all(model):
    def objective(x):
        # Прогнозируем значение с помощью модели CatBoost
        prediction = model.predict([x])

        return prediction

    x0 = [802.41, 0.75, 6.98, -0.13, 31.08, 10.05, 978.80, 0.00, 23.50, 0.57]

    min_v = [712.11, -7.41, -14.78, -9.63, -55.26, -6.49, 970.00, -0.61, 21.00, -12.50]

    max_v = [880.82, 16.65, 44.43, 9.38, 254.14, 89.56, 990.00, 0.92, 24.00, 9.09]

    result = dual_annealing(objective, x0=x0, bounds=list(zip(min_v, max_v)), maxiter=100, seed=1237)
    # Выводим результаты оптимизации
    print("Минимум:", result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    # Выводим результаты оптимизации


def catboost_model_import():

    train = res.sample(frac=0.8, random_state=42).copy()

    valid = res[~res.index.isin(train.index)].copy()
    X_col = ['Percentage_Sales_kg_pr', 'Percentage_Sales_Price_reg', 'Percentage_Sales_kg_reg','Percentage_Sales_Price','Percentage_Kod_TT', 'Sales_Price','Percentage_Sales_Price_pr', 'SKU_ID']
    # X_col = ['SKU_ID']
    # X_col = ['Percentage_Sales_Price']
    print(X_col)

    y_col = 'Percentage_Sales_rub'
    print(y_col)
    train_pool = Pool(train[X_col], train[y_col])
    valid_pool = Pool(valid[X_col], valid[y_col])

    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        verbose=10,
        random_state=42
    )
    model.fit(train_pool, eval_set=valid_pool)

    comparison = pd.DataFrame({'y_true': res[y_col],
                               'y_predict': model.predict(res[X_col])})
    print(comparison)
    # Предположим, у вас есть массив с истинными значениями y_true и массив с прогнозируемыми значениями y_pred
    mse = mean_squared_error(res[y_col], model.predict(res[X_col]))
    print(f'Среднеквадратичная ошибка результата между y_true и y_predict: {mse}')

    test_import(model=model)


# Определяем функцию для минимизации

def test_import(model):
    def objective(x):
        # Прогнозируем значение с помощью модели CatBoost
        prediction = model.predict([x])

        return prediction

    x0 = [31.08, -0.13, 978.80, 6.98, 0.75, 0.00, 802.41, 10.05, 23.50]
    min_v = [-55.26, -9.63, 970.00, -14.78,-7.41, -0.61,712.11, -6.49, 21.00]
    max_v = [254.14, 9.38, 990.00, 44.43,16.65, 0.92,802.41, 10.05, 24.00]
    # x0= [23.50]
    # min_v = [21.00]
    # max_v = [24.00]
    # x0 = [0.75]
    # min_v =[-7.41]
    # max_v = [16.65]
    result = dual_annealing(objective, x0=x0, bounds=list(zip(min_v, max_v)), maxiter=10000, seed=1237)
    # Выводим результаты оптимизации
    print("Минимум:", result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    # Выводим результаты оптимизации

def PCA():
    from sklearn.decomposition import PCA
    import numpy as np

    # Пример набора данных
    X = res[res.columns[7:]]
    y = res['Percentage_Sales_rub']
    print(y)
    print(X.shape)
    print(X)

    # Создание объекта PCA с числом компонентов = 2
    pca = PCA(n_components=9)


    # Применение PCA к набору данных
    X_pca = pca.fit_transform(X)

    # Вывод преобразованных данных
    print("Преобразованные данные:")
    print(X_pca)
    print(X_pca.shape)
    X_pca_df = pd.DataFrame(X_pca)
    print(X_pca_df)

    X_pca_df.insert(0, 'Percentage_Sales_rub', y.values.tolist())

    print(X_pca_df, 'ddd')
    new_cat_model(res_PCA=X_pca_df)


def new_cat_model(res_PCA):
    print('PCA')
    print(res_PCA)

    train = res_PCA.sample(frac=0.8, random_state=42).copy()

    valid = res_PCA[~res_PCA.index.isin(train.index)].copy()
    X_col = res_PCA.columns[1:]
    print(X_col)

    y_col = 'Percentage_Sales_rub'
    print(y_col)
    train_pool = Pool(train[X_col], train[y_col])
    valid_pool = Pool(valid[X_col], valid[y_col])

    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        verbose=10,
        random_state=42
    )
    model.fit(train_pool, eval_set=valid_pool)

    comparison = pd.DataFrame({'y_true': res[y_col],
                               'y_predict': model.predict(res_PCA[X_col])})
    print(comparison)
    # Предположим, у вас есть массив с истинными значениями y_true и массив с прогнозируемыми значениями y_pred
    mse = mean_squared_error(res[y_col], model.predict(res_PCA[X_col]))
    print(f'Среднеквадратичная ошибка результата между y_true и y_predict: {mse}')




if __name__ == '__main__':
    # catboost_model_all()

    # start = time.time()

    # catboost_model_import()

    # end = time.time() - start
    # print(f'Время выполнения: {round(end, 2)}сeк')
    PCA()
    # new_cat_model()