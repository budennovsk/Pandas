from sklearn.metrics import mean_squared_error
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import dual_annealing

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

    result = dual_annealing(objective, x0=x0, bounds=list(zip(min_v, max_v)), maxiter=10000, seed=1237)
    # Выводим результаты оптимизации
    print("Минимум:", result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    # Выводим результаты оптимизации


def catboost_model_import():
    train = res.sample(frac=0.8, random_state=42).copy()

    valid = res[~res.index.isin(train.index)].copy()
    X_col = ['Percentage_Sales_kg_pr', 'Percentage_Sales_Price_reg', 'Kod_TT', 'Percentage_Sales_kg_reg']
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

    x0 = [31.08, -0.13, 978.80, 6.98]
    min_v = [-55.26, -9.63, 970.00, -14.78]
    max_v = [254.14, 9.38, 990.00, 44.43]
    result = dual_annealing(objective, x0=x0, bounds=list(zip(min_v, max_v)), maxiter=10000, seed=1237)
    # Выводим результаты оптимизации
    print("Минимум:", result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    # Выводим результаты оптимизации


if __name__ == '__main__':
    catboost_model_all()
    # catboost_model_import()
