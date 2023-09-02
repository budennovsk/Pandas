from sklearn.metrics import mean_squared_error
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import dual_annealing

res = pd.read_excel('ML_manufacture_first_all_10_HYPER_Печенье_Хард-дискаунтер.xlsx')

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
    mse = mean_squared_error(res[y_col], model.predict(res[X_col]))
    print(f'Среднеквадратичная ошибка результата между y_true и y_predict: {mse}')

    test_all( model=model)


# Определяем функцию для минимизации

def test_all(model):


    def objective(x):
        # Прогнозируем значение с помощью модели CatBoost
        prediction = model.predict([x])

        return prediction

    x0 = [293.09,2.53,13.07,2.19,59.05,-10.17,254.50,21.05,2.30,1.67]
    min_v = [262.11,-3.93,-38.68,-4.09,-100.00,-100.00,89.00,12.34,2.00,-33.33]
    max_v = [308.34,13.18,82.34,13.13,648.97,7.19,517.00,40.53,3.00,50.00]
    result = dual_annealing(objective,x0=x0, bounds=list(zip(min_v, max_v)), maxiter=10000, seed=1237)
    # Выводим результаты оптимизации
    print("Минимум:", result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    # Выводим результаты оптимизации




def catboost_model_import():
    train = res.sample(frac=0.8, random_state=42).copy()

    valid = res[~res.index.isin(train.index)].copy()


    X_col = ['Percentage_Sales_kg_pr', 'Sales_Price', 'Kod_TT', 'Percentage_Kod_TT']
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
    mse = mean_squared_error(res[y_col], model.predict(res[X_col]))
    print(f'Среднеквадратичная ошибка результата между y_true и y_predict: {mse}')

    test_import( model=model)


# Определяем функцию для минимизации

def test_import(model):


    def objective(x):
        # Прогнозируем значение с помощью модели CatBoost
        prediction = model.predict([x])

        return -prediction

    x0 = [59.05,293.09,254.50,21.05]
    min_v = [-100,262.11,89.00,12.34]
    max_v = [648.97,293.09,517.00,40.53]
    result = dual_annealing(objective,x0=x0, bounds=list(zip(min_v, max_v)), maxiter=10000, seed=1237)
    # Выводим результаты оптимизации
    print("Максимум:", -result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    # Выводим результаты оптимизации




if __name__ == '__main__':
    # catboost_model_all()
    catboost_model_import()
