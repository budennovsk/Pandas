import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import dual_annealing
from sklearn.metrics import mean_squared_error

res = pd.read_excel('ML_manufacture_first_all_10_HYPER_Прочие МКИ_Супермаркет.xlsx')

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
    test_all(model=model)


# Определяем функцию для минимизации

def test_all(model):
    def objective(x):
        # Прогнозируем значение с помощью модели CatBoost
        prediction = model.predict([x])

        return -prediction

    x0 = [504.18, -0.51, 3.06, -0.69, 21.86, 11.27, 977.70, -0.01, 22.40, -4.54]
    min_v = [430.98, -18.95, -46.91, -20.00, -49.27, -15.96, 967.00, -0.82, 22.00, -36.84]
    max_v = [504.18, -0.51, 3.06, -0.69, 21.86, 11.27, 977.70, -0.01, 22.40, -4.54]
    result = dual_annealing(objective, x0=x0, bounds=list(zip(min_v, max_v)), maxiter=10000, seed=1237)
    # Выводим результаты оптимизации
    print("Максимум:", result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    # Выводим результаты оптимизации




def catboost_model_import():
    train = res.sample(frac=0.8, random_state=42).copy()

    valid = res[~res.index.isin(train.index)].copy()


    X_col = ['Percentage_Sales_kg_reg', 'Percentage_Sales_Price', 'Percentage_Sales_kg_pr', 'Percentage_Kod_TT']
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
    # proverka(model=model)


# Определяем функцию для минимизации

def test_import(model):


    def objective(x):
        # Прогнозируем значение с помощью модели CatBoost
        prediction = model.predict([x])

        return prediction

    x0 = [3.06,-0.51,21.86,-0.01]
    min_v = [-46.91,-18.95,-49.27,-0.82]
    max_v = [39.71,27.60,210.83,0.82]
    result = dual_annealing(objective,x0=x0, bounds=list(zip(min_v, max_v)), maxiter=10000, seed=1237)
    # Выводим результаты оптимизации
    print("Минимум:", result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    # Выводим результаты оптимизации

def proverka(model):
    x = [504.18, -0.51, 3.06]
    prediction = model.predict([x])
    '''[5.01398637],[5.01398637],[5.01398637]'''
    print(prediction)


if __name__ == '__main__':
    # catboost_model_all()
    catboost_model_import()