import pprint

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize

res = pd.read_excel('ML_manufacture_first_all_10_HYPER_Батончики_Супермаркет.xlsx')

FUNC_MIN =[]
OPTIMAL_VALUES = []
# y_col_res = res.iloc[:1, 6:7]
# print(y_col_res)


# X_col_res = res.iloc[:1, 6:]
# print(X_col_res)

def catboost_model_1():
    train = res.sample(frac=0.8, random_state=42).copy()

    valid = res[~res.index.isin(train.index)].copy()
    X_col = res.columns[6:]
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

    print('model', model)
    print('y_true', comparison)
    print(res['Sales_Price'].tolist())
    # model.save_model('trained_model.cbm')
    return model

# Определяем функцию для минимизации

def test(X_col_res, y_col_res, model):
    print('zeka')
    print('ho',X_col_res)
    # Создаем обученную модель CatBoost
    # model = CatBoostRegressor()
    # model.load_model('trained_model.cbm')

    # Определяем функцию для минимизации
    def objective(x):
        print('zeka1')
        print(x, 'zeka2')
        # Прогнозируем значение с помощью модели CatBoost
        prediction = model.predict([x])
        print('a1', prediction)
        print('qwe')
        print('a2', prediction)
        return -prediction

    # Задаем начальное приближение
    x0 = X_col_res.values.tolist()[0]  # Примерный вектор начального приближения
    print('x0',x0)
    # Вызываем функцию minimize для оптимизации method='Nelder-Mead'
    result = minimize(objective, x0, method='Nelder-Mead')

    FUNC_MIN.append(result.fun)
    OPTIMAL_VALUES.append(list(result.x))
    # Выводим результаты оптимизации
    print("Минимум:", result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    print(X_col_res.values.tolist())
    print('res',y_col_res)
    '''y_true       y_true  y_predict
0  16.175060  15.505207
1  -8.375100  -7.316421
2  -2.005420  -1.696215
3  27.355908  12.616009
4  -7.460414  -6.481244
5  16.721038  16.002730
6   4.215919   1.998064
7  -8.094811  -7.022287
8   3.081211   3.049732
9  45.610962  42.539152'''

if __name__ == '__main__':
    model=catboost_model_1()

    for i in range(1,len(res)+1):
        X_col_res = res.iloc[i-1:i, 6:]
        y_col_res = res.iloc[i-1:i, 6:7]
        test(X_col_res=X_col_res, y_col_res=y_col_res, model=model)
    print('____')
    print(FUNC_MIN)
    for i in OPTIMAL_VALUES:
        print(i)

