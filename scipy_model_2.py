import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize

res = pd.read_excel('graph__Sales_Price.xlsx')




def catboost_model_1():
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

    print('model', model)
    print('y_true', comparison)
    print(res['Sales_Price'].tolist())
    model.save_model('trained_model.cbm')


# Определяем функцию для минимизации

def test(X_col_res, y_col_res):
    print('zeka')
    # Создаем обученную модель CatBoost
    model = CatBoostRegressor()
    model.load_model('trained_model.cbm')

    # Определяем функцию для минимизации
    def objective(x):
        print('zeka1')
        print(x, 'zeka2')
        # Прогнозируем значение с помощью модели CatBoost
        prediction = model.predict([x])
        print('a1', prediction)
        print('qwe')
        print('a2', prediction[0])
        return prediction[0]

    # Задаем начальное приближение
    x0 = X_col_res.values.tolist()[0]  # Примерный вектор начального приближения
    print('x0',x0)
    # Вызываем функцию minimize для оптимизации
    result = minimize(objective, x0, method='Nelder-Mead')


    # Выводим результаты оптимизации
    print("Минимум:", result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    print(X_col_res.values.tolist())
    print(y_col_res)
    """7.12113682e+02  7.52440046e-01  6.97939291e+00 -1.31631473e-01
  3.10846913e+01  1.00547792e+01  9.78800000e+02  1.29029284e-03
  2.35000000e+01  5.70063994e-01"""
    '''[ 7.477e+02  7.524e-01  6.979e+00 -1.316e-01  3.108e+01
                  1.005e+01  9.788e+02  1.290e-03  2.350e+01  5.701e-01]'''
    '''[ 7.121e+02  7.524e-01  6.979e+00 -1.316e-01  3.108e+01
             1.005e+01  9.788e+02  1.290e-03  2.350e+01  5.701e-01]'''

if __name__ == '__main__':
    catboost_model_1()

    print(res.shape)
    for i in range(1,len(res)+1):
        y_col_res = res.iloc[i-1:i, 6:7]

        X_col_res = res.iloc[i-1:i, 7:]
        test(X_col_res=X_col_res, y_col_res=y_col_res)
        print('count')

