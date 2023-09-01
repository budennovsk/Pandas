import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize

res = pd.read_excel('ML_manufacture_first_all_10_HYPER_Батончики_Супермаркет.xlsx')
# res = pd.read_excel('ML_manufacture_first_all_10_HYPER_Печенье_Хард-дискаунтер.xlsx')
OPTIMAL_VALUES = []
FUNC_MIN = []
y_col_res = res.iloc[1:, 6:7]

X_col_res = res.iloc[1:, 7:]

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
    # model.save_model('trained_model.cbm')
    test(X_col_res=X_col_res, y_col_res=y_col_res, model=model)


# Определяем функцию для минимизации

def test(X_col_res, y_col_res, model):
    print('zeka')
    # Создаем обученную модель CatBoost
    # model = CatBoostRegressor()
    # model.load_model('trained_model.cbm')

    # Определяем функцию для минимизации
    def objective(x):

        # Прогнозируем значение с помощью модели CatBoost
        prediction = model.predict([x])

        return -prediction

    # Определяем функцию, которая вычисляет градиент

    # 4.160778729833449
    # Задаем начальное приближение
    # ограничения на X (min, max)
    # Батончики
    bnds = [(712, 880.82), (-7.41, 16.65), (-14.78, 44.43), (-9.63, 9.38), (-55.26, 254.14), (-6.49, 89.56), (970, 990), (-0.61, 0.92), (21.00, 24.00), (-12.50, 9.09)]
    # Печенье дискаунтер
    # bnds = [(262.11, 308.34), (-3.93, 13.38), (-38.68, 82.34), (-4.09, 13.13), (-100.00, 648.97), (-100.00, 7.19), (89.00, 517.00), (12.34, 40.53), (2.00, 3.00), (-33.33, 50.00)]

    # Примерный вектор начального приближения средние значения X
    # Батончики
    # x0 = [802.41, 0.7524400460224019, 6.979392914176508, -0.1316314732304839, 31.08469126909226, 10.05477921313387, 978.8, 0.001290292842637675, 23.5, 0.570063993946219]
    # x0 = [712.11, -7.41, -14.78 , -9.63, -55.26, -6.49, 970, -0.61, 21.00, -12.50]
    x0 = [880.82, 16.65, 44.43 , 9.38, 254.14, 89.56, 990, 0.92, 24.00, 9.09]

    # Печенье дискаунтер

    # x0 = [293.09, 2.53, 13.07, 2.19, 59.05, -10.17, 254.50, 21.05, 2.30, 1.67]
    # Вызываем функцию minimize для оптимизации method='Nelder-Mead', method='BFGS', method='L-BFGS-B'
    result = minimize(objective, x0, method='Nelder-Mead', tol=1e-6)

    # Выводим результаты оптимизации
    print("Максимум:", result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    # Выводим результаты оптимизации


if __name__ == '__main__':
    catboost_model_1()


    # print(res.shape)
    # for i in range(1, len(res) + 1):
    #     y_col_res = res.iloc[i - 1:i, 6:7]
    #
    #     X_col_res = res.iloc[i - 1:i, 7:]
    #     test(X_col_res=X_col_res, y_col_res=y_col_res)
#         print('count')
# for d in FUNC_MIN:
#     print(d)
# for g in OPTIMAL_VALUES:
#     print(g)
