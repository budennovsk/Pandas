import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize


res = pd.read_excel('ML_manufacture_first.xlsx')


def catboost_model_1():
    train = res.sample(frac=0.8).copy()
    valid = res[~res.index.isin(train.index)].copy()
    X_col = res.columns[1:]
    print(X_col)

    y_col = 'Percentage_Sales_rub'
    print(y_col)
    train_pool = Pool(train[X_col], train[y_col])
    valid_pool = Pool(valid[X_col], valid[y_col])

    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        verbose=10
    )
    model.fit(train_pool, eval_set=valid_pool)

    comparison = pd.DataFrame({'y_true': res[y_col],
                               'y_predict': model.predict(res[X_col])})

    print('model', model)
    print('y_true', comparison)
    print(res['Sales_Price'].tolist())
    model.save_model('trained_model.cbm')
    get_model_catboost(valid[X_col])

def get_model_catboost(data):
    # Загружаем сохраненную модель
    model = CatBoostRegressor()
    model.load_model('trained_model.cbm')

    # Используем модель для прогнозирования
    predictions = model.predict(data)
    # print('model', model)
    # print('y_true', predictions)


def objective(x):
    return x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]

def constraint1(x):
    return x[0]*x[1]*x[2]*x[3]-25.0

def constraint2(x):
    sum_eq = 40.0
    for i in range(4):
        sum_eq = sum_eq - x[i]**2
    return sum_eq

# initial guesses
n = 4
x0 = np.zeros(n)
x0[0] = 1.0
x0[1] = 5.0
x0[2] = 5.0
x0[3] = 1.0

# show initial objective
# print('Initial Objective: ' + str(objective(x0)))

# optimize
b = (1.0,5.0)
bnds = (b, b, b, b)
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
cons = ([con1,con2])
solution = minimize(objective,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)
x = solution.x

# # show final objective
# print('Final Objective: ' + str(objective(x)))
#
# # print solution
# print('Solution')
# print('x1 = ' + str(x[0]))
# print('x2 = ' + str(x[1]))
# print('x3 = ' + str(x[2]))
# print('x4 = ' + str(x[3]))


X = [712.11,728.98,745.86,762.73,779.60, 796.47,813.34,830.21,847.08,863.95]
y = [7.27,7.27,7.27,7.27,7.27,7.12,4.75,4.75,4.75,4.75]
# Определяем функцию для минимизации

def test():
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
        print('a1',prediction)
        print('qwe')
        print('a2',prediction[0])
        return prediction[0]

    # Задаем начальное приближение
    x0 = np.array([471.8725567009666, 515.2593399235844, 485.4430997013325, 511.4350106416183, 532.1794396115855, 530.7833947584547, 523.0896406712268, 490.8294772550552, 549.934178384257, 430.9825214419985])  # Примерный вектор начального приближения

    # Вызываем функцию minimize для оптимизации
    result = minimize(objective, x0)

    # Выводим результаты оптимизации
    print("Минимум:", result.fun)
    print("Оптимальные значения переменных:", result.x)


if __name__ == '__main__':
    catboost_model_1()
    test()
