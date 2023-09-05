from sklearn.metrics import mean_squared_error
import pandas as pd
from catboost import CatBoostRegressor, Pool

res = pd.read_excel('ML_manufacture_second_all_10_HYPER_Батончики_Супермаркет.xlsx')

def PCA(num_comp):
    from sklearn.decomposition import PCA
    import numpy as np
    # Настройка вывода DataFrame без сокращения колонок
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # Пример набора данных
    X = res[res.columns[6:]]

    print(f'Без нормализации по значениям количество компонент - {num_comp}')
    # print("Входные данные\n", X)

    y = res['Percentage_Sales_rub']

    # Создание объекта PCA с большим количеством компонентов
    # n_components = 10  # Указываем желаемое количество компонентов
    # pca = PCA(n_components=min(X.shape[0], X.shape[1], n_components))

    # Создание объекта PCA с числом компонентов = 2
    pca = PCA(n_components=num_comp, random_state=42)
    pca.set_output(transform='pandas')
    # Применение PCA к набору данных
    X_pca = pca.fit_transform(X)

    # print("Главные компоненты:\n", pca.components_)
    print("Объясненная дисперсия каждой компоненты:\n", pca.explained_variance_ratio_)
    print('Возвращает сингулярные значения, связанные с главными компонентами.')
    print(pca.singular_values_)
    # Вывод преобразованных данных
    print("Преобразованные данные:")
    print('Данные после преобразования методом PCA')
    print(X_pca)
    # print("Оценка ковариационной матрицы:")
    # print(pca.get_covariance())

    print('Размерность после ПСА',X_pca.shape)
    print('Размерность до ПСА', X.shape)
    X_pca_df = pd.DataFrame(X_pca)




    X_pca_df.insert(0, 'Percentage_Sales_rub', y.values.tolist())



    new_cat_model(res_PCA=X_pca_df)


def new_cat_model(res_PCA):


    train = res_PCA.sample(frac=0.8, random_state=42).copy()

    valid = res_PCA[~res_PCA.index.isin(train.index)].copy()
    X_col = res_PCA.columns[1:]


    y_col = 'Percentage_Sales_rub'

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
    from sklearn.metrics import mean_squared_error

    PCA(num_comp=0.99)
    data =[16.25,7.88,-2.57,15.82,-7.62,17.36,4.23,-7.74,3.13,45.01]
    x_d = pd.DataFrame({'y_predict': data})
    res_f = pd.concat([res['Percentage_Sales_rub'],x_d], axis=1)
    print(res_f)
    se = mean_squared_error(res_f['Percentage_Sales_rub'], res_f['y_predict'])
    print(f'Среднеквадратичная ошибка результата между y_true и y_predict: {se}')

    ''''Объясненная дисперсия каждой компоненты:
 [0.48342307 0.23094189 0.11599279 0.06006164 0.03150779 0.02821003
 0.02219619 0.01901849]
Возвращает сингулярные значения, связанные с главными компонентами.
[4825.55784383 3335.30227969 2363.73802995 1700.91427117 1231.94917279
 1165.69687708 1034.00589267  957.13215186]'''