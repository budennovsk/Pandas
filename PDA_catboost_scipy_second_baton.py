import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.optimize import dual_annealing

res = pd.read_excel('ML_manufacture_second_all_10_HYPER_Батончики_Супермаркет.xlsx')


def PCA(num_comp):
    from sklearn.decomposition import PCA
    import numpy as np
    # Настройка вывода DataFrame без сокращения колонок
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # Пример набора данных
    X = res[res.columns[6:]]
    head_df = res.columns[6:].tolist()
    print(head_df)
    print(f'С нормализацией по значениям количество компонент - {num_comp}')
    print("Входные данные\n", X)

    y = res['Percentage_Sales_rub']

    # Создание объекта PCA с числом компонентов = 2
    pca = PCA(n_components=num_comp, random_state=42)
    pca.set_output(transform='pandas')
    # Применение PCA к набору данных
    X_pca = pca.fit_transform(X)

    # Восстановление исходных данных
    restored_data = pca.inverse_transform(X_pca)
    restored_data_df = pd.DataFrame(restored_data, columns=head_df)
    print('Обратный преобразованный набор данных:\n', restored_data_df)
    mse = mean_squared_error(X, restored_data_df)
    print(f'Среднеквадратичная ошибка результата между входными данными и обратными данными: {mse}')
    # Вычисление средней абсолютной ошибки (MAE)
    mae = mean_absolute_error(X, restored_data_df)
    print(f'Среднеквадратичная ошибка результата между входными данными и обратными данными: {mae}')

    print("Главные компоненты:\n", pca.components_)
    print("Объясненная дисперсия каждой компоненты:\n", pca.explained_variance_ratio_)
    print('Возвращает сингулярные значения, связанные с главными компонентами.')
    print(pca.singular_values_)
    # Вывод преобразованных данных
    print("Преобразованные данные:")
    print('Данные после преобразования методом PCA')
    print(X_pca)
    print("Оценка ковариационной матрицы:")
    print(pca.get_covariance())

    print('Размерность после ПСА', X_pca.shape)
    print('Размерность до ПСА', X.shape)
    X_pca_df = pd.DataFrame(X_pca)

    X_pca_df.insert(0, 'Percentage_Sales_rub', y.values.tolist())

    print("Датафрейм после преобразования методом PCA")
    print(X_pca_df)
    new_cat_model(res_PCA=X_pca_df, pca=pca, head_df=head_df)


def new_cat_model(res_PCA, pca,head_df):
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
    # Вычисление средней абсолютной ошибки (MAE)
    mae = mean_absolute_error(res[y_col], model.predict(res_PCA[X_col]))
    print(f'Вычисление средней абсолютной ошибки между y_true и y_predict: {mae}')
    test_all(model=model, data=res_PCA, pca=pca, head_df=head_df)


def test_all(model, data, pca, head_df):
    print('_________')
    print(data)
    min_v = data[data.columns[1:]].min().tolist()
    max_v = data[data.columns[1:]].max().tolist()
    x0 = data[data.columns[1:]].mean().tolist()

    def objective(x):
        # Прогнозируем значение с помощью модели CatBoost
        prediction = model.predict([x])

        return -prediction

    result = dual_annealing(objective, x0=x0, bounds=list(zip(min_v, max_v)), maxiter=100, seed=1237)

    restored_data = pca.inverse_transform(result.x)


    # Выводим результаты оптимизации
    print("Максимум:", -result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    # Выводим результаты оптимизации

    restored_data_df = pd.DataFrame(restored_data.reshape(1, 100), columns=head_df)

    #
    print('Обратный преобразованный набор данных полученный на основании оптимальных значениях переменных в количесве 7 PDA:\n', restored_data_df)
    print('Входной датафраме')
    print(res[res.columns[6:]])

if __name__ == '__main__':
    # catboost_model_all()

    # start = time.time()

    # catboost_model_import()

    # end = time.time() - start
    # print(f'Время выполнения: {round(end, 2)}сeк')
    PCA(num_comp=0.97)
    # new_cat_model()

'''[-1355.11419511, -829.87592994, -135.39514395, 172.67546672, -446.00738012,
                                            -507.57524592, 490.67544063, -225.59932253]'''

'''Обратный преобразованный набор данных:
 [ 7.36110814e+02  6.53777264e+02  9.33571392e+02  1.26951094e+03
  1.01490650e+03  6.22503340e+02  1.59792314e+03  2.58248889e+03
  2.84645070e+03  1.41584180e+03 -4.01764793e+00  3.27007650e-01
  5.63041327e+00  8.11229862e+00  5.09512950e+01 -1.69481238e+01
 -1.75442118e+00  6.12721465e+00  1.59773738e-01  5.15000808e+00
  4.11081467e+01  1.21088538e+02  4.73963813e+01  2.80436887e+01
 -4.76735659e+00  2.98190604e+01  2.83830106e+01 -4.38969753e+00
  9.02904630e+00  2.63432058e+01 -1.67485504e+00 -8.96437091e+00
 -8.14874547e-01 -1.24304497e+00  4.74857786e-01 -2.62150101e+00
 -2.94920271e+00  8.36465827e-02 -5.63076505e-01 -1.30336054e-01
  7.06628349e+01  7.27085552e+01  8.66674816e+00 -2.04805328e+01
 -1.82545303e+02  6.93938475e+02  4.01929607e+00 -1.54503157e+01
  1.85964676e+01 -7.32286559e+00 -9.46603555e-01  4.12578506e+00
  6.46166668e+00  5.76929461e+00  1.18569367e+00  1.17050254e+01
 -4.79477355e+00  6.49535382e+00  6.79725583e-01  9.75243832e+00
  9.84087594e+02  9.81142239e+02  9.84801041e+02  9.78576075e+02
  9.79330066e+02  8.65044954e+01  1.01349990e+02  4.25582594e+02
  3.81063717e+02  2.53732002e+01  6.95344645e-01  6.70556514e-01
  7.81492904e-01  5.98732426e-01  5.80261011e-01  9.12156327e+00
  4.38119450e+00 -1.23591526e+00  2.94045103e+00  3.54439854e+00
  2.39531821e+01  6.15349271e+00  1.04937019e+01  6.72006323e+00
  5.75068812e+00  5.39764208e+00  3.48949889e+00  9.45222113e-01
  9.93887795e-01  1.11865554e+00  5.63816631e+00  1.47653426e+00
 -1.48238005e+00 -2.81806422e-01 -1.46903847e+01  8.57273331e-01
  1.13498302e+00 -7.09001452e-01  0.00000000e+00  4.01835818e-03]
Максимум: [41.40270093]
Оптимальные значения переменных: [-1355.11419511  -829.87592994  -135.39514395   172.67546672
  -446.00738012  -507.57524592   490.67544063  -225.59932253]'''