import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
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


    # Max values
    X_10 = res[res.columns[6:]].copy()
    X_max_10 = X_10.iloc[:, 0::10].max()
    other_columns = X_10.columns.difference(X_10.columns[0::10])
    # Генерация случайных чисел в диапазоне от 1 до 10
    random_values = np.random.uniform(1.01, 1.1, size=90)

    mean_values = X_10[other_columns].mean() * random_values

    X_10.loc[:, 0::10] = X_max_10.values
    X_10[other_columns] = mean_values.values
    X_10 = X_10.head(1)

    # Min values
    X_10_min = res[res.columns[6:]].copy()
    X_min_10 = X_10_min.iloc[:, 0::10].min()
    other_columns_min = X_10_min.columns.difference(X_10_min.columns[0::10])
    # Генерация случайных чисел в диапазоне от 1 до 10
    random_values_min = np.random.uniform(1.01, 1.1, size=90)

    mean_values_min = X_10_min[other_columns].mean() * random_values_min

    X_10_min.loc[:, 0::10] = X_min_10.values
    X_10_min[other_columns_min] = mean_values_min.values
    X_10_min = X_10_min.head(1)




    print(f'С нормализацией по значениям количество компонент - {num_comp}')


    y = res['Percentage_Sales_rub']

    # Создание объекта PCA с числом компонентов = 2
    pca = PCA(n_components=num_comp, random_state=42, svd_solver='full', whiten=True)
    pca.set_output(transform='pandas')
    # Применение PCA к набору данных
    X_pca = pca.fit_transform(X)
    projected_data_1 = pca.transform(X_10)
    projected_data_1_min = pca.transform(X_10_min)

    # Восстановление исходных данных
    restored_data = pca.inverse_transform(X_pca)
    restored_data_1 = pca.inverse_transform(projected_data_1)
    restored_data_df = pd.DataFrame(restored_data)
    restored_data_df_1 = pd.DataFrame(restored_data_1)
    restored_data_df_1_min = pca.inverse_transform(projected_data_1_min)

    rever_1 = pca.inverse_transform(projected_data_1)

    print('min')
    print(X_10_min.iloc[:, 0::10])
    print(pd.DataFrame(restored_data_df_1_min[0][0::10].reshape(1, -1)))
    print(mean_absolute_error(X_10_min.iloc[:, 0::10], pd.DataFrame(restored_data_df_1_min[0][0::10].reshape(1, -1))))

    print('_______________')

    print('max')
    print(X_10.iloc[:, 0::10])
    print(pd.DataFrame(rever_1[0][0::10].reshape(1, -1)))
    print(mean_absolute_error(X_10.iloc[:, 0::10], pd.DataFrame(rever_1[0][0::10].reshape(1, -1))))

    print('_______________')
    print('min')

    print(X_10_min)
    print(pd.DataFrame(restored_data_df_1_min))
    # Вычисление средней квадратичной ошибки

    mae = mean_absolute_error(X_10_min, restored_data_df_1_min)
    print(f'Среднеквадратичная ошибка результата между входными данными и обратными данными: {mae}')

    print('_______________')

    print(X_10)
    print(pd.DataFrame(rever_1))
    # Вычисление средней квадратичной ошибки
    mse = mean_squared_error(X, restored_data_df)
    print(f'Среднеквадратичная ошибка результата между входными данными и обратными данными: {mse}')
    # Вычисление средней абсолютной ошибки (MAE)
    mae = mean_absolute_error(X, restored_data_df)
    print(f'Среднеквадратичная ошибка результата между входными данными и обратными данными: {mae}')

    print("______________")
    print(X_10)
    print(restored_data_df_1)
    # Вычисление средней квадратичной ошибки
    mse = mean_squared_error(X_10, restored_data_df_1)
    print(f'Среднеквадратичная ошибка результата между входными данными и обратными данными: {mse}')
    # Вычисление средней абсолютной ошибки (MAE)
    mae = mean_absolute_error(X_10, restored_data_df_1)
    print(f'Среднеквадратичная ошибка результата между входными данными и обратными данными: {mae}')

    X_pca_df = pd.DataFrame(X_pca)

    X_pca_df.insert(0, 'Percentage_Sales_rub', y.values.tolist())

    print("Датафрейм после преобразования методом PCA")
    print(X_pca_df)
    new_cat_model(res_PCA=X_pca_df, pca=pca, head_df=head_df, limit=projected_data_1, limit_min=projected_data_1_min)


def new_cat_model(res_PCA, pca,head_df,limit, limit_min):
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




    test_all(model=model, data=res_PCA, pca=pca, head_df=head_df, limit=limit, limit_min=limit_min)


def test_all(model, data, pca, head_df, limit, limit_min):
    print('_________')
    # print(data)
    # min_v = data[data.columns[1:]].min().tolist()
    # print(min_v)
    # max_v = data[data.columns[1:]].max().tolist()

    min_v = limit_min.values.tolist()[0]
    max_v = limit.values.tolist()[0]
    # max_v = [0.00001083312685948725, 0.009964706370184661, 0.09071123549483899, 0.11834902394704239, 0.1355875277676379, 0.051508807604525364, 0.355620540358731, 0.06184438435012604]
    x0 = data[data.columns[1:]].mean().tolist()
    print(min_v)
    print('max')
    print(max_v,)
    print(pd.DataFrame(pca.inverse_transform(min_v)).T)
    print(pd.DataFrame(pca.inverse_transform(max_v)).T)
    print(list(zip(min_v, max_v)))
    def objective(x):
        # Прогнозируем значение с помощью модели CatBoost
        prediction = model.predict([x])

        return -prediction

    result = dual_annealing(objective, x0=x0, bounds=list(zip(max_v, min_v)), maxiter=100, seed=1237)


    restored_data = pca.inverse_transform(zx)


    # Выводим результаты оптимизации
    print("Максимум:", -result.fun)
    print("Оптимальные значения переменных:", result.x)
    print(result)
    # Выводим результаты оптимизации
    print("Преобразованный набор данных:")
    print(restored_data)
    restored_data_df = pd.DataFrame(restored_data.reshape(2, 100), columns=head_df)


    print('Обратный преобразованный набор данных полученный на основании оптимальных значениях переменных в количесве 7 PDA:\n', restored_data_df)
    print('Входной датафраме')
    print(res[res.columns[6:]])

if __name__ == '__main__':
    # catboost_model_all()

    # start = time.time()

    # catboost_model_import()

    # end = time.time() - start
    # print(f'Время выполнения: {round(end, 2)}сeк')
    PCA(num_comp=2)
    # new_cat_model()
#
# f = [[3330.086163,1376.669510,-390.963485,370.613997,-162.868954,-351.592406, 94.908800],
#      [-65.053599,1853.874726,471.129817,-687.025530,-77.420620,563.524473,208.715392]]
