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



    # Вычисление средней квадратичной ошибки
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
    zx = [[3330.086163, 1376.669510, -390.963485, 370.613997, -162.868954, -351.592406, 94.908800],
         [-65.053599, 1853.874726, 471.129817, -687.025530, -77.420620, 563.524473, 208.715392]]
    print('__________')
    print(type(result.x))
    print(result.x)
    # new_data  = pca.transform([[-1838.14614128, -702.33350026, -126.64866986, 806.69161517, -369.81326759]])
    print('new_data')
    # print(new_data)

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
    PCA(num_comp=0.97)
    # new_cat_model()

f = [[3330.086163,1376.669510,-390.963485,370.613997,-162.868954,-351.592406, 94.908800],
     [-65.053599,1853.874726,471.129817,-687.025530,-77.420620,563.524473,208.715392]]
