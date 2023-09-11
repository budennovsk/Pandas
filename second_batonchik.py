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

    X_10 = res[res.columns[6:]].copy()

    print('X_10',X_10)
    print(X_10.shape)
    X_row_1= res.iloc[1,6:].to_numpy()
    print('X_row_1',X_row_1)
    print(X_row_1.shape)
    head_df = res.columns[6:].tolist()

    print(f'С нормализацией по значениям количество компонент - {num_comp}')
    print("Входные данные\n", X)

    X_max_Y = pd.DataFrame({'Y_max':[res['Percentage_Sales_rub'].max()]})
    X_mix_Y = pd.DataFrame({'Y_mix': [res['Percentage_Sales_rub'].min()]})

    X_max_all = pd.DataFrame({'Y_max_all': [res[0].max()]})

    X_max_10 = X_10.iloc[:, 0::10].max()

    # Применение среднего значения ко всем остальным столбцам
    other_columns = X_10.columns.difference(X_10.columns[0::10])
    mean_values = X_10[other_columns].mean()


    X_10.loc[:, 0::10] = X_max_10.values


    X_10[other_columns] = mean_values.values
    X_10 = X_10.head(1)


    X_value_mean_max_all = pd.DataFrame(res[res.columns[7:]].mean()).T
    X_value_mean_max = pd.DataFrame(res[res.columns[6:]].mean()).T
    X_value_mean_min = pd.DataFrame(res[res.columns[6:]].mean()).apply(lambda x: x*0.99).T

    df_max_value_all = pd.concat([X_max_all, X_value_mean_max_all], axis=1)
    df_max_value = pd.concat([X_max_Y,X_value_mean_max], axis=1)
    print(df_max_value.shape)
    df_min_value = pd.concat([X_mix_Y, X_value_mean_min], axis=1)
    print(df_max_value)
    print(df_min_value)
    print(df_max_value_all)
    y = res['Percentage_Sales_rub']

    # Создание объекта PCA с числом компонентов = 2
    pca = PCA(n_components=num_comp, random_state=42)
    pca.set_output(transform='pandas')
    # Применение PCA к набору данных
    print("q1")

    X_all_pca = pca.fit_transform(X)
    # print('scokre', pca.score(X_all_pca))
    print('X_all',X_all_pca)
    print("Главные компоненты:\n", pca.components_)
    print("Объясненная дисперсия каждой компоненты:\n", pca.explained_variance_ratio_)
    print('get',pca.get_covariance())
    # Умножение ковариационной матрицы на входные данные
    # new_data = df_max_value_all.to_numpy().dot(pca.get_covariance())

    # КОД ПРЕОБРАЗОВАНИЯ НАДО СДЕЛАТЬ
    # new_data_1 = df_max_value_all.to_numpy().dot(pca.components_.T)
    new_data_11 = X_10.to_numpy().dot(pca.components_.T)
    new_data_1 = pca.components_.dot(X_10.to_numpy().T)
    print('ww1')
    print(new_data_1)
    print(new_data_1.shape)
    print(new_data_1.reshape(1, -1))
    print('shape')
    print(X_10.shape)
    print('pca')
    print(pca.components_.shape)
    print(pca.components_.T.shape)
    print(X_10.T.shape)
    # new_data_2 = X_10.dot(pca.components_.T)
    # new_data_2 = .dot(pca.components_, X_10.T)
    print('Входные данные')
    invers = pd.DataFrame(pca.inverse_transform(new_data_1.reshape(1, -1)))

    print(X_10)
    print('Транспонироване данные ')
    print(invers)
    print("___")
    print(new_data_1)
    print('____')
    # print('new_data_2',new_data_2)
    print('____')
    invers11 = pd.DataFrame(pca.inverse_transform(new_data_11))
    print(invers11)
    """[[ -744.48837536   823.34638541 -2632.80590047 -1186.33496123
    996.97855297    78.66783423 -2355.45658098]]"""
    '''new_data_2 [-808.7232276631614 2676.235108891138 -2161.512019180137
 -1876.434156090446 917.4656514929347 636.2561645378595 -2143.066556245742]
'''
    # X_pca_max= pca.fit_transform((df_max_value[df_max_value.columns[1:]]).T)
    # print('scokre',pca.score(X_pca_max))
    # print("Главные компоненты:\n", pca.components_)
    # print("Объясненная дисперсия каждой компоненты:\n", pca.explained_variance_ratio_)

    # print('q2')
    #
    # X_pca_min = pca.fit_transform((df_min_value[df_min_value.columns[1:]]).T)
    # print("q3")
    # print('Обратный преобразованный набор данных:\n', X_pca_max)
    # # Восстановление максимального значения
    # restored_data_max = pca.inverse_transform(X_pca_max)
    # restored_data_min = pca.inverse_transform(X_pca_min)
    # print('ss',mean_squared_error((df_max_value[df_max_value.columns[1:]]).T,restored_data_max))
    # print(restored_data_max)
    # print('q4')
    # restored_data_df_max = pd.DataFrame(restored_data_max, columns=head_df)
    # print('q5')
    # print('Обратный преобразованный набор данных:\n', restored_data_df_max)
    # # Восстановление минимальное значения
    #
    # restored_data_df_min = pd.DataFrame(restored_data_min, columns=head_df)
    # print('Обратный преобразованный набор данных:\n', restored_data_df_min)
    # print('q6')



    # # Вычисление средней квадратичной ошибки
    # mse = mean_squared_error(X, restored_data_df)
    # print(f'Среднеквадратичная ошибка результата между входными данными и обратными данными: {mse}')
    # # Вычисление средней абсолютной ошибки (MAE)
    # mae = mean_absolute_error(X, restored_data_df)
    # print(f'Среднеквадратичная ошибка результата между входными данными и обратными данными: {mae}')
    #
    # print("Главные компоненты:\n", pca.components_)
    # print("Объясненная дисперсия каждой компоненты:\n", pca.explained_variance_ratio_)
    # print('Возвращает сингулярные значения, связанные с главными компонентами.')
    # print(pca.singular_values_)
    # # Вывод преобразованных данных
    # print("Преобразованные данные:")
    # print('Данные после преобразования методом PCA')
    # print(X_pca)
    # print("Оценка ковариационной матрицы:")
    # print(pca.get_covariance())
    #
    # print('Размерность после ПСА', X_pca.shape)
    # print('Размерность до ПСА', X.shape)
    # X_pca_df = pd.DataFrame(X_pca)
    #
    # X_pca_df.insert(0, 'Percentage_Sales_rub', y.values.tolist())
    #
    # print("Датафрейм после преобразования методом PCA")
    # print(X_pca_df)
    # new_cat_model(res_PCA=X_pca_df, pca=pca, head_df=head_df)

#
# def new_cat_model(res_PCA, pca,head_df):
#     train = res_PCA.sample(frac=0.8, random_state=42).copy()
#
#     valid = res_PCA[~res_PCA.index.isin(train.index)].copy()
#     X_col = res_PCA.columns[1:]
#
#     y_col = 'Percentage_Sales_rub'
#
#     train_pool = Pool(train[X_col], train[y_col])
#     valid_pool = Pool(valid[X_col], valid[y_col])
#
#     model = CatBoostRegressor(
#         iterations=100,
#         learning_rate=0.1,
#         verbose=10,
#         random_state=42
#     )
#     model.fit(train_pool, eval_set=valid_pool)
#
#     comparison = pd.DataFrame({'y_true': res[y_col],
#                                'y_predict': model.predict(res_PCA[X_col])})
#     print(comparison)
#     # Предположим, у вас есть массив с истинными значениями y_true и массив с прогнозируемыми значениями y_pred
#     mse = mean_squared_error(res[y_col], model.predict(res_PCA[X_col]))
#     print(f'Среднеквадратичная ошибка результата между y_true и y_predict: {mse}')
#     # Вычисление средней абсолютной ошибки (MAE)
#     mae = mean_absolute_error(res[y_col], model.predict(res_PCA[X_col]))
#     print(f'Вычисление средней абсолютной ошибки между y_true и y_predict: {mae}')
#
#
#
#
#     test_all(model=model, data=res_PCA, pca=pca, head_df=head_df)
#
#
# def test_all(model, data, pca, head_df):
#     print('_________')
#     print(data)
#     min_v = data[data.columns[1:]].min().tolist()
#     max_v = data[data.columns[1:]].max().tolist()
#     x0 = data[data.columns[1:]].mean().tolist()
#
#     def objective(x):
#         # Прогнозируем значение с помощью модели CatBoost
#         prediction = model.predict([x])
#
#         return -prediction
#
#     result = dual_annealing(objective, x0=x0, bounds=list(zip(min_v, max_v)), maxiter=100, seed=1237)
#     zx = [[3330.086163, 1376.669510, -390.963485, 370.613997, -162.868954, -351.592406, 94.908800],
#          [-65.053599, 1853.874726, 471.129817, -687.025530, -77.420620, 563.524473, 208.715392]]
#     print('__________')
#     print(type(result.x))
#     print(result.x)
#     # new_data  = pca.transform([[-1838.14614128, -702.33350026, -126.64866986, 806.69161517, -369.81326759]])
#     print('new_data')
#     # print(new_data)
#
#     restored_data = pca.inverse_transform(zx)
#
#
#     # Выводим результаты оптимизации
#     print("Максимум:", -result.fun)
#     print("Оптимальные значения переменных:", result.x)
#     print(result)
#     # Выводим результаты оптимизации
#     print("Преобразованный набор данных:")
#     print(restored_data)
#     restored_data_df = pd.DataFrame(restored_data.reshape(2, 100), columns=head_df)
#
#
#     print('Обратный преобразованный набор данных полученный на основании оптимальных значениях переменных в количесве 7 PDA:\n', restored_data_df)
#     print('Входной датафраме')
#     print(res[res.columns[6:]])

if __name__ == '__main__':
    # catboost_model_all()

    # start = time.time()

    # catboost_model_import()

    # end = time.time() - start
    # print(f'Время выполнения: {round(end, 2)}сeк')
    PCA(num_comp=0.97)
    # new_cat_model()
