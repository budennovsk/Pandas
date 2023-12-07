# import matplotlib.pyplot as plt
# import numpy as np
#
# # Предположим, что у вас есть некоторые данные для построения графика
# x = np.linspace(1, 730, 730)
# y = np.sin(x)
#
# plt.plot(x, y)
#
# # Создайте новые метки для оси x
# new_xticks = np.arange(0, 830, 100)
# print(len(new_xticks))
#
# new_labels = new_xticks % 365
# print(new_labels)
# new_labels[1:][new_labels[1:] == 0] = 365  # заменить все нули, кроме первого, на 365
# print(new_labels)
# c =[0]+[100,200,300,365]*2
# print(len(c))
# plt.xticks(new_xticks,c)
#
# plt.show()
# def testing():
#     new_xticks = np.arange(0, 730, 73)
#     new_label1 = new_xticks % 365
#     print(new_label1)
#     new_labels =  np.where(new_label1 % 365 == 35, 365, new_xticks) # вычисляем остаток от деления на 365
#
#     print(new_labels)
#     new_labels[1:][new_labels[1:] == True] = 365  # заменяем все нули, кроме первого, на 365
#     print(new_labels)
# testing()
#
# c =[[0]+[100,200,300,365]*2]
# print(c)
#
# import pandas as pd
#
# # Предположим, что 'df' - это ваш DataFrame, а 'day_of_year' - столбец с номерами дней в году
# df = pd.DataFrame({'day_of_year': range(1, 731)})  # Пример DataFrame
#
# # Ваш список интервалов
# data = [(10, 24), (47, 61), (62, 76), (280, 294), (302, 316), (378, 392), (418, 432), (426, 440), (644, 658), (668, 682)]
#
# # Создаем новый столбец 'data_promo' и заполняем его нулями
# df['data_promo'] = 0
#
# # Проходим по списку интервалов и устанавливаем '1' там, где 'day_of_year' попадает в интервал
# for start, end in data:
#     df.loc[(df['day_of_year'] >= start) & (df['day_of_year'] <= end), 'data_promo'] = 1
#
# print(df[9:25])
# df = df.reset_index().merge(df_1year, how='left', on='day_of_year').set_index('index')
import pandas as pd

# df = pd.DataFrame({'promo_merge': [0,0,0,4,5,6,0,0,0,0,0,2,3,4,5,0,0,0,0,0,0,0,0,0,0,0,7,8,9,10]})
#
#
#
# result = pd.DataFrame({'result':[0,2,3,4,5,6,7,8,9,10]})

# import pandas as pd
# import numpy as np
#
# # Ваш исходный DataFrame
# df = pd.DataFrame({'promo_merge': [0,0,0,4,5,6,0,0,0,0,0,2,3,4,5,0,0,0,0,0,0,0,0,0,0,0,7,8,9,10]})
# print(df.shape)
#
# # Разделите DataFrame на 3 части и сбросьте индексы
# dfs = [group.reset_index(drop=True) for _, group in df.groupby(np.arange(len(df)) // 10)]
#
# # Получите ненулевые значения из каждого блока
# non_zero_values = [block[block['promo_merge'] != 0] for block in dfs]
#
# # Объедините ненулевые значения
# result = pd.concat(non_zero_values)
#
# print(result)

# import pandas as pd
# import numpy as np
#
# # Исходный DataFrame
# df = pd.DataFrame({'promo_merge': [0,0,0,4,5,6,0,0,0,0,1,0,3,4,5,0,0,0,0,0,0,0,0,4,0,0,7,8,0,10]})
# df['count'] =[19,19,19,3,2,1,19,19,19,19,1,19,1,3,2,19,19,19,19,19,19,19,19,3,19,19,1,1,19,1]
# print(df)
# # Разделите DataFrame на 3 части и сбросьте индексы
# dfs = [group.reset_index(drop=True) for _, group in df.groupby(np.arange(len(df)) // 10)]
#
# # Индексы нулей в первой части
# zero_indices = dfs[0][dfs[0]['promo_merge'] == 0].index
# zero_indices_1 = dfs[0][dfs[0]['count'] == 0].index
#
# # Замените нули в первой части значениями из второй и третьей частей
# for i in range(1, len(dfs)):
#     replace_indices = dfs[i][dfs[i]['promo_merge'] != 0].index
#     common_indices = np.intersect1d(zero_indices, replace_indices)
#     dfs[0].loc[common_indices, 'promo_merge'] = dfs[i].loc[common_indices, 'promo_merge'].values
#     zero_indices = dfs[0][dfs[0]['promo_merge'] == 0].index
#
# print(dfs[0])
# for i in range(1, len(dfs)):
#     replace_indices = dfs[i][dfs[i]['count'] != 0].index
#     common_indices = np.intersect1d(zero_indices_1, replace_indices)
#     dfs[0].loc[common_indices, 'count'] = dfs[i].loc[common_indices, 'count'].values
#     zero_indices_1 = dfs[0][dfs[0]['count'] == 0].index
# print('_______________________')

# import pandas as pd
# import numpy as np
#
# # Исходный DataFrame
# df = pd.DataFrame({'count': [19,19,19,3,2,1,19,19,19,19,1,19,1,3,2,19,19,19,19,19,19,19,19,3,19,19,1,1,19,1]})
#
# # Разделите DataFrame на 3 части
# dfs = np.array_split(df, 3)
#
# # Создаем список для хранения результатов
# result = []
#
# # Цикл по каждому индексу
# for i in range(10):  # мы знаем, что каждый блок имеет 10 строк
#     # Получаем текущее значение из каждого блока
#     values = [dfs[j].iloc[i]['count'] for j in range(3)]
#     # Если значения не одинаковы, выбираем минимальное значение
#     if len(set(values)) > 1:
#         result.append(min(values))
#     else:
#         # Если значения одинаковы, сохраняем значение
#         result.append(values[0])
#
# # Уменьшаем размер DataFrame до 10 строк и добавляем результат
# df = df.iloc[:10]
# df['res'] = result
#
# print(df)
# import numpy as np
# from catboost import CatBoostRegressor
#
# # Предположим, что у нас есть временной ряд
# time_series = np.random.normal(size=(1000,))
# time_series = time_series[:-7].copy()
# y_true = time_series[-7:]
#
#
# # Мы будем использовать 30 последних значений для прогнозирования каждого из следующих 7 значений
# X = np.array([time_series[i-30:i] for i in range(30, len(time_series)-7)])
# Y = np.array([time_series[i:i+7] for i in range(30, len(time_series)-7)])
#
# # Обучаем отдельную модель CatBoost для каждого шага прогнозирования
# models = []
# for i in range(7):
#     model = CatBoostRegressor(loss_function='RMSE', verbose=False)
#     model.fit(X, Y[:, i])
#     models.append(model)
#
# # Прогнозируем следующие 7 значений
# input_data = time_series[-30:]
# predictions = [model.predict(input_data.reshape(1, -1)) for model in models]
# print('pred',predictions)
# print('y_true',y_true)

# from sklearn.ensemble import GradientBoostingRegressor
# import numpy as np
# time_series = np.random.normal(size=(1000,))
#
# # Мы будем использовать 30 последних значения для прогнозирования следующих 7 значений
# X = np.array([time_series[i-30:i] for i in range(30, len(time_series)-7)])
# Y = np.array([time_series[i:i+7] for i in range(30, len(time_series)-7)])
#
# # Initiate the model
# model = GradientBoostingRegressor()
#
# # Fit the model
# model.fit(X, Y)
#
# # Make predictions
# input_data = time_series[-30:]
# predictions = model.predict(input_data.reshape(1, -1))
# print(predictions)
# # Получаем важность признаков
# importances = model.get_feature_importance()
#
# # Нормализуем важности признаков
# importances = importances / sum(importances)
#
# # Выводим на экран нормализованные важности признаков
# for feature_name, importance in zip(X_train.columns, importances):
#     print(f'{feature_name}: {importance:.4f}')

import pandas as pd
import numpy as np

# создаем DataFrame
df = pd.DataFrame({
    'eee': pd.Categorical(['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']),
    'ccc': pd.Categorical(['c', 'd', 'c', 'd', 'c', 'd', 'c', 'd']),
    'other_column': np.random.normal(size=8),
    'target_column': np.random.choice([0, 1], size=8),
})

print(df)

cat_features = [df.columns.get_loc(name) for name in df.columns.tolist()]
print(cat_features)

import numpy as np

x = np.array(1)
for i in x:
    print(i)