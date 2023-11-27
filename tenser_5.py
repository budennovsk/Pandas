import numpy as np

random_numbers = np.random.randint(10)

print(random_numbers)

from keras.optimizers import Adam
import keras.layers
from keras.optimizers import Adamax
from keras.optimizers import Adadelta
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import keras.preprocessing
from keras.preprocessing.sequence import TimeseriesGenerator

# import numpy as np
# from keras.src.utils.timeseries_dataset import timeseries_dataset_from_array
#
# # Создание данных
# data = np.array([[i] for i in range(100)])
# targets = np.random.rand(100, 5)
#
# # Параметры датасета
# sequence_length = 10
# batch_size = 1
#
#
# # Создание датасета временных рядов
# dataset = timeseries_dataset_from_array(
#     data=data,
#     targets=targets,
#     sequence_length=sequence_length,
#     batch_size=batch_size)
#
# for batch in dataset:
#     inputs, targets = batch
#     print("Inputs shape:", inputs.shape)
#     print(inputs)
#     print("Targets shape:", targets.shape)
#     print(targets)
#     break

# sales_data_list = []  # создаём пустой список
#
# # предположим, у вас есть цикл, который генерирует sales_data
# for i in range(10):  # замените этот цикл на ваш реальный код
#     sales_data = np.random.rand(720, 3)  # замените эту строку на ваш реальный код
#     sales_data_list.append(sales_data)  # добавляем sales_data в список
#
# # после того, как все sales_data добавлены в список, объединяем их
# combined_data = np.hstack(sales_data_list)
# print(combined_data.shape)
#
# # теперь combined_data имеет форму (720, 30)
# sales_1 = [[1,1,1],[2,2,2],[3,3,3]]
#
# sales_2 = [[4,4,4],[5,5,5],[6,6,6]]
#
# sales_3 = [[7,7,7],[8,8,8],[9,9,9]]
# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
#
# # Ваши исходные данные
# sales_1 = [[1,1,1],[2,2,2],[3,3,3]]
# sales_2 = [[4,4,4],[5,5,5],[6,6,6]]
# sales_3 = [[7,7,7],[8,8,8],[9,9,9]]
#
# # Объедините все данные в один список
# data = [sales_1, sales_2, sales_3]
#
# # Преобразуйте список в numpy массив
# data = np.array(data)
# print('all',data.shape)
# # Подготовьте данные
# prepared_data = []
# for i in range(data.shape[0]):
#     current_set = []
#     for j in range(data.shape[1]-1):
#         current_set.append(data[i, j:j+2, :])
#     prepared_data.append(current_set)
# prepared_data = np.array(prepared_data)
#
# # Разделите данные на входные данные (X) и целевую переменную (y)
# X = prepared_data[:, :, 0, :]  # все данные на шаге t
#
# y = prepared_data[:, :, 1, 0]  # только 'Regular_Sales_Promo_Sales' на шаге t+1
#
# # Определите модель
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(2, 3)))
# model.add(Dense(1))
#
# # Скомпилируйте модель
# model.compile(optimizer='adam', loss='mse')
# print(X.shape)
# print(X)
# print(y.shape)
# print(y)
# # Обучите модель
# model.fit(X, y, epochs=2, verbose=0)
#
#
# # Предскажите значения с помощью модели
# sales_pred = model.predict(X)
#
# # Выведите предсказанные значения
# print(sales_pred)
#
# def split_data(df, test_split=0.15):
#     n = int(len(df) * test_split)
#     train, test = df[:-n], df[-n:]
#     return train, test

# import numpy as np
#
# route_distances = np.array([[0, 1, 2],
#                             [3, 4, 5],
#                             [6, 7, 8]])
# import numpy as np
#
# route_distances = np.array([[0, 1, 2],
#                             [3, 4, 5],
#                             [6, 7, 8]])
# print(route_distances[:,1:
#
#       ])
#
# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
#
# # Данные
# sku_1 = [1,2,3,4,5,6,7,8,9,10]
# sku_2 = [10,20,30,40,50,60,70,80,90,100]
# sku_3 = [100,200,300,400,500,600,700,800,900,1000]
# sku_4 = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
# sku_5 = [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
#
# # Преобразование в numpy массивы
# sku_1 = np.array(sku_1)
# print(sku_1)
#
#
# sku_2 = np.array(sku_2)
# sku_3 = np.array(sku_3)
# sku_4 = np.array(sku_4)
# sku_5 = np.array(sku_5)
#
# # Объединение SKU в один массив
# data = np.stack((sku_1, sku_2, sku_3, sku_4, sku_5), axis=-1)
# print(data.shape)
# print(data)
#
#
#
#
# # Формирование входных данных для LSTM
# X = data[:-1, :]  # Все данные, кроме последнего шага
# y = data[1:, :]  # Следующий шаг
# print(X)
# print(y)
#
#
# # Преобразование в форму [samples, time steps, features]
# X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
# y = np.reshape(y, (y.shape[0], y.shape[1]))
# print(X.shape)
# print(y.shape)
# # Создание LSTM модели
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(1, 5)))
# model.add(Dense(5))
# model.compile(optimizer='adam', loss='mse')
#
# # Обучение модели
# model.fit(X, y, epochs=200, verbose=0)
# # Предсказание для следующего шага
# last_step = X[-1,:,:]  # последний доступный шаг в данных
# last_step = np.reshape(last_step, (1, 1, 5))  # преобразование в форму, ожидаемую моделью
# print(last_step.shape)
#
#
# next_step = model.predict(last_step)  # предсказание следующего шага
#
# print(next_step)

# import numpy as np
#
# # Предположим, что array - это ваш исходный массив размером (265, 10)
# array = np.random.rand(265, 10)
#
# # Создаем массив нулей размером (100, 10)
# zeros = np.zeros((100, 10))
#
# # Добавляем массив нулей к исходному массиву
# result = np.vstack((zeros, array))
#
# print(result.shape)  # Выводит: (365, 10)
# print(result)

# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
#
#
# scaler = MinMaxScaler()
# scaler_std = StandardScaler()
#
#
# # Предположим, что array - это ваш исходный массив размером (1400, 10)
# array = np.random.rand(1400, 10)
#
# arr = array.copy()
# arr1 = array.copy()
# arr2 = array.copy()
#
# # Находим минимальное и максимальное значение в массиве
# min_val = np.min(array)
# max_val = np.max(array)
#
# # Выполняем нормализацию
# normalized_array = (array - min_val) / (max_val - min_val)
#
# print(normalized_array)
# print('__________')
# normalized_array_1 = scaler.fit_transform(arr)
#
#
# print(normalized_array_1)
# print("____________")
# mean, std = arr1.mean(axis=0), arr1.std(axis=0)
#
# train_array = (arr1 - mean) / std
# print(train_array)
#
# scaled_array_s= scaler_std.fit_transform(arr1)
# print(scaled_array_s)
# print('_______')
#
# scaled_array_ss= scaler_std.transform(arr2)
# print(scaled_array_ss)
#
# import dgl
# import itertools
#
# # Создаем список всех SKU. В этом примере у нас есть 10 SKU, поэтому мы используем range(10).
# nodes = list(range(10))
#
# # Создаем все возможные пары SKU, потому что каждый SKU влияет на все остальные.
# edges = list(itertools.product(nodes, repeat=2))
# print(edges)
#
# # Разделяем список пар на два списка: исходные узлы и конечные узлы.
# src, dst = zip(*edges)
# print(src)
# print(dst)
#
# # Создаем граф.
# g = dgl.graph((src, dst))
# print(g)
# import matplotlib.pyplot as plt
# import networkx as nx
#
# # Преобразование в формат networkx
# nx_g = g.to_networkx()
#
# # Визуализация графа
# plt.figure(figsize=(8, 8))
# nx.draw(nx_g, with_labels=True)
# plt.show()
#
# 'Deep Graph Library'
# 'Машинное обучение на графах с помощью DGL и PyG'
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
#
#
# def manual_standard_scaler(data):
#     mean = np.mean(data, axis=0)
#     std = np.std(data, axis=0)
#     normalized_data = (data - mean) / std
#     return normalized_data, mean, std
#
# def manual_inverse_standard_scaler(normalized_data, mean, std):
#     data = normalized_data * std + mean
#     return data
#
# sample = np.abs(np.random.normal(loc=0, scale=1, size=(365, 10)))
#
# scaler = StandardScaler()
# X_train_normalize_np = scaler.fit_transform(sample)
# X_train_result = scaler.inverse_transform(X_train_normalize_np)
#
# X_train_norm_def, mean, std = manual_standard_scaler(sample)
# X_train_result_def = manual_inverse_standard_scaler(X_train_norm_def, mean, std)
#
# print('original',sample)
# print('inverse',X_train_result)
# print('mean_res',X_train_result.mean() - sample.mean())
#
# print('____________')
#
# print('original_new',sample)
# print('inverse_new',X_train_result_def)
# print('mean_res_new',X_train_result_def.mean() - sample.mean())
#



# for i in range(10):
#
#     x = np.arange(0, 365)
#
#
#     plt.plot(x, sample[:,i])
#
#     plt.xlabel('x')
#     plt.ylabel('sample')
#     plt.title('Curve Plot')
#     plt.show()

from sklearn.model_selection import train_test_split

# import numpy as np
#
# # Генерируем данные по нормальному закону
# data = np.random.normal(size=(1, 365))
#
# # Возьмем модуль от данных, чтобы они были положительными
# data = np.abs(data)
#
#
# print(data.max())
# print(data.min())
#
# # Масштабируем данные, чтобы они были в диапазоне от 0 до 40
# data1 = 40 * data / np.max(data)
# print(data1.max())
# print(data1.min())
# print(data1)

# from keras.models import Sequential
# from keras.layers import Dense
#
# # Создаем модель
# model1 = Sequential([Dense(10, activation='relu', input_shape=(8,))])
#
#
# # Создаем другую модель
# model2 = Sequential([Dense(10, activation='relu', input_shape=(8,))])
#
#
# print(model1.name)
# print(model2.name)
# print(model2)
# print(model1)
import numpy as np

# предположим, что arr - это ваш исходный массив размером (2,7,3)
arr = np.random.rand(2,7,3)
print(arr)

# выбираем 0 столбец из каждого трехмерного массива
result = arr[:,:,0]

print(result.shape)
print(result.flatten())

vv =[0] * 3
print(vv)
c = [1,2,3]+ vv
print(c)