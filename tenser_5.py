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
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Ваши исходные данные
sales_1 = [[1,1,1],[2,2,2],[3,3,3]]
sales_2 = [[4,4,4],[5,5,5],[6,6,6]]
sales_3 = [[7,7,7],[8,8,8],[9,9,9]]

# Объедините все данные в один список
data = [sales_1, sales_2, sales_3]

# Преобразуйте список в numpy массив
data = np.array(data)
print('all',data.shape)
# Подготовьте данные
prepared_data = []
for i in range(data.shape[0]):
    current_set = []
    for j in range(data.shape[1]-1):
        current_set.append(data[i, j:j+2, :])
    prepared_data.append(current_set)
prepared_data = np.array(prepared_data)

# Разделите данные на входные данные (X) и целевую переменную (y)
X = prepared_data[:, :, 0, :]  # все данные на шаге t

y = prepared_data[:, :, 1, 0]  # только 'Regular_Sales_Promo_Sales' на шаге t+1

# Определите модель
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(2, 3)))
model.add(Dense(1))

# Скомпилируйте модель
model.compile(optimizer='adam', loss='mse')
print(X.shape)
print(X)
print(y.shape)
print(y)
# Обучите модель
model.fit(X, y, epochs=2, verbose=0)


# Предскажите значения с помощью модели
sales_pred = model.predict(X)

# Выведите предсказанные значения
print(sales_pred)