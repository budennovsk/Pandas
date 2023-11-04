import os

import pandas as pd
from keras.src.layers import LSTM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense
from keras.constraints import MaxNorm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#
# # Генерируем синтетические данные для задачи регрессии
# X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
#
# # Разделяем данные на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train.shape)
# print(y_train.shape)
#
# # Создаем модель нейронной сети
# model = Sequential()
#
# # Добавляем слои с ограничением MaxNorm
# model.add(Dense(64, activation='relu', kernel_constraint=MaxNorm(max_value=2)))
# model.add(Dense(32, activation=MaxNorm(max_value=1), kernel_constraint=MaxNorm(max_value=2)))
# model.add(Dense(1, activation='linear'))
#
# # Компилируем модель
# model.compile(optimizer='adam', loss='mean_squared_error')
#
#
# # Обучаем модель на обучающей выборке
# history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test),verbose=0)
#
# # Оцениваем модель на тестовой выборке
# loss = model.evaluate(X_test, y_test)
# print(f"Mean Squared Error: {loss}")

# # Визуализация графика потерь
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Model Loss')
# plt.legend()
# plt.show()


# from keras import backend as K
# from keras.layers import LSTM
#
# class NonNegLSTM(LSTM):
#     def __init__(self, units, **kwargs):
#         super(NonNegLSTM, self).__init__(units, **kwargs)
#
#     def build(self, input_shape):
#         super(NonNegLSTM, self).build(input_shape)
#
#     def call(self, inputs, initial_state=None, constants=None, **kwargs):
#         last_output, outputs, states = super(NonNegLSTM, self).call(inputs, initial_state=initial_state, **kwargs)
#
#         # Применяем ограничение на неотрицательность к состояниям
#         states = [K.maximum(s, 0) for s in states]
#
#         return last_output, outputs, states
#
#
# from keras.models import Sequential
# import numpy as np
#
# model = Sequential()
# model.add(NonNegLSTM(64, return_sequences=True))
#
# # Генерируем фиктивные входные данные
# inputs = np.zeros((1, 10, 64))
#
#
# # Получаем выход модели
# outputs = model.predict(inputs)

import pandas as pd
import matplotlib.pyplot as plt

# Создаем DataFrame с числовыми данными
df_num = pd.DataFrame({'A': [1,1,2,2,3],
                       'B': [10, 20, 30, 40, 50],
                       'C': [100, 200, 300, 400, 500]})

# Отображаем гистограммы для всех числовых столбцов
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

# Показываем все графики
plt.show()