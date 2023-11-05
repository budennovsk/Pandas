#
#
# import numpy as np
#
# def split_dataset(dataset, test_ratio=0.30):
#     print(type(np.random.rand(len(dataset))))
#     test_indices = np.random.rand(len(dataset)) < test_ratio
#     print(test_indices)
#     return dataset[~test_indices], dataset[test_indices]
#
# # Пример данных
# dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#
# # Разделение набора данных на тренировочный и тестовый наборы
# train_data, test_data = split_dataset(dataset, test_ratio=0.30)
#
# print("Тренировочный набор данных:", train_data)
# print("Тестовый набор данных:", test_data)
#
# import yfinance as yf
# from datetime import datetime
#
# # Список технологических компаний
# tech_list = ["AAPL", "GOOG", "MSFT", "TSLA", "FB", "BRKB"]
#
# # Текущая дата и время
# end = datetime.now()
#
# # Дата за один год назад
# start = datetime(end.year - 1, end.month, end.day)
#
# # Скачивание исторических данных для каждой компании из списка
# for stock in tech_list:
#     globals()[stock] = yf.download(stock, start, end)
#


#
# import tensorflow as tf
# import numpy as np
#
# # Создание данных для обучения
# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# y = np.array([2, 4, 6, 8])
#
# # Создание модели
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1, input_shape=(2,))
# ])
#
# # Обучение модели
# model.compile(optimizer='adam', loss=tf.keras.losses.Huber())
# model.fit(X, y, epochs=1000)
#
# # Получение предсказаний
# predictions = model.predict(X)
#
# # Функция потерь Mean Squared Error
# mse = tf.keras.losses.MeanSquaredError()
# mse_loss = mse(y, predictions)
# print("Mean Squared Error:", mse_loss)
#
# # Функция потерь Mean Absolute Error
# mae = tf.keras.losses.MeanAbsoluteError()
# mae_loss = mae(y, predictions)
# print("Mean Absolute Error:", mae_loss)
#
# m = tf.keras.losses.Huber()
#
# m_loss = m(y, predictions)
# print("Error:", m_loss)

import pandas as pd

# Create a DataFrame with a column named 'Date'
data = {'Date': ['2016-01-04 00:00:00+01:00', '2016-01-05 00:00:00+01:00']}
df = pd.DataFrame(data)
print(df)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
print(df)

# Extract the date part from the datetime column
df['Date'] = df['Date'].dt.date

# Print the DataFrame
print(df)
