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
import numpy as np

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

# import pandas as pd
#
# # Create a DataFrame with a column named 'Date'
# data = {'Date': ['2016-01-04 00:00:00+01:00', '2016-01-05 00:00:00+01:00']}
# df = pd.DataFrame(data)
# print(df)
#
# # Convert the 'Date' column to datetime format
# df['Date'] = pd.to_datetime(df['Date'])
# print(df)
#
# # Extract the date part from the datetime column
# df['Date'] = df['Date'].dt.date
#
# # Print the DataFrame
# print(df)
# import numpy as np
# from sklearn.metrics import r2_score
#
# # True values of the dependent variable
# y_true = np.array([10, 15, 12, 18, 16])
#
# # Predicted values from the regression model
# y_pred = np.array([12, 17, 14, 20, 18])
#
# # Calculate the R2 score
# r2 = r2_score(y_true, y_pred)
#
# print("R2 score:", r2)
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Прогнозируемые значения
# predictions = np.array([10, 15, 12, 18, 16])
#
# # Реальные значения
# actual_values = np.array([10, 14, 14, 20, 18])
#
#
# # Вычисление корреляции Росса
# corr_ross = np.corrcoef(predictions, actual_values)
#
# # Построение графика корреляции Росса
# plt.figure(figsize=(10, 6))
# plt.plot(predictions, actual_values, 'o')
# plt.plot([np.min(predictions), np.max(predictions)], [np.min(actual_values), np.max(actual_values)], '--k')
# plt.xlabel('Прогнозируемое значение')
# plt.ylabel('Реальное значение')
# plt.title('График корреляции Росса')
# plt.show()
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Генерируем массив x от -10 до 10, включая 1000 значений
# # y = np.random.normal(np.random.randint(20, 50), 1, 100)
# # print(y)
# # Генерируем массив x от -10 до 10, включая 1000 значений
# yy = np.random.randint(0, 100, 100)
#
# y = np.random.normal(yy, 50)
#
#
#
# # Вычисляем y как квадрат x
# x = [i for i in range(len(y))]
#
#
# # Строим график
# plt.figure(figsize=(6, 4)) # Устанавливаем размер графика
# plt.plot(x, y) # Рисуем график
#
# # Включаем сетку
# plt.grid(True)
#
# # Устанавливаем название графика и осей
# plt.title("График функции y = x^2")
# plt.xlabel("x")
# plt.ylabel("y")
#
# # Отображаем график
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# y = np.random.normal(100, 1, 100)
#
# plt.hist(y, bins=10)  # Указываем количество бинов (столбцов) для гистограммы
# plt.xlabel('Значение')
# plt.ylabel('Частота')
# plt.title('Гистограмма нормального распределения')
# plt.show()

# for weight_promo in [np.random.randint(1, 10, 5)]*3:
#     print(weight_promo)

values = np.random.randint(1, 10, 5)
weight_promo_list = []
for _ in range(1):
    weight_promo_list.extend(values)
print(weight_promo_list)

# values = np.random.randint(1, 10, 5)
# weight_promo_list = values[:5] * 3
# print(weight_promo_list)