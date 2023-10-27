# import tensorflow as tf
# from tensorflow import keras
#
# # Загрузка данных
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#
# # Предобработка данных
# x_train = x_train / 255.0
# x_test = x_test / 255.0
#
# # Определение модели
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])
#
# # Компиляция модели
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# # Обучение модели
# history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
#
# # Визуализация метрик обучения
# import matplotlib.pyplot as plt
#
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
#
#
#
# import tensorflow as tf
# from keras.src.optimizers import Adam
# from tensorflow import keras
#
#
# # Загрузка данных
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#
# # Предобработка данных
# x_train = x_train / 255.0
# x_test = x_test / 255.0
#
# # Определение модели
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])
#
#
#
# # Компиляция модели с оптимизатором Adam
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='mse',
#               metrics=['mae'])
# # Компиляция модели
#
#
#
# # Определение коллбэка для записи метрик в TensorBoard
# # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='C:/log_Tenser', histogram_freq=1)
# # callbacks=[tensorboard_callback]
#
#
# # Обучение модели с использованием коллбэка
# model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

#
#
# import tensorflow as tf
# from sklearn.model_selection import KFold
# import numpy as np
# from tensorflow.keras.callbacks import EarlyStopping
#
# from tensorflow.keras.callbacks import TensorBoard
#
#
# # Генерация примера данных
# # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# # y = np.array([1, 2, 3, 4])
# X = np.random.rand(1000, 1)
# y = np.random.randint(0, 2, size=1000)
#
# # Инициализация кросс-валидации
# kfold = KFold(n_splits=3, shuffle=True)
#
#
#
# # Создание модели
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(1, input_dim=1, activation='linear'))
# model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
#
# # Вычисление оценки кросс-валидации
# results = []
# for train_index, test_index in kfold.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     # Определение обратного вызова EarlyStopping
#     early_stopping = EarlyStopping(monitor='val_loss', patience=3)
#
#     # Определение пути для сохранения журналов обучения
#     log_dir = "C:/logs_Tenser"
#
#     # Определение обратного вызова TensorBoard
#     tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch',write_images=True)
#
#     model.fit(X_train, y_train, epochs=100, verbose=1,callbacks=[early_stopping,tensorboard_callback], validation_data=(X_test, y_test))
#
#
#     loss = model.evaluate(X_test, y_test, verbose=0)
#     results.append(loss)
#
# print("Результаты кросс-валидации:", results)
# print("Среднее значение результатов:", np.mean(results))

#
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.callbacks import EarlyStopping
#
# # Генерация примера данных
# X_train = np.random.rand(800, 2)
# y_train = np.random.randint(0, 2, size=800)
# X_val = np.random.rand(200, 2)
# y_val = np.random.randint(0, 2, size=200)
# # Создание модели
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(1, input_dim=2, activation='linear'))
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])
#
#
# # Определение пути для сохранения журналов обучения
# log_dir = "C:/logs_Tenser"
# early_stopping = EarlyStopping(monitor='val_loss', patience=3)
#
# # Определение обратного вызова TensorBoard
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,update_freq='epoch')
#
# # Обучение модели с контролем валидации
# model.fit(X_train, y_train, epochs=300, verbose=1, callbacks=[tensorboard_callback,early_stopping], validation_data=(X_val, y_val))
#
# # Оценка модели на тестовых данных
# loss = model.evaluate(X_val, y_val, verbose=0)
# print("Потери на тестовых данных:", loss)

#
import numpy as np
import pandas as pd
# from sklearn.model_selection import TimeSeriesSplit
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
#
# # Генерируем временные ряды
# time_series = np.arange(1, 101)
#
# # Разбиваем временные ряды на временные блоки
# tscv = TimeSeriesSplit(n_splits=5)
# for train_index, test_index in tscv.split(time_series):
#     train_data = time_series[train_index]
#     test_data = time_series[test_index]
#
#     # Создаем и компилируем модель LSTM
#     model = Sequential()
#     model.add(LSTM(10, input_shape=(1, 1)))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#
#     # Преобразуем данные в формат, подходящий для модели LSTM
#     X_train = np.reshape(train_data, (len(train_data), 1, 1))
#     y_train = np.reshape(train_data, (len(train_data), 1))
#     X_test = np.reshape(test_data, (len(test_data), 1, 1))
#     y_test = np.reshape(test_data, (len(test_data), 1))
#
#     # Обучаем модель на обучающем наборе данных
#     model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=0)
#
#     # Оцениваем производительность модели на тестовом наборе данных
#     loss = model.evaluate(X_test, y_test, verbose=0)
#     print(f"Loss: {loss}")

# import numpy as np
#
# data = [np.array([0.09340992, 0.12070766, 0.17080997]), np.array([0.22305561, 0.30798442, 0.3734497])]
# data = np.vstack(data)
# print(data.shape)  # Результат: (2, 3)
#
from sklearn.metrics import mean_squared_error
data_1 = np.random.rand(140, 230)

data_2 = np.random.rand(140, 230)




df_1 = pd.DataFrame({'data_1': data_1.tolist(), 'data_2': data_2.tolist()})


df_1['MSE'] = df_1.apply(lambda row: mean_squared_error(row['data_1'], row['data_2']), axis=1)

print(df_1)
min_row = df_1.loc[df_1['MSE'].idxmin()]
max_row = df_1.loc[df_1['MSE'].idxmax()]

print('min',df_1['MSE'].idxmin())
print('max',df_1['MSE'].idxmax())
min_mse = df_1.loc[df_1['MSE'].idxmin(),'MSE']
print(min_mse,'min_mse')

max_mse = df_1.loc[df_1['MSE'].idxmax(),'MSE']
print(max_mse,'max_mse')

# mean_row = df_1.where(df_1['MSE'] == df_1['MSE'].mean()).idxmin()
# min_row_index = np.argmin(df_1['MSE'].apply(np.mean))
# print(min_row_index,'min_row_index')
# mean_row = df_1.loc[min_row_index,'MSE']
# print(mean_row,'mean_row')

index_with_min_mse = df_1['MSE'].idxmin()
index_with_max_mse = df_1['MSE'].idxmax()

mean_mse = df_1['MSE'].mean()
print(mean_mse)


value = df_1.loc[1, 'MSE']
# value = round(value, 6)
print(value,'value')
count = df_1['MSE'].value_counts().get(value, 0)
print(count,'count')
# column_index_with_value = df_1.columns[df_1.loc[2] == value].tolist()
# print(column_index_with_value,'column_index_with_value')



# Вычисление среднеквадратичной ошибки (MSE) между фактическими и предсказанными значениями
# mse = mean_squared_error(data_1, data_2)

#
# print("Mean Squared Error (MSE):", mse)