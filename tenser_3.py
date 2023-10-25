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



import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import TensorBoard


# Генерация примера данных
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([1, 2, 3, 4])
X = np.random.rand(1000, 2)
y = np.random.randint(0, 2, size=1000)

# Инициализация кросс-валидации
kfold = KFold(n_splits=10, shuffle=True)



# Создание модели
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=2, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

# Вычисление оценки кросс-валидации
results = []
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Определение обратного вызова EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # Определение пути для сохранения журналов обучения
    log_dir = "C:/logs_Tenser"

    # Определение обратного вызова TensorBoard
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch',write_images=True)

    model.fit(X_train, y_train, epochs=100, verbose=1,callbacks=[early_stopping,tensorboard_callback], validation_data=(X_test, y_test))


    loss = model.evaluate(X_test, y_test, verbose=0)
    results.append(loss)

print("Результаты кросс-валидации:", results)
print("Среднее значение результатов:", np.mean(results))

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