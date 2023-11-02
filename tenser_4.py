import os

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # import tensorflow as tf
# # import numpy as np
# #
# # # Генерация случайных данных
# # np.random.seed(0)
# # x_train = np.random.rand(100, 1)
# #
# # y_train = 2 * x_train + np.random.randn(100, 1) * 0.1
# #
# # # Определение архитектуры модели
# # model = tf.keras.Sequential([
# #     tf.keras.layers.Dense(1, activation='relu', input_shape=(1,)),
# #
# #     tf.keras.layers.Dense(1, activation='s'),
# # ])
# #
# # # Компиляция модели
# # model.compile(optimizer='adam',
# #               loss='mean_squared_error',metrics='accuracy')
# #
# #
# # # Обратный вызов для записи истории значений
# #
# # # Получение значений между слоями
# # layer_outputs = [layer.output for layer in model.layers]
# #
# # activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
# # print(activation_model.summary())
# # activations = activation_model.predict(x_train)
# #
# #
# # # Вывод значений между слоями
# #
# # for activation in activations:
# #     print(activation)
#
#
# import tensorflow as tf
# import numpy as np
#
# # Генерация случайных данных
# np.random.seed(0)
# x_train = np.random.rand(100, 4)
# y_train = 2 * x_train + np.random.randn(100, 1) * 0.1
# x_train = np.expand_dims(x_train, axis=1)
#
#
# # Определение архитектуры модели
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(16, activation='relu', input_shape=(1,4)),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Flatten(data_format=None),
#     tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None),
#     tf.keras.Input(shape=(10,)),
#     tf.keras.layers.Dense(1)
# ])
#
# # Компиляция модели
# model.compile(optimizer='adam',
#               loss='mean_squared_error')
#
# # Обучение модели
# model.fit(x_train, y_train, epochs=10,verbose=0)
# # print(pd.DataFrame([[x_train, y_train]]))
# # Получение выхода промежуточного слоя
# intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
#                                                  outputs=model.layers[0].output)
# intermediate_output = intermediate_layer_model.predict(x_train)
#
# # Вывод выхода промежуточного слоя
# # print(intermediate_output)
# # print(pd.DataFrame(intermediate_output))
#
# print(model.summary())
#
# # Получаем первый слой модели
# first_layer = model.layers[0]
#
# # Получаем форму входного тензора
# input_shape = first_layer.input_shape
#
#
#
# import tensorflow as tf
# from tensorflow.keras import layers
#
# # Создаем входной слой с формой (None, 10)
# input_layer = tf.keras.Input(shape=(10,))
#
# # Добавляем другие слои к модели
# hidden_layer = layers.Dense(16, activation='relu')(input_layer)
# output_layer = layers.Dense(1, activation='sigmoid')(hidden_layer)
#
# # Создаем модель
# model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
# print(model.summary())