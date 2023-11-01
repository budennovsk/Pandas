

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import numpy as np
# from keras.layers import LeakyReLU
# import tensorflow as tf
#
# # Создание слоя ELU
# elu_layer = LeakyReLU(alpha=0.01)
#
# # Пример входных данных
#
#
# # Пример входных данных
# input_data = np.array([1, 2, 3,-200, -4])
# reshaped_tensor = tf.expand_dims(input_data, axis=0)
#
# # Применение слоя ELU к входным данным
# output_data = elu_layer(reshaped_tensor)
#
#
# print("Входные данные:", input_data)
# print("Выходные данные:", output_data)
#
# import numpy as np
# from keras.layers import Input,  LayerNormalization
# from keras.models import Model
#
# # Создаем входной тензор
# batch_size = 3
# features = 4
# input_tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 22], [9, 10, 11, 12]])
#
# # Создаем слой BatchNormalization
# bn_layer = LayerNormalization(axis=-1)
#
# # Применяем нормализацию к входному тензору
# normalized_tensor = bn_layer(input_tensor)
# print(normalized_tensor)

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import numpy as np

# Create an instance of the Rescaling layer
rescaling_layer = Rescaling(scale=1./255)  # Example scale value for image data

# Data to be rescaled
data = np.array([[100, 150, 200], [50, 75, 100]])

# Rescale the data
scaled_data = rescaling_layer(data)

print(scaled_data)


