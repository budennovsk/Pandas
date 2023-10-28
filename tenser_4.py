from sklearn.preprocessing import StandardScaler
import numpy as np

# Пример данных
data = np.array([[1, 22], [3, 4], [5, 6], [7, 8]])
data1 = np.array([[1, 2], [4, 5]])


# Инициализация объекта стандартизации
scaler = StandardScaler()

# Применение стандартизации к данным
# scaled_data = scaler.fit_transform(data)
scaled_data1 = scaler.fit_transform(data)



 # Вывод результатов
print("Исходные данные:")
print(data)

print("\nСтандартизованные данные:")


print(scaled_data1)

se1 = scaler.inverse_transform(scaled_data1)
print(se1)



print('___')

import numpy as np

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data
print(normalize(data))
'''[[0.00000000e+00 1.00000000e+00]
 [9.00495272e-04 1.35074291e-03]
 [1.80099054e-03 2.25123818e-03]
 [2.70148582e-03 3.15173345e-03]]'''

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Создание DataFrame с числовыми данными
data3 = {'A': [10, 20, 30, 40],
        'B': [1, 2, 3, 4],
        'C': [100, 200, 300, 400]}
df = pd.DataFrame(data3)

# Создание экземпляра класса StandardScaler
scaler = StandardScaler()

# Подгонка и трансформация данных
normalized_data = scaler.fit_transform(df)

# Конвертация в новый DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

# Вывод нормализованного DataFrame
print(normalized_df)
print('_____________')

n = 0
x = np.expand_dims([[1, 2] ,[3, 4]], axis=0)
print(x)
print(x.shape)

ff = 12
dd = 12
cc =1

mask = ff==cc
print(mask)

from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.metrics import mean_absolute_error
# y_true - фактические значения, y_pred - предсказанные значения
y_true = [3, 5, 2.5, 9]
y_pred = [2.5, 5, 4, 8]

msle = mean_squared_log_error(y_true, y_pred)
print(f'MSLE: {msle}')
mse = mean_squared_error(y_true, y_pred)
print(f'MSE: {mse}')
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae}')


