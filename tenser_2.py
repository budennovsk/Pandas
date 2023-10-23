import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

data = {
    'Price': [120.389206, 175.224269, 198.746992, 155.133750, 98.367439, 168.447587, 23.736292, 100.957663, 78.330690, 31.649144, 140.696216, 53.650347, 91.649764, 127.753895, 76.732584, 23.016112, 119.542430, 55.677498, 142.086386, 128.471559, 124.304292, 77.688104, 153.854658, 197.402522, 110.001106, 99.439109, 12.786506, 143.038716, 173.110738, 57.016865, 69.854670, 105.128634, 23.695230, 27.032070, 90.011044, 59.631601, 87.147087, 96.333397, 195.491945, 189.409366, 36.548729, 177.691335, 25.239266, 106.450739, 41.142881, 63.247538, 120.287702, 72.105218, 129.900641, 116.581543],
    'Regular_Sales_Promo_Sales': [38.826679, 31.526468, 34.456615, 46.139441, 50.453648, 50.895382, 41.579677, 41.945723, 42.135445, 46.895481, 43.451084, 41.369417, 39.894785, 42.825602, 44.104221, 41.552756, 41.968196, 43.559962, 43.076221, 44.334923, 43.675867, 35.639354, 41.617767, 43.180784, 50.316444, 59.495796, 52.046695, 34.266775, 36.046611, 37.873469, 39.574747, 49.852214, 45.74856, 52.657005, 15.281235, 17.043563, 16.574249, 18.820411, 19.887336, 26.445687, 24.068581, 15.961874, 19.940716, 18.194436, 21.411892, 43.035008, 42.114314, 43.652903, 42.66104, 42.430884],
    'Stock': [136, 118, 56, 198, 114, 98, 153, 191, 67, 113, 109, 124, 52, 148, 64, 132, 79, 124, 138, 100, 136, 167, 67, 163, 131, 109, 60, 76, 65, 124, 194, 96, 80, 57, 124, 171, 114, 177, 153, 53, 152, 56, 58, 185, 89, 191, 78, 155, 176, 131]
}

df = pd.DataFrame(data)

sales = df['Regular_Sales_Promo_Sales'].values

sales_min = np.min(sales)
sales_max = np.max(sales)

sales_normalized = (sales - sales_min) / (sales_max - sales_min)

sequence_length = 10
forecast_days = 30

X_train = []
y_train = []

for i in range(len(sales_normalized) - sequence_length - forecast_days + 1):
    X_train.append(sales_normalized[i : i + sequence_length])
    y_train.append(sales_normalized[i + sequence_length : i + sequence_length + forecast_days])

X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train)



model = keras.models.Sequential([
    keras.layers.LSTM(64, input_shape=(sequence_length, 1)),
    keras.layers.Dense(forecast_days)
])

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=32)

X_test = sales_normalized[-sequence_length:].reshape(1, -1, 1)
y_pred_normalized = model.predict(X_test)
y_pred = y_pred_normalized * (sales_max - sales_min) + sales_min

print("Прогноз на следующие 30 дней:")
print(y_pred)

import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Regular_Sales_Promo_Sales'], label='Исходные данные')
plt.plot(np.arange(len(sales), len(sales) + len(y_pred[0])), y_pred[0], label='Прогноз', linestyle='dashed')
plt.xlabel('Дни')
plt.ylabel('Regular_Sales_Promo_Sales')
plt.legend()
plt.show()