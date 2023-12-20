import pandas as pd

data = {
    'Month': [12, 12, 11, 11, 10, 10, 10, 9, 9, 8],
    'Year': [2022, 2021, 2022, 2021, 2023, 2022, 2021, 2023, 2022, 2023],
    'Bonfesto_Mozzarella_REG': [1222519.82, 6059275.35, 1225486.6, 4756991.69, 74888.44, 1865203.13, 4037181.32,
                                21787.63, 2132201.21, 0]
}

df = pd.DataFrame(data)
print(df)
print('________')

data1 = {
    'Month': [10, 11, 12, 12, 11, 10, 10, 9, 9, 8],
    'Year': [2021, 2021, 2021, 2022, 2022, 2022, 2022, 2023, 2023, 2023],
    'Bonfesto_Mozzarella_REG': [4037181.32, 4756991.69, 6059275.35, 1222519.82, 1225486.60, 1865203.13, 2132201.21,
                                74888.44, 21787.63, 0]
}

df1 = pd.DataFrame(data1)
print(df1)
print("eeee____Eeeee")
# df['Year'] = df['Year'] * -1  # multiply the years by -1
# df = df.sort_values(by=['Year', 'Month'], ascending=[True, False])
# df['Year'] = df['Year'] * -1  # multiply the years by -1 again to restore original values
df.sort_values(by=['Year', 'Month'], ascending=[True, True], inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)

# print('__')
# import pandas as pd
# from statsmodels.tsa.seasonal import seasonal_decompose
# import matplotlib.pyplot as plt
#
# data = {
#     'Month': [12, 11, 10, 9, 8, 7, 6, 5, 4, 3],
#     'Year': [2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022],
#     'Bonfesto_Mozzarella_REG': [1222519.82, 6059275.35, 1225486.6, 4756991.69, 74888.44, 1865203.13, 4037181.32,
#                                 21787.63, 2132201.21, 0]
# }
# df = pd.DataFrame(data)
#
# # Устанавливаем индекс DataFrame в качестве даты
# # df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
# # df = df.set_index('Date')
# #
# # # Упорядочиваем данные по дате
# # df = df.sort_index()
#
# # Делаем декомпозицию временного ряда
# result = seasonal_decompose(df['Bonfesto_Mozzarella_REG'], model='additive', period=2)
# # Строим графики
# result.plot()
# plt.show()
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.graphics.tsaplots import plot_acf
#
# # Создаем временной ряд с годовой сезонностью
# np.random.seed(0)
# n = 120 # количество точек данных
# t = np.arange(n)
# years = n // 12
# y = np.tile(np.sin(2 * np.pi * t[:12] / 12), years) + np.random.normal(size=n) / 10
#
# # Создаем DataFrame
# df = pd.DataFrame(y, columns=['Value'])
#
# # Создаем автокорреляционный график
# fig, ax = plt.subplots(figsize=(10, 5))
# plot_acf(df['Value'], ax=ax)
#
# # Отображаем график
# plt.show()
#
# print(df)
import pandas as pd

# Изначальный DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
})

# DataFrame, из которого мы хотим взять столбцы
df1 = pd.DataFrame({
    'C': [7, 8, 9],
    'D': [10, 11, 12],
})

# Добавляем столбцы 'C' и 'D' из df1 в df
df = pd.concat([df1[['C', 'D']], df], axis=1)

print(df)