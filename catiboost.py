# import matplotlib.pyplot as plt
# import numpy as np
#
# # Предположим, что у вас есть некоторые данные для построения графика
# x = np.linspace(1, 730, 730)
# y = np.sin(x)
#
# plt.plot(x, y)
#
# # Создайте новые метки для оси x
# new_xticks = np.arange(0, 830, 100)
# print(len(new_xticks))
#
# new_labels = new_xticks % 365
# print(new_labels)
# new_labels[1:][new_labels[1:] == 0] = 365  # заменить все нули, кроме первого, на 365
# print(new_labels)
# c =[0]+[100,200,300,365]*2
# print(len(c))
# plt.xticks(new_xticks,c)
#
# plt.show()
# def testing():
#     new_xticks = np.arange(0, 730, 73)
#     new_label1 = new_xticks % 365
#     print(new_label1)
#     new_labels =  np.where(new_label1 % 365 == 35, 365, new_xticks) # вычисляем остаток от деления на 365
#
#     print(new_labels)
#     new_labels[1:][new_labels[1:] == True] = 365  # заменяем все нули, кроме первого, на 365
#     print(new_labels)
# testing()
#
# c =[[0]+[100,200,300,365]*2]
# print(c)

import pandas as pd

# Предположим, что 'df' - это ваш DataFrame, а 'day_of_year' - столбец с номерами дней в году
df = pd.DataFrame({'day_of_year': range(1, 731)})  # Пример DataFrame

# Ваш список интервалов
data = [(10, 24), (47, 61), (62, 76), (280, 294), (302, 316), (378, 392), (418, 432), (426, 440), (644, 658), (668, 682)]

# Создаем новый столбец 'data_promo' и заполняем его нулями
df['data_promo'] = 0

# Проходим по списку интервалов и устанавливаем '1' там, где 'day_of_year' попадает в интервал
for start, end in data:
    df.loc[(df['day_of_year'] >= start) & (df['day_of_year'] <= end), 'data_promo'] = 1

print(df[9:25])
df = df.reset_index().merge(df_1year, how='left', on='day_of_year').set_index('index')