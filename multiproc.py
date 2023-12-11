import time

start_time = time.time()

import numpy as np

import pandas as pd

# Задайте максимальное количество столбцов для вывода
pd.set_option('display.max_columns', None)

# Задайте максимальную ширину вывода для каждого столбца
pd.set_option('display.max_colwidth', None)

# Генерация данных
np.random.seed(0)

N_years = 7
days = 365 * N_years
sku_count = 500

# 8.8
koef = 8.8


# праздничные_коэффициенты = {}

# for i in range(N_years):
#     offset = i * 365
#     for key, value in holiday_coefficients.items():
#         new_key = key + offset
#         праздничные_коэффициенты[new_key] = value

def holiday_def(N_years):
    holiday_int = np.array([1, 39, 89, 190, 327, 359])
    return np.repeat(holiday_int, N_years) + np.tile(np.arange(N_years) * 365, len(holiday_int))


# Задаем веса для SKU согласно принципу Парето
pareto_weights = (np.arange(sku_count * 8, 0, -8) ** -1.5)
# pareto_weights = (np.arange(sku_count, 0, -1) ** -1.5)
# for i in range(len(pareto_weights)):
#     pareto_weights[i] += 0.2
# pareto_weights = [weight + 0.2 for weight in pareto_weights_new]
# pareto_weights = np.array([weight + 0.2 for weight in pareto_weights_new])


pareto_weights /= pareto_weights.sum()
# [0.01584834 0.0185618  0.02214873 0.02706056 0.0341002  0.04482588 0.06264608 0.09644994 0.17718986 0.5011686 ]


# for i in range(len(pareto_weights)):
#     if i == 0:
#         pareto_weights[i] += 1000


list_df_q = []


def append_num_promo(data, sku):
    df = pd.DataFrame({'day_of_year': range(1, days + 1)})
    df['day_promo_only_num'] = 0
    df['SKU'] = sku
    df['holiday_int'] = 0

    for start, end in data:
        df['day_promo_only_num'] = np.where((df['day_of_year'] >= start) & (df['day_of_year'] <= end), 1,
                                            df['day_promo_only_num'])

    holiday_days = holiday_def(N_years)
    df['holiday_int'] = np.where((df['day_of_year'].isin(holiday_days)) |
                                 (df['day_of_year'].isin(holiday_days + 1)) |
                                 (df['day_of_year'].isin(holiday_days + 2)) |
                                 (df['day_of_year'].isin(holiday_days + 3)) |
                                 (df['day_of_year'].isin(holiday_days + 4)) |
                                 (df['day_of_year'].isin(holiday_days + 5)) |
                                 (df['day_of_year'].isin(holiday_days + 6)), 1, df['holiday_int'])

    list_df_q.append(df)


for i in range(len(pareto_weights)):
    pareto_weights[i] /= 3


# [0.00158483 0.00185618 0.00221487 0.00270606 0.00341002 0.00448259 0.00626461 0.00964499 0.01771899 0.05011686]
# print(pareto_weights)
# [0.01584834 0.0185618  0.02214873 0.02706056 0.0341002  0.04482588 0.06264608 0.09644994 0.17718986 0.5011686 ]
# [0.51584834 0.5185618  0.52214873 0.52706056 0.5341002  0.54482588 0.56264608 0.59644994 0.67718986 1.0011686 ]
# [10.01584834  0.0185618   0.02214873  0.02706056  0.0341002   0.04482588 0.06264608  0.09644994  0.17718986  0.5011686 ]
# [0.1899093  0.16912702 0.14857482 0.12827856 0.10827111 0.08859582 0.0693126  0.05051026 0.03233547 0.01508503]


def calculate_influence(current_sku, other_sku, num_v):
    if num_v == 'min_v':
        return ((current_sku / other_sku) * 0.6) * koef
    else:
        x = (0.6 * (current_sku - 10 / 100)) * koef
        if x < 0:
            return x * (-1)
        return (0.6 * (current_sku - 10 / 100)) * koef


all_sku = pareto_weights


def influence():
    influences = []
    for i, current_sku in enumerate(all_sku):
        influences.append([
            f"sku_{j + 1}_{i + 1}: {calculate_influence(current_sku, other_sku, 'min_v')}" if current_sku < other_sku else f"sku_{j + 1}_{i + 1}: {calculate_influence(other_sku, current_sku, 'max_v')}"
            for j, other_sku in enumerate(all_sku) if other_sku != current_sku
        ])
    return influences


# Получаем значения влияния в виде вложенного списка с ключами
influences = influence()


# def generate_promo_intervals():
#     starts = np.sort(np.random.choice(np.arange(1, days - 15), size=25, replace=False))
#     return [(start, start + 14) for start in starts]

def generate_promo_intervals(sku):
    new_holiday = []
    starts = np.sort(np.random.choice(np.arange(1, 365 - 15), size=5, replace=False))
    tuple_date = [(start, start + 14) for start in starts]
    rnd_nums = np.random.randint(10, size=N_years)
    for i in range(N_years):
        offset = i * 365
        new_holiday.extend([(key + offset + rnd_nums[i], value + offset + rnd_nums[i]) for key, value in tuple_date])

    append_num_promo(new_holiday, sku)

    # print(new_holiday)
    return new_holiday


promo_intervals_per_sku = {sku: generate_promo_intervals(sku) for sku in range(1, sku_count + 1)}
df_combined = pd.concat(list_df_q, ignore_index=True)

data = {
    'Day': np.tile(np.arange(1, days + 1), sku_count),
    'SKU': np.repeat(np.arange(1, sku_count + 1), days),
    'Price': np.random.uniform(10, 200, size=days * sku_count),
    'Regular Sales': np.abs(np.random.normal(100, 5, days * sku_count) * np.repeat(pareto_weights, days)),
    'Promo Sales': np.zeros(days * sku_count),
    'Stock': np.random.randint(50, 200, size=days * sku_count)
}

df = pd.DataFrame(data)

new_dicts = []

data = promo_intervals_per_sku

# Создание нового словаря с замененными ключами
new_data = {f"sku_{key}": value for key, value in data.items()}

# Создание списка ключей из исходного словаря
keys = list(data.keys())

# Создание списка словарей без одной строки
#  new_dicts  список интервалов на каждой строке sku*sku
new_dicts = []
d = 0
for i in range(len(keys)):
    d += 1
    new_dict = {f"sku_{key}_{d}": value for key, value in data.items() if key != keys[i]}
    new_dicts.append(new_dict)

# Недельная сезонность
недельные_коэффициенты = {
    0: 0.9,  # Понедельник
    1: 1.0,  # Вторник
    2: 1.0,  # Среда
    3: 1.1,  # Четверг
    4: 1.3,  # Пятница
    5: 1.5,  # Суббота
    6: 1.4  # Воскресенье
}
# Месячная сезонность
# month_coefficients = {
#     1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9,
#     6: 0.9, 7: 1.0, 8: 1.0, 9: 1.1, 10: 1.1,
#     11: 1.2, 12: 1.3, 13: 1.2, 14: 1.1, 15: 1.0, 16: 0.9, 17: 0.9,
#     18: 0.9, 19: 1.0, 20: 1.0, 21: 1.1, 22: 1.1,
#     23: 1.2, 24: 1.3,
#     25: 1.2, 26: 1.1, 27: 1.0, 28: 0.9, 29: 0.9, 30: 0.9,
#     31: 1.0, 32: 1.0, 33: 1.1, 34: 1.1, 35: 1.2, 36: 1.3
# }

month_coefficients = {
    1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9,
    6: 0.9, 7: 1.0, 8: 1.0, 9: 1.1, 10: 1.1,
    11: 1.2, 12: 1.3
}
месячные_коэффициенты = {num + i * 12: indx for i in range(N_years) for num, indx in month_coefficients.items()}

# Праздничная сезонность
# праздничные_коэффициенты = {
#     1: 1.5,   # Новый год
#     45: 1.2,  # День влюбленных
#     95: 1.3,  # Пасха
#     196: 1.4, # Летние распродажи
#     333: 1.7, # Черная пятница
#     359: 1.5,  # Рождество
#     366: 1.5,   # Новый год
#     410: 1.2,  # День влюбленных
#     460: 1.3,  # Пасха
#     561: 1.4, # Летние распродажи
#     698: 1.7, # Черная пятница
#     723: 1.5,  # Рождество
#     731: 1.5,   # Новый год
#     775: 1.2,  # День влюбленных
#     825: 1.3,  # Пасха
#     926: 1.4, # Летние распродажи
#     1063: 1.7, # Черная пятница
#     1089: 1.5,  # Рождество
# }

holiday_coefficients = {
    1: 1.5,  # Новый год
    7: 1.5,  # Рождество
    45: 1.2,  # День влюбленных
    95: 1.3,  # Пасха
    196: 1.4,  # Летние распродажи
    333: 1.7,  # Черная пятница
    365: 1.5,  # Рождество
}

праздничные_коэффициенты = {(365 * i + day): coeff for i in range(N_years) for day, coeff in
                            holiday_coefficients.items()}
last_index = [0]
sorted_holidays = sorted(праздничные_коэффициенты.items())


def holiday_multiplier(day):
    while last_index[0] < len(sorted_holidays) and sorted_holidays[last_index[0]][0] < day + 7:
        last_index[0] += 1
    if last_index[0] > 0:
        holiday, boost = sorted_holidays[last_index[0] - 1]
        days_before_holiday = holiday - day
        if 0 <= days_before_holiday <= 6:
            return 1 + (boost - 1) * (1 - days_before_holiday / 7)
    return 1


# Применяем функцию с помощью map
df['Holiday Boost'] = list(map(holiday_multiplier, df['Day']))

df['Weekday'] = df['Day'] % 7

df['Seasonality'] = df['Weekday'].map(недельные_коэффициенты)
df['Month'] = ((df['Day'] - 1) // 30) + 1
df['Monthly Seasonality'] = df['Month'].map(месячные_коэффициенты)

df['Monthly Seasonality'].fillna(value=1, inplace=True)

df['Regular Sales'] *= df['Seasonality'] * df['Monthly Seasonality'] * df['Holiday Boost']
# nan_count = df['Monthly Seasonality'].isna().sum()

# print("Number of NaN values in 'Monthly Seasonality':", nan_count)

# nan_count_h = df['Holiday Boost'].isna().sum()

# print("Number of NaN values in 'Holiday Boost':", nan_count_h)
# nan_count_q = df['Seasonality'].isna().sum()

# print("Number of NaN values in 'Seasonality':", nan_count_q)


# values = np.random.randint(1, 3, 5)
# weight_promo_list = []

# for _ in range(1,6):
#     weight_promo_list.extend(values)

# print(weight_promo_list)
# wght_promo = 0
# _________________________
# promo_intervals_per_sku = {sku: generate_promo_intervals(sku) for sku in range(1, sku_count + 1)}
# pareto_weights = (np.arange(sku_count*8, 0, -8) ** -1.5)
# for sku, intervals in promo_intervals_per_sku.items():

#     # Вычисляем основную маску один раз для каждого SKU
#     mask_sku = df['SKU'] == sku

#     for start, end in intervals:
#         # Обновляем только те части маски, которые соответствуют текущему интервалу
#         mask = mask_sku & (df['Day'] >= start) & (df['Day'] <= end)

#         df.loc[mask, 'Price'] *= np.random.uniform(0.7, 0.8)
#         promo_increase = np.abs(np.random.normal(200, 20, np.sum(mask)) * pareto_weights[sku-1])
#         df.loc[mask, 'Promo Sales'] = promo_increase * np.random.uniform(1.0, 3.0)
#         df.loc[mask, 'Regular Sales'] = 0
# __________________
for sku, intervals in promo_intervals_per_sku.items():

    # Вычисляем основную маску один раз для каждого SKU
    mask_sku = df['SKU'].values == sku

    for start, end in intervals:
        # Обновляем только те части маски, которые соответствуют текущему интервалу
        mask = mask_sku & (df['Day'].values >= start) & (df['Day'].values <= end)

        df['Price'].values[mask] *= np.random.uniform(0.7, 0.8)
        promo_increase = np.abs(np.random.normal(200, 20, np.sum(mask)) * pareto_weights[sku-1])
        df['Promo Sales'].values[mask] = promo_increase * np.random.uniform(1.0, 3.0)
        df['Regular Sales'].values[mask] = 0
# # _____________________________


# df.loc [mask,'weight_promo'] = weight_promo_list[wght_promo]
#     if intervals:
#         weight_promo_list[wght_promo]
#         wght_promo+=1

# wght_promo = 0


# if wght_promo != len(weight_promo_list):
#     wght_promo+=1
# flag = False


# other_skus_mask = (df['SKU'] != sku) & (df['Day'] >= start) & (df['Day'] <= end)
# df.loc[other_skus_mask, 'Regular Sales'] *= np.random.uniform(0.7, 0.9)


df['index_promo'] = ''

# Iterate over the 'data' dictionary
for dos in new_dicts:
    for sku, ranges in dos.items():
        for start, end in ranges:
            # Expand the range and check if any value matches the 'DAYS' column
            matching_days = range(start, end + 1)
            df.loc[df['Day'].isin(matching_days), 'index_promo'] = sku

for sublist in influences:

    for item in sublist:
        key, value = item.split(': ')
        df.loc[df['index_promo'] == key, 'res_index_promo'] = float(value)
# замена nan на 1 чтобы умножить столбцы без изменений
df['res_index_promo'].fillna(value=1, inplace=True)
# срез максимальных индексов 286 позиция слишком улетела 4 значение
# df['res_index_promo'] = df['res_index_promo'].apply(lambda x: 1.7 if x > 1.7 else x)
# показ результатов полученных индексов
df['result_1'] = df['Regular Sales'] * df['res_index_promo']
# регулярные индексы с связью sku*sku

# print(df[df['SKU']==1][25:40]) #[5:23]
df['Regular Sales'] = df['Regular Sales'] * df['res_index_promo']
df['Regular_Sales_Promo_Sales'] = df['Regular Sales'] + df['Promo Sales']

# combined_df1_df2 = df.merge(df_combined, on='SKU')
df['promo_int'] = df_combined['day_promo_only_num'].values
df['holiday_int'] = df_combined['holiday_int'].values
df['day_of_year'] = np.where(df['Day'].values % 365 == 0, 365, df['Day'].values % 365)
df['promo_merge'] = np.where(df['promo_int'].values == 1, df['day_of_year'].values, 0)

df.to_csv('df_1000.csv', index=False)

# for sku in range(1, sku_count + 1):
#     plt.figure(figsize=(22, 6))
#     subset = df[df['SKU'] == sku]
#     plt.plot(subset['Day'], subset['Regular Sales'], label='Regular Sales')
#     plt.plot(subset['Day'], subset['Promo Sales'], label='Promo Sales', color='red')
#     plt.title(f"Sales for SKU {sku}")
#     plt.legend()


#     # Изменение шага на оси x
#     plt.xticks(range(0, days, 100))
#     # праздничные_коэффициенты
#     # Выделение интервалов зеленым цветом
#      # Выделение интервалов зеленым цветом
#     interval_colors = ['green'] * sku_count
#     bound_pr = []
#     bound_ol = []
#     for start in праздничные_коэффициенты:
#         bound_ol.append(start)
#         if start != 1:
#             end = start - 6
#             bound_pr.append((start,end))
#             plt.axvspan(start, end, facecolor=interval_colors[sku-1], alpha=0.3)


#     # interval_colors = ['yellow'] * sku_count
#     # for intervals in new_dicts:
#     #     for sku_indx, tuple_num in intervals.items():

#     #         for start, end in tuple_num:
#     #             plt.axvspan(start, end, facecolor=interval_colors[sku-1], alpha=0.3)

#     interval_colors = ['yellow'] * sku_count

#     for sku_indx, tuple_num in new_dicts[sku-1].items():

#         for start, end in tuple_num:
#             plt.axvspan(start, end, facecolor=interval_colors[sku-1], alpha=0.3)


# plt.show()
# # print(bound_pr,'green')
# # print(bound_ol, 'ol')
# # c = df[df['Holiday Boost']>1]['Day'].tolist()
# # print(c)
# # print(df.loc[-10:10950])


# # Суммарные продажи по SKU
# total_sales_per_sku = df.groupby('SKU')[['Regular Sales', 'Promo Sales']].sum().sum(axis=1)
# plt.figure(figsize=(12, 6))
# total_sales_per_sku.plot(kind='bar')
# plt.title("Total Sales for Each SKU")
# plt.ylabel("Total Sales")
# plt.xlabel("SKU")
# plt.show()


# # Добавим столбец 'Date', чтобы использовать его для группировки по неделям
# df['Date'] = pd.to_datetime(df['Day'], origin='2023-01-01', unit='D')

# # Группируем по SKU и неделям, затем суммируем продажи
# weekly_sales = df.groupby('SKU').apply(lambda group: group.set_index('Date').resample('W').sum())

# # Отрисовка графиков недельных продаж для каждого SKU
# for sku in range(1, sku_count + 1):
#     plt.figure(figsize=(22, 6))

#     # Выбор нужного SKU из отсортированных данных
#     subset = weekly_sales.loc[sku]

#     # Рисуем график недельных продаж
#     plt.plot(subset.index, subset['Regular Sales'] + subset['Promo Sales'], label='Total Weekly Sales')
#     plt.title(f"Weekly Sales for SKU {sku}")
#     plt.ylabel("Sales")
#     plt.xlabel("Week")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


end_time = time.time()

print("Время выполнения сек: ", end_time - start_time)

print("Время выполнения в минутах: ", ((end_time - start_time) / 60))


# Время выполнения сек:  2.055630922317505
# Время выполнения в минутах:  0.034260515371958414

# Время выполнения сек:  105.88523411750793
# Время выполнения в минутах:  1.7647539019584655
# 50