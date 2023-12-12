import time
from tqdm import tqdm

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
sku_count = 10
# 8.8
koef = 8.8


def holiday_def(N_years):
    holiday_int = np.array([1, 39, 89, 190, 327, 359])
    return np.repeat(holiday_int, N_years) + np.tile(np.arange(N_years) * 365, len(holiday_int))


pareto_weights = (np.arange(sku_count * 8, 0, -8) ** -1.5)

pareto_weights /= pareto_weights.sum()

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
    extended_holiday_days = np.concatenate([holiday_days + i for i in range(7)])
    df['holiday_int'] = np.where(df['day_of_year'].isin(extended_holiday_days), 1, df['holiday_int'])

    list_df_q.append(df)


# def append_num_promo(data, sku):
#     df = pd.DataFrame({
#         'day_of_year': range(1, days + 1),
#         'day_promo_only_num': 0,
#         'SKU': sku,
#         'holiday_int': 0
#     })

#     df.loc[df['day_of_year'].isin(data), 'day_promo_only_num'] = 1

#     holiday_days = holiday_def(N_years)
#     df['holiday_int'] = np.where((df['day_of_year'].isin(holiday_days)) |
#                                 (df['day_of_year'].isin(holiday_days + 1)) |
#                                 (df['day_of_year'].isin(holiday_days + 2)) |
#                                 (df['day_of_year'].isin(holiday_days + 3)) |
#                                 (df['day_of_year'].isin(holiday_days + 4)) |
#                                 (df['day_of_year'].isin(holiday_days + 5)) |
#                                 (df['day_of_year'].isin(holiday_days + 6)), 1, df['holiday_int'])
#     return df
# list_df_q = [append_num_promo(generate_promo_intervals(sku), sku) for sku in range(1, sku_count + 1)]
# df_combined = pd.concat(list_df_q, ignore_index=True)

def normalize_pareto_weights(pareto_weights):
    for i in range(len(pareto_weights)):
        pareto_weights[i] /= 3
    return pareto_weights


pareto_weights = normalize_pareto_weights(pareto_weights)


def calculate_influence(current_sku, other_sku, num_v):
    if num_v == 'min_v':
        return ((current_sku / other_sku) * 0.6) * koef
    else:
        x = (0.6 * (current_sku - 10 / 100)) * koef
        return abs(x)


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


def generate_promo_intervals(sku):
    starts_all_years = np.random.choice(np.arange(1, 365 - 15), size=(N_years, 5), replace=False)
    starts_all_years.sort(axis=1)
    ends_all_years = starts_all_years + 14
    offsets = np.arange(N_years) * 365
    starts_all_years += offsets[:, np.newaxis]
    ends_all_years += offsets[:, np.newaxis]
    new_holiday = np.dstack((starts_all_years.ravel(), ends_all_years.ravel()))[0]

    append_num_promo(new_holiday, sku)

    return new_holiday


promo_intervals_per_sku = {sku: generate_promo_intervals(sku) for sku in range(1, sku_count + 1)}
df_combined = pd.concat(list_df_q, ignore_index=True)


def create_dataframe(days, sku_count, pareto_weights):
    df = pd.DataFrame(index=range(days * sku_count))

    df['Day'] = np.tile(np.arange(1, days + 1), sku_count)
    df['SKU'] = np.repeat(np.arange(1, sku_count + 1), days)
    df['Price'] = np.random.uniform(10, 200, size=days * sku_count)
    df['Regular Sales'] = np.abs(np.random.normal(100, 5, days * sku_count) * np.repeat(pareto_weights, days))
    df['Promo Sales'] = np.zeros(days * sku_count)
    df['Stock'] = np.random.randint(50, 200, size=days * sku_count)
    return df


df = create_dataframe(days, sku_count, pareto_weights)

data = promo_intervals_per_sku


def transform_promo_data(promo_intervals_per_sku):
    # Создание нового словаря с замененными ключами
    new_data = {f"sku_{key}": value for key, value in promo_intervals_per_sku.items()}

    # Создание списка ключей из исходного словаря
    keys = list(promo_intervals_per_sku.keys())

    # Создание списка словарей без одной строки
    new_dicts = []
    d = 0
    for i in range(len(keys)):
        d += 1
        new_dict = {f"sku_{key}_{d}": value for key, value in promo_intervals_per_sku.items() if key != keys[i]}
        new_dicts.append(new_dict)

    return new_data, new_dicts


new_data, new_dicts = transform_promo_data(promo_intervals_per_sku)

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

month_coefficients = {
    1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9,
    6: 0.9, 7: 1.0, 8: 1.0, 9: 1.1, 10: 1.1,
    11: 1.2, 12: 1.3
}
месячные_коэффициенты = {num + i * 12: indx for i in range(N_years) for num, indx in month_coefficients.items()}

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


def process_sales_data(df, df_combined):
    df['res_index_promo'].fillna(value=1, inplace=True)

    df['result_1'] = df['Regular Sales'] * df['res_index_promo']
    df['Regular Sales'] = df['Regular Sales'] * df['res_index_promo']
    df['Regular_Sales_Promo_Sales'] = df['Regular Sales'] + df['Promo Sales']

    df['promo_int'] = df_combined['day_promo_only_num'].values
    df['holiday_int'] = df_combined['holiday_int'].values
    df['day_of_year'] = np.where(df['Day'].values % 365 == 0, 365, df['Day'].values % 365)
    df['promo_merge'] = np.where(df['promo_int'].values == 1, df['day_of_year'].values, 0)
    return df


def add_seasonality(df, holiday_multiplier, недельные_коэффициенты, месячные_коэффициенты):
    df['Holiday Boost'] = list(map(holiday_multiplier, df['Day']))

    df['Weekday'] = df['Day'] % 7
    df['Seasonality'] = df['Weekday'].map(недельные_коэффициенты)
    df['Month'] = ((df['Day'] - 1) // 30) + 1
    df['Monthly Seasonality'] = df['Month'].map(месячные_коэффициенты)

    df['Monthly Seasonality'].fillna(value=1, inplace=True)

    df['Regular Sales'] *= df['Seasonality'] * df['Monthly Seasonality'] * df['Holiday Boost']
    return df


df = add_seasonality(df, holiday_multiplier, недельные_коэффициенты, месячные_коэффициенты)


def apply_promo(df, promo_intervals_per_sku, pareto_weights):
    for sku, intervals in tqdm(promo_intervals_per_sku.items(), desc='Applying promos'):
        # Вычисляем основную маску один раз для каждого SKU
        mask_sku = df['SKU'].values == sku

        for start, end in intervals:
            # Обновляем только те части маски, которые соответствуют текущему интервалу
            mask = mask_sku & (df['Day'].values >= start) & (df['Day'].values <= end)

            df['Price'].values[mask] *= np.random.uniform(0.7, 0.8)
            promo_increase = np.abs(np.random.normal(200, 20, np.sum(mask)) * pareto_weights[sku - 1])
            df['Promo Sales'].values[mask] = promo_increase * np.random.uniform(1.0, 3.0)
            df['Regular Sales'].values[mask] = 0

    df['index_promo'] = ''
    return df


df = apply_promo(df, promo_intervals_per_sku, pareto_weights)


# for sku, intervals in promo_intervals_per_sku.items():

#     # Вычисляем основную маску один раз для каждого SKU
#     mask_sku = df['SKU'].values == sku

#     for start, end in intervals:
#         # Обновляем только те части маски, которые соответствуют текущему интервалу
#         mask = mask_sku & (df['Day'].values >= start) & (df['Day'].values <= end)

#         df['Price'].values[mask] *= np.random.uniform(0.7, 0.8)
#         promo_increase = np.abs(np.random.normal(200, 20, np.sum(mask)) * pareto_weights[sku-1])
#         df['Promo Sales'].values[mask] = promo_increase * np.random.uniform(1.0, 3.0)
#         df['Regular Sales'].values[mask] = 0


# df['index_promo'] = ''

# Iterate over the 'data' dictionary


# ____________
# for dos in new_dicts:
#     for sku, ranges in dos.items():
#         for start, end in ranges:
#             # Expand the range and check if any value matches the 'DAYS' column
#             matching_days = range(start, end + 1)
#             df.loc[df['Day'].isin(matching_days), 'index_promo'] = sku

# for sublist in influences:

#     for item in sublist:
#         key, value = item.split(': ')
#         df.loc[df['index_promo'] == key, 'res_index_promo'] = float(value)
# замена nan на 1 чтобы умножить столбцы без изменений
# _____________

def add_promo_and_influence(df, new_dicts, influences):
    index_promo = {}
    res_index_promo = {}

    # Mapping promo index
    for dos in new_dicts:
        for sku, ranges in dos.items():
            for start, end in ranges:
                matching_days = range(start, end + 1)
                for day in matching_days:
                    index_promo[day] = sku

    # Mapping influence values
    for sublist in influences:
        for item in sublist:
            key, value = item.split(': ')
            res_index_promo[key] = float(value)

    # Update DataFrame using mapped values
    df['index_promo'] = df['Day'].map(index_promo)
    df['res_index_promo'] = df['index_promo'].map(res_index_promo)

    return df


df = add_promo_and_influence(df, new_dicts, influences)


def process_sales_data(df, df_combined):
    df['res_index_promo'].fillna(value=1, inplace=True)

    df['result_1'] = df['Regular Sales'] * df['res_index_promo']
    df['Regular Sales'] = df['Regular Sales'] * df['res_index_promo']
    df['Regular_Sales_Promo_Sales'] = df['Regular Sales'] + df['Promo Sales']

    df['promo_int'] = df_combined['day_promo_only_num'].values
    df['holiday_int'] = df_combined['holiday_int'].values
    df['day_of_year'] = np.where(df['Day'].values % 365 == 0, 365, df['Day'].values % 365)
    df['promo_merge'] = np.where(df['promo_int'].values == 1, df['day_of_year'].values, 0)
    return df


df = process_sales_data(df, df_combined)

# df.to_csv('df_1000.csv', index=False)


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
#       # Выделение интервалов зеленым цветом
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
print()

print("Время выполнения сек: ", end_time - start_time)

print("Время выполнения в минутах: ", ((end_time - start_time) / 60))


# setInterval(ClickConnect,1800000)

# function ClickConnect() {
#     console.log("Clicked on connect button");
#     document.querySelector("colab-connect-button").click()
# }
# setInterval(ClickConnect,60000)