import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Генерация данных
np.random.seed(0)

days = 365
sku_count = 10

# Задаем веса для SKU согласно принципу Парето
pareto_weights = (np.arange(sku_count, 0, -1) ** -1.5)
pareto_weights /= pareto_weights.sum()


def generate_promo_intervals():
    starts = np.sort(np.random.choice(np.arange(1, days - 15), size=5, replace=False))
    return [(start, start + 14) for start in starts]

promo_intervals_per_sku = {sku: generate_promo_intervals() for sku in range(1, sku_count + 1)}

data = {
    'Day': np.tile(np.arange(1, days + 1), sku_count),
    'SKU': np.repeat(np.arange(1, sku_count + 1), days),
    'Price': np.random.uniform(10, 200, size=days * sku_count),
    'Regular Sales': np.abs(np.random.normal(100, 10, days * sku_count) * np.repeat(pareto_weights, days)),
    'Promo Sales': np.zeros(days * sku_count),
    'Stock': np.random.randint(50, 200, size=days * sku_count)
}

df = pd.DataFrame(data)

# Недельная сезонность
недельные_коэффициенты = {
    0: 0.9,  # Понедельник
    1: 1.0,  # Вторник
    2: 1.0,  # Среда
    3: 1.1,  # Четверг
    4: 1.3,  # Пятница
    5: 1.5,  # Суббота
    6: 1.4   # Воскресенье
}

# Месячная сезонность
месячные_коэффициенты = {
    1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9,
    6: 0.9, 7: 1.0, 8: 1.0, 9: 1.1, 10: 1.1,
    11: 1.2, 12: 1.3
}

# Праздничная сезонность
праздничные_коэффициенты = {
    1: 1.5,   # Новый год
    45: 1.2,  # День влюбленных
    95: 1.3,  # Пасха
    196: 1.4, # Летние распродажи
    333: 1.7, # Черная пятница
    359: 1.5  # Рождество
}

def holiday_multiplier(day):
    for holiday, boost in праздничные_коэффициенты.items():
        days_before_holiday = holiday - day
        if 0 <= days_before_holiday <= 6:
            return 1 + (boost - 1) * (1 - days_before_holiday / 7)
    return 1

df['Weekday'] = df['Day'] % 7

df['Seasonality'] = df['Weekday'].map(недельные_коэффициенты)
df['Month'] = ((df['Day'] - 1) // 30) + 1
df['Monthly Seasonality'] = df['Month'].map(месячные_коэффициенты)
df['Holiday Boost'] = df['Day'].apply(holiday_multiplier)
df['Regular Sales'] *= df['Seasonality'] * df['Monthly Seasonality'] * df['Holiday Boost']

for sku, intervals in promo_intervals_per_sku.items():
    for start, end in intervals:
        mask = (df['SKU'] == sku) & (df['Day'] >= start) & (df['Day'] <= end)

        df.loc[mask, 'Price'] *= np.random.uniform(0.7, 0.8)
        promo_increase = np.abs(np.random.normal(200, 15, np.sum(mask)) * pareto_weights[sku-1])
        df.loc[mask, 'Promo Sales'] = promo_increase
        df.loc[mask, 'Regular Sales'] = 0

        other_skus_mask = (df['SKU'] != sku) & (df['Day'] >= start) & (df['Day'] <= end)
        df.loc[other_skus_mask, 'Regular Sales'] *= np.random.uniform(0.7, 0.9)


for sku in range(1, sku_count + 1):
    plt.figure(figsize=(12, 6))
    subset = df[df['SKU'] == sku]
    plt.plot(subset['Day'], subset['Regular Sales'], label='Regular Sales')
    plt.plot(subset['Day'], subset['Promo Sales'], label='Promo Sales', color='red')
    plt.title(f"Sales for SKU {sku}")
    plt.legend()
    plt.show()

# Суммарные продажи по SKU
total_sales_per_sku = df.groupby('SKU')[['Regular Sales', 'Promo Sales']].sum().sum(axis=1)
plt.figure(figsize=(12, 6))
total_sales_per_sku.plot(kind='bar')
plt.title("Total Sales for Each SKU")
plt.ylabel("Total Sales")
plt.xlabel("SKU")
plt.show()






# Добавим столбец 'Date', чтобы использовать его для группировки по неделям
df['Date'] = pd.to_datetime(df['Day'], origin='2023-01-01', unit='D')

# Группируем по SKU и неделям, затем суммируем продажи
weekly_sales = df.groupby('SKU').apply(lambda group: group.set_index('Date').resample('W').sum())

# Отрисовка графиков недельных продаж для каждого SKU
for sku in range(1, sku_count + 1):
    plt.figure(figsize=(14, 7))

    # Выбор нужного SKU из отсортированных данных
    subset = weekly_sales.loc[sku]

    # Рисуем график недельных продаж
    plt.plot(subset.index, subset['Regular Sales'] + subset['Promo Sales'], label='Total Weekly Sales')
    plt.title(f"Weekly Sales for SKU {sku}")
    plt.ylabel("Sales")
    plt.xlabel("Week")
    plt.legend()
    plt.grid(True)
    plt.show()


print('______________-')

