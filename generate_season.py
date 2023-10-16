import pandas as pd

start = 1
end = 365
step = 7

intervals = []
current = start

while current <= end:
    interval_start = current
    interval_end = min(current + step - 1, end)
    intervals.append((interval_start, interval_end))
    current += step

# Вывод результатов
print(intervals)

print('_______')
start = 1
end = 365
step = 7

intervals = [(current, min(current + step - 1, end)) for current in range(start, end + 1, step)][:-1]

# Вывод результатов
print(intervals)

print('________')
sku_count = 10


def generate_promo_intervals():
    start = 1
    end = 365
    step = 7

    for _ in range(10):
        intervals = [(current, min(current + step - 1, end)) for current in range(start, end + 1, step)][:-1]

        return intervals


promo_intervals_per_sku = {sku: generate_promo_intervals() for sku in range(1, sku_count + 1)}
print(pd.DataFrame(promo_intervals_per_sku).T)

print('________')

def generate_promo_intervals_1():
    start = 1
    end = 365
    step = 7

    for _ in range(10):
        intervals = [(current+4, min(current + step - 1, end)) for current in range(start, end + 1, step)][:-1]
        return intervals


promo_intervals_per_sku = {sku: generate_promo_intervals_1() for sku in range(1, sku_count + 1)}
print(pd.DataFrame(promo_intervals_per_sku).T)


