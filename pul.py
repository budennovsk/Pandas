import pandas as pd

data = {
    'Month': [12, 12, 11, 11, 10, 10, 10, 9, 9, 8],
    'Year': [2022, 2021, 2022, 2021, 2023, 2022, 2021, 2023, 2022, 2023],
    'Bonfesto_Mozzarella_REG': [1222519.82, 6059275.35, 1225486.6, 4756991.69, 74888.44, 1865203.13, 4037181.32, 21787.63, 2132201.21, 0]
}

df = pd.DataFrame(data)
print(df)
print('________')

data1 = {
    'Month': [10, 11, 12, 12, 11, 10, 10, 9, 9, 8],
    'Year': [2021, 2021, 2021, 2022, 2022, 2022, 2022, 2023, 2023,2023],
    'Bonfesto_Mozzarella_REG': [4037181.32, 4756991.69, 6059275.35, 1222519.82, 1225486.60 , 1865203.13,  2132201.21 , 74888.44 ,  21787.63 , 0]
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

print('__')
import pandas as pd

# Создаем объект Series
s = pd.Series([1, 2, 3], name='my_series')

# Получаем название
print(s.name.split('_'))  # Выводит: my_series
print(len(s))