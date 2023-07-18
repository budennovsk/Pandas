import pandas as pd
import numpy as np
from io import StringIO

# df = pd.DataFrame([[1, 'Bob', 'Builder'],
#                    [2, 'Sally', 'Baker'],
#                    [3, 'Scott', 'Candle Stick Maker']],
#                   columns=['id', 'name', 'occupation'],
#                   index=['a', 'b', 'c']
#                   )

# print()
# print('_____')
# ds = pd.Series({1: '1111'})
# ds1 = pd.Series([1, 1, 3, 4], index=['a', 'a', 'c', 'd'], name='foo')
# ds1['a'] = 222
# print(ds1)
#
# # print(df.loc['a']['id'])
# # qw = np.arange(16).reshape(4,4)
# # print(qw)
#
# df.index.name = 'ii'
# df.columns.name = '111'
#
# print(df)
#
# fc = np.array([1, '3', {1: 'd'}])
# print(fc)
# print('____')
#
#
# def metka():
#     return 'a','b'
# print(metka())
#
#
# d3 = [{'price': 3, 'count': 22}, {'price': 4, 'count': 11}]
# df3 = pd.DataFrame(d3, index=[*metka()])
# print(df3)
# print(df3.iloc[0])
# print(df3.sample(axis=1))
# print('________')
# print((df3.dtypes))
# df3['count']=df3['count'].astype('int8')
# print(df3.dtypes)
# print('_______')
# print(df3)
# print('____________')
#

df = pd.read_csv('E:/яндекс загрузка/file-csv.csv', sep=',', encoding='utf-8', usecols=['quarter', 'SER_REF'])

# print(df.dtypes)

# new_row = pd.Series(
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#     [
#         'quarter', 'SER_REF', 'industry_code', 'industry_name', 'filled jobs', 'filled jobs revised',
#         'filled jobs diff', 'filled jobs % diff', 'total_earnings', 'total earnings revised', 'earnings diff',
#         'earnings % diff'
#     ]
#
# )
# print(new_row)
#
# print(df)
# # df.loc[len(df.index)]= ['1111','11111']
#
# df.insert(2,'fff','Value')
# print(df)

df1 = pd.DataFrame([[1, 55],
                    [2, 45],
                    [3, 35]],
                   columns=['id', 'value'],

                   )
df2 = pd.DataFrame([[1, 45],
                    [2, 35],
                    [3, 25]],
                   columns=['id', 'value'],

                   )

merged_data_top_m = pd.merge(df1, df2,
                             on=['id'], how='outer')
merged_data_top_m.fillna(0, inplace=True)

merged_data_top_m['Sales_Difference'] = merged_data_top_m['value_x'] - merged_data_top_m[
        'value_y']

c = merged_data_top_m.drop(['value_x','value_y'],axis=1)
print(c)

eps = 1.0e-9

    # Рассчитайте абсолютную разницу и относительное отклонение

merged_data_top_m['Sales_Percentage_Change'] = (merged_data_top_m['Sales_Difference'] / (merged_data_top_m[
                                                                                                 'value_x'] + eps)) * 100

print(merged_data_top_m)