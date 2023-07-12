

import pandas as pd
import numpy as np
from io import StringIO

# df = pd.DataFrame([[1, 'Bob', 'Builder'],
#                    [2, 'Sally', 'Baker'],
#                    [3, 'Scott', 'Candle Stick Maker']],
#                   columns=['id', 'name', 'occupation'],
#                   index=['a', 'b', 'c']
#                   )
#
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

print(df.dtypes)
# print(df.loc[0:5].iloc[0:10])
# df['name'] = 1
new_row = pd.Series(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [
        'quarter', 'SER_REF', 'industry_code', 'industry_name', 'filled jobs', 'filled jobs revised',
        'filled jobs diff', 'filled jobs % diff', 'total_earnings', 'total earnings revised', 'earnings diff',
        'earnings % diff'
    ]

)
# df.iloc[201]=new_row
# df._append(new_row,ignore_index=True)
# df.drop([0],inplace=True)
# df.drop(['industry_code','filled jobs','filled jobs revised','filled jobs diff','filled jobs % diff','total_earnings','total earnings revised','earnings diff','earnings % diff'], axis=1, inplace=True)
# df.iloc[:10].drop([6],inplace = True)
print(df)

csv_d1 = 'col_A, col_B, col_C\na1, b1, c1\na2, b2, c2'

df = pd.read_csv(StringIO(csv_d1))
print(df)
print("_____________")
json_buf='[[12, 24, 14, 17], [1, 54, 25, 83], [65,35, 12, 72]]'
fox = pd.read_json(json_buf, orient='value')
print(fox.agg(['sum', 'mean', 'std'], axis=1))
print(pd.get_option('display.encoding'))
def gou():
    return 25+25
gou.__name__='loo'
print(gou.__name__)

