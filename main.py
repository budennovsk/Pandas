import pandas as pd
import numpy as np

df = pd.DataFrame([[1, 'Bob', 'Builder'],
                   [2, 'Sally', 'Baker'],
                   [3, 'Scott', 'Candle Stick Maker']],
                  columns=['id', 'name', 'occupation'],
                  index=['a','b','c']
                  )

print()
print('_____')
ds = pd.Series({1: '1111'})
ds1 = pd.Series([1, 1, 3, 4], index=['a', 'a', 'c', 'd'], name='foo')
ds1['a']=222
print(ds1)

# print(df.loc['a']['id'])
# qw = np.arange(16).reshape(4,4)
# print(qw)

df.index.name='ii'
df.columns.name='111'

print(df)

