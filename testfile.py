import pandas as pd


def gor():
    fff = 1
    return fff


def fe(name):
    ds = 1

    one = []
    two = []
    free = []
    if name == 'one':
        one = gor() + (1 + ds)

    if name == 'two':
        two = gor() + (2 + ds)

    if name == 'free':
        free = gor() + (3 + ds)
    return one, two, free


for i in ['one', 'two', 'free']:
    # on,tw,tre=fe(i)
    # print(on,tw,tre)
    on, tr, g = fe(i)
    print(on)
    # next(on)
    # print([i for i in on])
data = [{'Articul3': 'aaaa',
         'Group_name': 'bbbb',
         'Format_TT': 'ccccc'
         }]
df = pd.DataFrame(data=data)
print(df)
print(df.columns)
print(df.columns.values)
if 'Format_TT' in df.columns.values:
    print('ddd')
