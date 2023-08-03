import pandas as pd
from xlsxwriter.utility import xl_col_to_name


def auto_fit_columns(worksheet, data):
    for col in range(len(data.columns)):
        max_length = 0
        column = xl_col_to_name(col)

        # Вычисляем максимальную длину значения в столбце
        for row in range(len(data)):
            cell_value = data.iloc[row, col]
            if cell_value is not None:
                max_length = max(max_length, len(str(cell_value)))

        # Устанавливаем ширину столбца на основе максимальной длины
        worksheet.set_column(col, col, max_length + 1)


# def gor():
#     fff = 1
#     return fff


# def fe(name):
#     ds = 1
#
#     one = []
#     two = []
#     free = []
#     if name == 'one':
#         one = gor() + (1 + ds)
#
#     if name == 'two':
#         two = gor() + (2 + ds)
#
#     if name == 'free':
#         free = gor() + (3 + ds)
#     return one, two, free
#
#
# for i in ['one', 'two', 'free']:
#     # on,tw,tre=fe(i)
#     # print(on,tw,tre)
#     on, tr, g = fe(i)
#     print(on)
#     # next(on)
#     # print([i for i in on])
# data = [{'Articul3': 'aaaa',
#          'Group_name': 'bbbb',
#          'Format_TT': 'ccccc'
#          }]
# df1 = pd.DataFrame(data=data)
# print(df1)
# print(df1.columns)
# print(df1.columns.values)
# if 'Format_TT' in df1.columns.values:
#     print('ddd')
# dfd = pd.DataFrame({'Data': [1, 20.5, 30, -555, 50.7, 62, 70],
#                     'cata': [1, 0, 1, -555, -8, 4, -44],
#                     'vata': [1, 444, 444, -555, 444, 444, 444],
#                     'ccata': [1, -44, 1, -555, 1, 1, -55],
#                     'ccta': [1, 1, 1, -555, 1, 1, 2]})
dfd = pd.read_excel('sales_Manufacture__all__HYPER_Печенье_Хард-дискаунтер.xlsx')
file_name = 'data.xlsx'
print('qqq')
ccc = dfd.drop(['Сладкий Дом'],axis=1)
writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
ccc.to_excel(writer, index=False, sheet_name='SalesData')

# Получите рабочий лист и рабочую книгу
workbook = writer.book
worksheet = writer.sheets['SalesData']

# Авто-подгонка размеров столбцов по ширине
auto_fit_columns(worksheet, ccc)

# Примените форматирование
number_format = workbook.add_format({'num_format': '#,##0.00'})
worksheet.set_column('D:L', 15, number_format)

# Двухцветное форматирование столбца Last_Month_Sales_kg
print(len(dfd.columns),'col')
print(len(dfd.index),'row')
# print(dfd.loc[dfd['Сладкий Дом'] > 100],'loc')
# print((dfd.columns.get_loc('Mondeleze')),'row1')
# print(dfd.columns.get_loc('Measures'),'loc')
#
# print(dfd[dfd['Measures']])
# c =dfd[dfd.loc[:, dfd.columns !='Measures'] >400]=0
# dfd[dfd.iloc[5][7]=3333333333333]
# c= (dfd.loc[:,dfd.columns !='Measures']>400000000).any(axis=1)
# print(c)
# print(dfd.max(axis=1))
# f =dfd[dfd.columns[1:]]
# f= dfd.iloc[5][lambda x: x == 1 if x >=400000 else x]
# c =dfd.drop('Measures',axis=1)
#
# f= c.iloc[5][c,lambda x: x == 1 if x >=400000 else x]
# print(f)
# for i in
# range(len(dfd.index)+1):
#     dfd[dfd.loc[i, dfd.columns !='Measures']>4000]=999
# print(dfd)
# qq = dfd.head(0)
# print('dd',qq)
# dfd.reset_index(drop=True, inplace=True)
# dfd.set_index('Measures',inplace=True)
# c = dfd.transpose()
# c.reset_index(drop=True, inplace=True)
# c.set_index(0,inplace=True)

print(dfd.columns)
# dfd.loc[:, 'Сладкий Дом'] = dfd['Сладкий Дом'][dfd.loc[:, 'Сладкий Дом']>=1000]=4
# dfd['Сладкий Дом'][dfd['Сладкий Дом']>=1000]=1
dfd['Сладкий Дом']=dfd['Сладкий Дом'].apply(lambda x: 1 if x>1000 else x)
print(dfd)
ccc = dfd.drop(['Сладкий Дом'],axis=1)
print(ccc)
# print(c.head(0))
# c[c['Percentage_Sales_kg']>=40000000]=1
#
# dfd = c.transpose()
#
# dfd.columns=[qq]
# print(dfd)
# print(dfd)
# for index, row in dfd.iterrows():
#
#     if row['Measures'] == 'Percentage_Sales_kg':
#         for i in row[1:]:
#             if i >=1000:
#                 row[i]=11


        # and max(row[row.index][1:]) > 100
        # df[df > 9] = 11
        # d = row.replace('88888')
        # print(d)
        # print(row)
        # print(row.loc['Сладкий Дом'],'sas')


        # 'and max(row[row.index][1:]) > 100'
        # print(row[row.index].tolist()[1:],'sass')
        # print(row['Сладкий Дом'])
        # print(index,'col')
        # print(row.index, 'row')
        # print(row['Сладкий Дом'])
        # print('s')
        # print(row.index)
        # break

# prev = dfd.columns.get_loc(f"Data")
# last = dfd.columns.get_loc(f"vata")
# v=0
# # dfd.index
number_format = workbook.add_format({'num_format': '#,##0.00'})
efg = workbook.add_format({'bg_color': 'green'})
big_data = workbook.add_format({'bg_color': 'yellow'})


worksheet.conditional_format(1, 1, len(ccc.index), len(ccc.columns)-1,
                             {'type': 'blanks',
                              'format': efg})
worksheet.set_column('B:Z', 15, number_format)
worksheet.conditional_format(1, 1, len(ccc.index), len(ccc.columns) - 1,
                                {'type': 'cell',
                                 'criteria': '>=',
                                 'value': 1000000000,
                                 'format': big_data})


for i,k in enumerate(ccc.columns):

    if k != 'Difference_Sales_Price':
        worksheet.conditional_format(i+1,0,i+1,len(ccc.columns),
                                     {'type': '3_color_scale',
            'min_color': "#ff6d6d",
            'mid_color': "white",
            'max_color': "#729fcf",
            'mid_value': '0',
            'mid_type': "num"})
    else:
        worksheet.conditional_format(i+1, 0, i+1, len(ccc.columns),
                                     {'type': '3_color_scale',
                                      'min_color': "black",
                                      'mid_color': "white",
                                      'max_color': "yellow",
                                      'mid_value': '0',
                                      'mid_type': "num"})

    # print(f'first_row {i}, first_col {0}, last_row {len(dfd.index)}, last_col {len(dfd.columns)}')

writer.close()
# print(v)
# Create a Pandas dataframe from some data.
# dfd = pd.DataFrame({'Data': [10, 20.5, 30, 40, 50.7, 62, 70]})
#
# # Create a Pandas Excel writer using XlsxWriter as the engine.
# writer = pd.ExcelWriter('pandas_conditional.xlsx', engine='xlsxwriter')
#
# # Convert the dataframe to an XlsxWriter Excel object.
# dfd.to_excel(writer, sheet_name='Sheet1')
#
# # Get the xlsxwriter workbook and worksheet objects.
# workbook = writer.book
# worksheet = writer.sheets['Sheet1']
#
# # Apply a conditional format to the cell range.
# worksheet.conditional_format('B2:B8', {'type': '3_color_scale'})
# writer.close()

# Close the Pand
