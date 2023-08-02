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
dfd = pd.read_excel('sales_Manufacture__all__HYPER_Батончики_Супермаркет.xlsx')
file_name = 'data.xlsx'
print('qqq')
writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
dfd.to_excel(writer, index=False, sheet_name='SalesData')

# Получите рабочий лист и рабочую книгу
workbook = writer.book
worksheet = writer.sheets['SalesData']

# Авто-подгонка размеров столбцов по ширине
auto_fit_columns(worksheet, dfd)

# Примените форматирование
number_format = workbook.add_format({'num_format': '#,##0.00'})
worksheet.set_column('D:L', 15, number_format)

# Двухцветное форматирование столбца Last_Month_Sales_kg
print(len(dfd.columns),'col')
print(len(dfd.index),'row')
# print((dfd.columns.get_loc('Mondeleze')),'row1')

# prev = dfd.columns.get_loc(f"Data")
# last = dfd.columns.get_loc(f"vata")
v=0
# dfd.index

for i,k in enumerate(dfd['Measures']):
    print(i,k)

    if k != 'Difference_Sales_Price':
        worksheet.conditional_format(i+1,0,i+1,len(dfd.columns),
                                     {'type': '3_color_scale',
            'min_color': "red",
            'mid_color': "white",
            'max_color': "green",
            'mid_value': '0',
            'mid_type': "num"})
    else:
        worksheet.conditional_format(i+1, 0, i+1, len(dfd.columns),
                                     {'type': '3_color_scale',
                                      'min_color': "black",
                                      'mid_color': "white",
                                      'max_color': "yellow",
                                      'mid_value': '0',
                                      'mid_type': "num"})
    v+=1
    # print(f'first_row {i}, first_col {0}, last_row {len(dfd.index)}, last_col {len(dfd.columns)}')

writer.close()
print(v)
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
