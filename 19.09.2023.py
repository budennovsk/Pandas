import pandas as pd
import xlsxwriter

file_name = 'optimize.xlsx'



data1 = pd.DataFrame({'col1':[1,2,3],
                      'col': [4,5,6]})
data_col_name = data1.columns
data2 = pd.DataFrame({'col1':[111,21,31],
                      'col': [34,53,62]})
data_col_name_1 = data2.columns

data3 = pd.DataFrame({'col1':[1555,5552,55553],
                      'col': [4555,5,6]})

workbook = xlsxwriter.Workbook(filename=file_name)
worksheet = workbook.add_worksheet()




start_row = 0  # Начальная строка для записи данных


col = 0


# Запишите названия колонок
worksheet.write_row(start_row , col, data_col_name)
start_row+=1

# Запись значений первого DataFrame
for row in range(len(data1)):
    for col in range(len(data1.columns)):
        worksheet.write(start_row + row, col, data1.iloc[row, col])

# Перемещение к следующей строке для записи значений следующего DataFrame
start_row += len(data1) + 2

worksheet.write_row(start_row , col-1, data_col_name_1)
start_row+=1

# Запись значений второго DataFrame
for row in range(len(data2)):
    for col in range(len(data2.columns)):
        worksheet.write(start_row + row, col, data2.iloc[row, col])

# Перемещение к следующей строке для записи значений третьего DataFrame
start_row += len(data2) + 3

worksheet.write_row(start_row , col-1, data_col_name_1)
start_row+=1

# Запись значений третьего DataFrame
for row in range(len(data3)):
    for col in range(len(data3.columns)):
        worksheet.write(start_row + row, col, data3.iloc[row, col])



# Закрытие файла Excel
workbook.close()