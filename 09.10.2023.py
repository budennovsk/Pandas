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


DATA = []
DATA_DF = pd.DataFrame()
dict_data = {}
dfd = pd.read_excel('sales_Manufacture__all_3_HYPER_Печенье_Дискаунтер.xlsx')
print(dfd.columns[1:])
head_name = ['Percentage_Sales_rub', 'Sales_rub',
             'Percentage_Sales_Price', 'Percentage_Sales_kg_reg',
             'Percentage_Sales_Price_reg', 'Percentage_Sales_kg_pr',
             'Percentage_Sales_Price_pr', 'Kod_TT', 'Percentage_Kod_TT',
             'SKU_ID', 'Percentage_SKU_ID']

for i in range(2):
    print('dd')
    mask = dfd[(dfd['Measures'] == 'Percentage_Sales_rub') |
               (dfd['Measures'] == 'Sales_Price') |
               (dfd['Measures'] == 'Percentage_Sales_Price') |
               (dfd['Measures'] == 'Percentage_Sales_kg_reg') |
               (dfd['Measures'] == 'Percentage_Sales_Price_reg') |
               (dfd['Measures'] == 'Percentage_Sales_kg_pr') |
               (dfd['Measures'] == 'Percentage_Sales_Price_pr') |
               (dfd['Measures'] == 'Kod_TT') |
               (dfd['Measures'] == 'Percentage_Kod_TT') |
               (dfd['Measures'] == 'SKU_ID') |
               (dfd['Measures'] == 'Percentage_SKU_ID')
               ]
    print('11',mask)
    q = mask.iloc[:, 1].tolist()
    DATA.append(q)
print(DATA)
ap = pd.DataFrame(columns=head_name, data=DATA)
print(ap)
# res_data = pd.concat(ap, ignore_index=True)
# print(res_data)


#     /////////////////////
#
#     mask = mask.drop(mask.columns[2:], axis=1)
#     mask1 = mask.drop(mask.columns[:1], axis=1)
#
#     print(mask1)
#     print(mask1.columns[0])
#     mask1.set_index((mask1.columns[0]), inplace=True)
#     qa = mask1.transpose()
#     print('dddd', qa.shape, qa)
#     print(dfd.shape, dfd)
#
#     DATA.append(list(qa))
# ap = pd.DataFrame(columns=head_name, data=DATA)
# print(ap)
# //////////////
# mask = mask.transpose()
# DATA_DF=mask._append(mask.transpose())
#
# print('qqq')
# print(DATA_DF)
# print(DATA_DF[DATA_DF['Measures'] == 'Sales_rub'])
#     list_data =pd.DataFrame()
#     for index, row in dfd.iterrows():
#         if row['Measures'] == 'Percentage_Sales_rub' or row['Measures'] == 'Sales_rub':
#
#             list_data = row._append(row)
#         # print(row['Measures'])
#     print('list_data',list_data)


# mask1 = dfd[(dfd['Measures'] == 'MARKET_DIFFERENCE_Sales_kg')]
# mask2 = dfd[(dfd['Measures'] == 'MARKET_DIFFERENCE_PERCENT_Sales_kg')]
#
#
# DATA.append([mask, mask1, mask2])

# print(DATA)

# #########################################
# df_header = pd.concat(DATA[0], ignore_index=True)
# print(df_header)
# df_header.insert(1, 'Mondeleze', df_header.pop('Mondeleze'))
#
# print(df_header)
#
# file_name = f'sales_difference_1.xlsx'
# writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
# df_header.to_excel(writer, index=False, sheet_name='SalesData')
#
# # Получите рабочий лист и рабочую книгу
# workbook = writer.book
# worksheet = writer.sheets['SalesData']
#
# # Авто-подгонка размеров столбцов по ширине
# auto_fit_columns(worksheet, df_header)
#
# # Примените форматирование
# number_format = workbook.add_format({'num_format': '#,##0.00'})
# worksheet.set_column('D:L', 15, number_format)
# ///////////////////////////////////////
# Двухцветное форматирование столбца Last_Month_Sales_kg
# Last_Month_Sales_kg_m = df_header.columns.get_loc(f"Last_Month_{ver}_m")
# worksheet.conditional_format(1, Last_Month_Sales_kg_m, len(merged_data), Last_Month_Sales_kg_m,
#                              {'type': '2_color_scale',
#                               'min_color': 'white',
#                               'max_color': 'green',
#                               'min_type': 'min',
#                               'max_type': 'max'})
# writer.close()
