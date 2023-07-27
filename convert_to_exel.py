from xlsxwriter.utility import xl_col_to_name
import os
import pandas as pd


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


def excel_df_m(merged_data, manufact_name, index, ver=''):
    pass

    # merged_data.loc[len(merged_data.index)] = ['Sales_items_reg_kg', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']


    # Создайте файл Excel с использованием XlsxWriter

    c = os.getcwd()

    file_name = f'{c}/sales_Manufacture_{ver}_{str(index)}.xlsx'
    if ver == '_all_':
        file_name = f'{c}/{manufact_name}/sales_Manufacture_{ver}_{str(index[0][1])}_{str(index[1][1])}_{str(index[2][1])}.xlsx'

    # file_name = f'{c}/{manufact_name}/sales_Manufacture_{ver}_{str(index)}.xlsx'
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    merged_data.to_excel(writer, index=False, sheet_name='SalesData')

    # Получите рабочий лист и рабочую книгу
    workbook = writer.book
    worksheet = writer.sheets['SalesData']

    # Авто-подгонка размеров столбцов по ширине
    auto_fit_columns(worksheet, merged_data)

    # Примените форматирование
    number_format = workbook.add_format({'num_format': '#,##0.00'})
    worksheet.set_column('B:Z', 15, number_format)

    writer.close()

    print(f'File saved as {file_name}')


def excel_df(Manufacture, merged_data, manufact_name, ver='', f1="Last_Month_", f2="Sales_Difference", pr1=1,
             pr2=1):
    # Создайте файл Excel с использованием XlsxWriter
    f1 = f"Last_Month_{ver}"
    c = os.getcwd()
    file_name = f'{c}/{manufact_name}/sales_difference_{Manufacture}_{ver}.xlsx'
    if Manufacture == '':
        file_name = f'{c}/sales_difference_{Manufacture}_{ver}.xlsx'
        f1 = f"Last_Month_{ver}"

    writer = pd.ExcelWriter(path=file_name, engine='xlsxwriter')
    merged_data.to_excel(writer, index=False, sheet_name='SalesData')

    # Получите рабочий лист и рабочую книгу
    workbook = writer.book
    worksheet = writer.sheets['SalesData']

    # Авто-подгонка размеров столбцов по ширине
    auto_fit_columns(worksheet, merged_data)

    # Примените форматирование
    number_format = workbook.add_format({'num_format': '#,##0.00'})
    worksheet.set_column('D:H', 15, number_format)

    # Условное форматирование для столбца Sales_Difference
    # color_scale_format = workbook.add_format({'type': '2_color_scale',
    #                                           'min_color': 'white',
    #                                           'max_color': 'blue'})
    # worksheet.conditional_format('G2:G{}'.format(len(merged_data) + 1), {'type': 'cell',
    #                                                                      'format': color_scale_format})

    # Закройте ExcelWriter и сохраните файл

    # Двухцветное форматирование столбца Last_Month_Sales_kg
    print('s',f1,ver)
    if pr1:
        Last_Month_Sales_kg = merged_data.columns.get_loc(f1)
        worksheet.conditional_format(1, Last_Month_Sales_kg, len(merged_data), Last_Month_Sales_kg,
                                     {'type': '2_color_scale',
                                      'min_color': 'white',
                                      'max_color': 'blue',
                                      'min_type': 'min',
                                      'max_type': 'max'})

    # # Двухцветное форматирование столбца Sales_Difference
    # Sales_Difference = merged_data.columns.get_loc("Sales_Difference")
    # worksheet.conditional_format(1, Sales_Difference, len(merged_data), Sales_Difference,
    #                              {'type': '2_color_scale',
    #                               'min_color': 'red',
    #                               'max_color': 'white',
    #                               'min_type': 'min',
    #                               'max_type': 'max'})
    if pr2:
        Sales_Difference = merged_data.columns.get_loc(f2)
        worksheet.conditional_format(1, Sales_Difference, len(merged_data), Sales_Difference,
                                     {'type': '3_color_scale',
                                      'min_color': 'red',
                                      'mid_color': 'white',
                                      'max_color': 'blue',
                                      'min_type': 'min',
                                      'mid_type': 'num',
                                      'max_type': 'max',
                                      'min_value': None,
                                      'mid_value': 0,
                                      'max_value': None})
    else:
        Sales_Difference = merged_data.columns.get_loc(f2)
        worksheet.conditional_format(1, Sales_Difference, len(merged_data), Sales_Difference,
                                     {'type': '3_color_scale',
                                      'min_color': 'blue',
                                      'mid_color': 'white',
                                      'max_color': 'red',
                                      'min_type': 'min',
                                      'mid_type': 'num',
                                      'max_type': 'max',
                                      'min_value': None,
                                      'mid_value': 0,
                                      'max_value': None})

    writer.close()

    print(f'File saved as {file_name}')


# фактор рынка
def excel_df2(Manufacture, manufact_name, merged_data, ver):
    # Создайте файл Excel с использованием XlsxWriter

    print('parms', Manufacture)

    c = os.getcwd()

    file_name = f'{c}/{manufact_name}/sales_difference_{Manufacture}_{ver}.xlsx'
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    merged_data.to_excel(writer, index=False, sheet_name='SalesData')

    # Получите рабочий лист и рабочую книгу
    workbook = writer.book
    worksheet = writer.sheets['SalesData']

    # Авто-подгонка размеров столбцов по ширине
    auto_fit_columns(worksheet, merged_data)

    # Примените форматирование
    number_format = workbook.add_format({'num_format': '#,##0.00'})
    worksheet.set_column('D:L', 15, number_format)

    # Двухцветное форматирование столбца Last_Month_Sales_kg
    Last_Month_Sales_kg_m = merged_data.columns.get_loc(f"Last_Month_{ver}_m")
    worksheet.conditional_format(1, Last_Month_Sales_kg_m, len(merged_data), Last_Month_Sales_kg_m,
                                 {'type': '2_color_scale',
                                  'min_color': 'white',
                                  'max_color': 'blue',
                                  'min_type': 'min',
                                  'max_type': 'max'})

    # Last_Month_Sales_kg_r = merged_data.columns.get_loc("Last_Month_Sales_kg_r")
    # worksheet.conditional_format(1, Last_Month_Sales_kg_r, len(merged_data), Last_Month_Sales_kg_r,
    #                              {'type': '2_color_scale',
    #                               'min_color': 'white',
    #                               'max_color': 'blue',
    #                               'min_type': 'min',
    #                               'max_type': 'max'})

    # Sales_Difference_m = merged_data.columns.get_loc("Sales_Difference_m")
    # worksheet.conditional_format(1, Sales_Difference_m, len(merged_data), Sales_Difference_m,
    #                              {'type': '3_color_scale',
    #                               'min_color': 'red',
    #                               'mid_color': 'white',
    #                               'max_color': 'blue',
    #                               'min_type': 'min',
    #                               'mid_type': 'num',
    #                               'max_type': 'max',
    #                               'min_value': None,
    #                               'mid_value': 0,
    #                               'max_value': None})

    writer.close()

    print(f'File saved as {file_name}')