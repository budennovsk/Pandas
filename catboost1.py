import pandas as pd
import os

import xlsxwriter
from catboost import CatBoostRegressor, Pool
import catboost
from xlsxwriter.utility import xl_col_to_name
import numpy as np

res = pd.read_excel('ML_manufacture_first.xlsx')


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


def excel_df(merged_data, name_ver):
    # Создайте файл Excel с использованием XlsxWriter
    c = os.getcwd()
    name_folder = 'Catboost_DATA'
    if not os.path.isdir(f'{c}/{name_folder}'):
        os.mkdir(f'{c}/{name_folder}')

    file_name = f'{c}/{name_folder}/table_{name_ver}.xlsx'

    writer = pd.ExcelWriter(path=file_name, engine='xlsxwriter')
    merged_data.to_excel(writer, index=False, sheet_name='SalesData')

    # Получите рабочий лист и рабочую книгу
    workbook = writer.book
    worksheet = writer.sheets['SalesData']

    # Авто-подгонка размеров столбцов по ширине
    auto_fit_columns(worksheet, merged_data)

    # Примените форматирование
    number_format = workbook.add_format({'num_format': '#,##0.00'})
    worksheet.set_column('A:Z', 15, number_format)
    writer.close()

    print(f'File saved as {file_name}')



def exel_graph(merged_data, name_ver):
    c = os.getcwd()
    name_folder = 'Grapth_DATA'
    if not os.path.isdir(f'{c}/Catboost_DATA/{name_folder}'):
        os.mkdir(f'{c}/Catboost_DATA/{name_folder}')
    print('0')
    workbook = xlsxwriter.Workbook(f'Catboost_DATA/Grapth_DATA/graph__{name_ver}.xlsx')
    worksheet = workbook.add_worksheet()
    print('1')
    # Данные для графика

    worksheet.write_row('A1', merged_data.columns)
    worksheet.write_column('A2', merged_data['y_true'])
    worksheet.write_column('B2', merged_data['y_predict'])
    worksheet.write_column('C2', merged_data[name_ver])

    print('2')
    # Создание графика
    chart = workbook.add_chart({'type': 'line'})
    print('3')
    # 'values': '=Sheet1!$B$2:$B$11'
    # Добавление данных в график
    chart.add_series({
        'categories': '=Sheet1!$C$2:$C$11',
        'values': '=Sheet1!$B$2:$B$11',
        'name': name_ver,
    })
    print('4')
    chart.set_title({'name': 'Зависимость между y_predict и mesuares'})
    chart.set_x_axis({'name': name_ver})
    chart.set_y_axis({'name': 'y_predict'})
    # Вставка графика на лист
    worksheet.insert_chart('D2', chart)
    print('5')
    # Закрытие файла
    workbook.close()


def catboost_model():
    train = res.sample(frac=0.8).copy()
    valid = res[~res.index.isin(train.index)].copy()
    X_col = res.columns[1:]
    y_col = 'Percentage_Sales_rub'
    train_pool = Pool(train[X_col], train[y_col])
    valid_pool = Pool(valid[X_col], valid[y_col])
    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        verbose=10)
    model.fit(train_pool, eval_set=valid_pool)
    comparison = pd.DataFrame({'y_true': valid[y_col],
                               'y_predict': model.predict(valid[X_col])})
    comparison_r = pd.DataFrame({'y_true': res[y_col],
                                 'y_predict': model.predict(res[X_col])})
    # print(comparison)
    # print(comparison_r)
    return model


def get_df_graph():
    new_res_df = res.copy()
    for coloumns in new_res_df.columns[1:]:
        print(coloumns)
        new_res_df.iloc[:10, 1:] = new_res_df.agg(['mean']).values[0][1:]

        min_val, max_val = res[coloumns].agg(['min', 'max'])

        step_val = (max_val - min_val) / 10
        list_val = [min_val + (step_val * i) for i in range(10)]
        new_res_df[coloumns] = list_val

        column_val_X = new_res_df[coloumns]
        X_col = new_res_df.columns[1:]

        y_col = 'Percentage_Sales_rub'

        model_cat = catboost_model()
        comparison = pd.DataFrame({'y_true': new_res_df[y_col],
                                   'y_predict': model_cat.predict(new_res_df[X_col])})

        add_col = pd.concat([comparison, column_val_X], axis=1)
        excel_df(merged_data=new_res_df, name_ver=coloumns)
        exel_graph(merged_data=add_col, name_ver=coloumns)


if '__main__' == __name__:
    catboost_model()
    get_df_graph()
