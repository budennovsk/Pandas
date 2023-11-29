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

for i in range(3):
    print('dd')
    mask = (dfd[(dfd['Measures'] == 'Percentage_Sales_rub').values |
               (dfd['Measures'] == 'Sales_Price').values |
               (dfd['Measures'] == 'Percentage_Sales_Price').values |
               (dfd['Measures'] == 'Percentage_Sales_kg_reg').values |
               (dfd['Measures'] == 'Percentage_Sales_Price_reg').values |
               (dfd['Measures'] == 'Percentage_Sales_kg_pr').values |
               (dfd['Measures'] == 'Percentage_Sales_Price_pr').values |
               (dfd['Measures'] == 'Kod_TT').values |
               (dfd['Measures'] == 'Percentage_Kod_TT').values |
               (dfd['Measures'] == 'SKU_ID').values |
               (dfd['Measures'] == 'Percentage_SKU_ID').values
               ])

    mask = dfd[dfd['Measures'] == 'Percentage_Sales_rub'].values[0][1:]
    mask1 = dfd[dfd['Measures'] == 'Sales_Price'].values[0][1:]
    mask2= dfd[dfd['Measures'] == 'Percentage_Sales_Price'].values[0][1:]
    mask3= dfd[dfd['Measures'] == 'Percentage_Sales_kg_reg'].values[0][1:]
    mask4= dfd[dfd['Measures'] == 'Percentage_Sales_Price_reg'].values[0][1:]
    mask5= dfd[dfd['Measures'] == 'Percentage_Sales_kg_pr'].values[0][1:]
    mask6= dfd[dfd['Measures'] == 'Percentage_Sales_Price_pr'].values[0][1:]
    mask7= dfd[dfd['Measures'] == 'Kod_TT'].values[0][1:]
    mask8= dfd[dfd['Measures'] == 'Percentage_Kod_TT'].values[0][1:]
    mask9= dfd[dfd['Measures'] == 'SKU_ID'].values[0][1:]
    mask10= dfd[dfd['Measures'] == 'Percentage_SKU_ID'].values[0][1:]
    # зв
    DATA.append([*mask,*mask1,*mask2,*mask3,*mask4,*mask5,*mask6,*mask7,*mask8,*mask9,*mask10])
    # DATA.append([*mask10, *mask3, *mask2, *mask, *mask4, *mask5, *mask6, *mask7, *mask8, *mask9, *mask1])
    r = range(len(DATA[0])-10)
    qr = ['Percentage_Sales_rub']*10
    print(qr)
    re = [*qr, *r]
    ap = pd.DataFrame(columns=re, data=DATA)
    ap['NAME']=ap.groupby(99)[98].rank()
    print(ap)

    if ap['NAME'].values[0] ==2.0:
        print('ssss',ap[['NAME',99]].values[0])

        d,q,e = ['1111','eee','3333']
        print(f'{d} {q} {e}')
    if (ap.columns.values=='NAME').any():
        y_coll = ap['Percentage_Sales_rub'].values
        print('er',y_coll)
        y_true= ['y_true']*y_coll.shape[0]
        print('ff',y_coll.tolist())
        df_y = pd.DataFrame(columns=[y_true],
                            data=y_coll[0])
        print('df_y',df_y)

# count = model.predict(df_catboost[X_col]).shape[0]
#         true_df = pd.DataFrame()
#         predict_df = pd.DataFrame()
#         for i in range(count):
#             true_df = pd.concat([true_df, pd.DataFrame({f'Y_True': [f'Deep_recursive: {i + 1}']})])
#             true_df = pd.concat([true_df, pd.DataFrame({f'Y_True': df_catboost[y_col].values[i]})])
#             predict_df = pd.concat([predict_df, pd.DataFrame({f'Y_Predict': [f'Deep_recursive: {i + 1}']})])
#             predict_df = pd.concat([predict_df, pd.DataFrame({f'Y_Predict': model.predict(df_catboost[X_col])[i]})])
#         result_df_many = pd.concat([true_df, predict_df], axis=1)
#         result_df_many.reset_index(drop=True, inplace=True)
#         get_feature_importance_df_many = pd.DataFrame(model.get_feature_importance(prettified=True)[:10])
#
#         comparison = pd.concat([result_df_many,get_feature_importance_df_many], axis=1)
# r = range(len(DATA[0])-10)
# qr = ['Percentage_Sales_rub']*10
# print(qr)
# re = [*qr, *r]
#
#
# ap = pd.DataFrame(columns=re, data=DATA)
# print(ap)
# ap['NAME']=ap.groupby(99)[98].rank()
# print(ap)
# f = ap.groupby(99)

# for i in f.groups:
#     print('qqq',i)
#     print('www',f.get_group(i))

# print(ap['Percentage_Sales_rub'])

    # print(dfd[dfd['Measures'] == 'Sales_Price'])

    # q = mask.iloc[:, 2:7]
    # print('qqqq', q)

#     DATA.append(q)
# print(DATA)
# ap = pd.DataFrame(columns=head_name, data=DATA)
# print(ap)
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
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# Создание и обучение модели
model = CatBoostRegressor(iterations=500,
                          depth=10,
                          learning_rate=0.01,
                          loss_function='RMSE',
                          eval_metric='RMSE',
                          early_stopping_rounds=20)

model.fit(X_train, y_train,
          eval_set=(X_test, y_test),
          verbose=False)

# Получение результатов обучения
evals_result = model.get_evals_result()

# Визуализация результатов
plt.plot(evals_result['learn']['RMSE'], label='Train')
plt.plot(evals_result['validation']['RMSE'], label='Test')
plt.title('Model error dynamics')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Создаем модель
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компилируем модель
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Создаем экземпляр EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

# Обучаем модель
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5000, verbose=0, callbacks=[es])