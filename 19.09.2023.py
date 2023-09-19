import pandas as pd
res = pd.read_excel('ML_manufacture_first_all_10_HYPER_Батончики_Супермаркет.xlsx')
print(res)

X_col = res.columns[6:]
print(X_col)
y_col = 'Percentage_Sales_rub'
print(res[y_col])
min_index = res[y_col].idxmin()
# Выводим всю строку с минимальным значением
min_row = res.iloc[min_index, 6:].values.tolist()
print(min_row)


min_row.insert(2, 'min_row')
print(pd.DataFrame(min_row))
