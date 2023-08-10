
import pandas as pd

DATA = []
dict_data = {}
dfd = pd.read_excel('sales_Manufacture__all_3_HYPER_Печенье_Дискаунтер.xlsx')
print(dfd)
mask = dfd[(dfd['Measures'] == 'MARKET_Sales_kg')]
for index, row in mask.iterrows():
    # print(row.index[1:])
    dict_data = {}
    dict_data['dd']= list(row.values)
    print(dict_data)

а
# mask1 = dfd[(dfd['Measures'] == 'MARKET_DIFFERENCE_Sales_kg')]
# mask2 = dfd[(dfd['Measures'] == 'MARKET_DIFFERENCE_PERCENT_Sales_kg')]
#
#
# DATA.append([mask, mask1, mask2])

# print(DATA)
