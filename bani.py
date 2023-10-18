import pandas as pd


data = {1: [(7, 21), (46, 60), (53, 67), (271, 285), (296, 310)], 2: [(15, 29), (17, 31), (87, 101), (141, 155), (155, 169)], 3: [(21, 35), (37, 51), (108, 122), (206, 220), (228, 242)], 4: [(35, 49), (64, 78), (142, 156), (208, 222), (334, 348)], 5: [(34, 48), (85, 99), (124, 138), (287, 301), (305, 319)], 6: [(74, 88), (141, 155), (184, 198), (212, 226), (334, 348)], 7: [(72, 86), (88, 102), (119, 133), (149, 163), (161, 175)], 8: [(57, 71), (88, 102), (244, 258), (261, 275), (308, 322)], 9: [(20, 34), (169, 183), (174, 188), (299, 313), (339, 353)], 10: [(119, 133), (175, 189), (176, 190), (277, 291), (290, 304)]}




# Создание нового словаря с замененными ключами
new_data = {f"sku_{key}": value for key, value in data.items()}

# Вывод нового словаря
for i in new_data.items():

    print(i)




# Создание списка ключей из исходного словаря
keys = list(data.keys())

# Создание списка словарей без одной строки
new_dicts = []
d = 0
for i in range(len(keys)):
    d+=1
    new_dict = {f"sku_{key}_{d}": value for key, value in data.items() if key != keys[i]}
    new_dicts.append(new_dict)
print(new_dicts)

# # Вывод списка словарей
for new_dict in new_dicts:

    print(new_dict)


# # Создание списка ключей из исходного словаря
# keys = list(data.keys())
#
# # Создание списка словарей без одной строки с добавлением префикса
# new_dicts = []
# for i in range(len(keys)):
#     new_dict = {f"sku_{i+2}_1": value for key, value in data.items() if key != keys[i]}
#     new_dicts.append(new_dict)
#
# # Вывод списка словарей
# for new_dict in new_dicts:
#     print(new_dict)

# Create a new column called 'index_promo' with empty values

import pandas as pd

# Create a new DataFrame with 3650 rows
df = pd.DataFrame({'DAYS': range(1, 3651)})

# Create a new column called 'index_promo' with empty values
df['index_promo'] = ''

# Iterate over the 'data' dictionary
for dos in new_dicts:
    for sku, ranges in dos.items():
        for start, end in ranges:
            # Expand the range and check if any value matches the 'DAYS' column
            matching_days = range(start, end + 1)
            df.loc[df['DAYS'].isin(matching_days), 'index_promo'] = sku
print(df.loc[15:30])
import pandas as pd

# Assuming you have a DataFrame `df` with the `index_promo` column
# df['res_index_promo'] = df['index_promo']
# ko = [[['sku_2_1: 0.42352940752941187', 'sku_3_1: 0.3529411799999999', 'sku_4_1: 0.3058823570196078', 'sku_5_1: 0.27176470832823524', 'sku_6_1: 0.24705882229411769', 'sku_7_1: 0.22805430057222412', 'sku_8_1: 0.2127450969754902', 'sku_9_1: 0.19999999999999998', 'sku_10_1: 0.1891764716476235'], ['sku_1_2: 0.010588235999999996', 'sku_3_2: 0.5000000099999999', 'sku_4_2: 0.43333334344444446', 'sku_5_2: 0.38500000731499995', 'sku_6_2: 0.35000000174999996', 'sku_7_2: 0.3230769290414201', 'sku_8_2: 0.3013888903958333', 'sku_9_2: 0.28333333616666656', 'sku_10_2: 0.26800000418079994'], ['sku_1_3: 0.010588235999999996', 'sku_2_3: 0.040000001999999986', 'sku_4_3: 0.5200000017333334', 'sku_5_3: 0.46199999953800003', 'sku_6_3: 0.41999999370000013', 'sku_7_3: 0.387692307095858', 'sku_8_3: 0.3616666612416668', 'sku_9_3: 0.3399999966', 'sku_10_3: 0.32159999858495997'], ['sku_1_4: 0.010588235999999996', 'sku_2_4: 0.040000001999999986', 'sku_3_4: 0.06', 'sku_5_4: 0.533076920766923', 'sku_6_4: 0.48461537573076935', 'sku_7_4: 0.4473372759271734', 'sku_8_4: 0.41730768465705137', 'sku_9_4: 0.3923076870769231', 'sku_10_4: 0.3710769202072615'], ['sku_1_5: 0.010588235999999996', 'sku_2_5: 0.040000001999999986', 'sku_3_5: 0.06', 'sku_4_5: 0.07846153799999998', 'sku_6_5: 0.545454537818182', 'sku_7_5: 0.50349650322539', 'sku_8_5: 0.46969696312121223', 'sku_9_5: 0.4415584375844156', 'sku_10_5: 0.4176623362422857'], ['sku_1_6: 0.010588235999999996', 'sku_2_6: 0.040000001999999986', 'sku_3_6: 0.06', 'sku_4_6: 0.07846153799999998', 'sku_5_6: 0.095844156', 'sku_7_6: 0.553846161301775', 'sku_8_6: 0.5166666666666666', 'sku_9_6: 0.48571428814285705', 'sku_10_6: 0.4594285762985142'], ['sku_1_7: 0.010588235999999996', 'sku_2_7: 0.040000001999999986', 'sku_3_7: 0.06', 'sku_4_7: 0.07846153799999998', 'sku_5_7: 0.095844156', 'sku_6_7: 0.11142857399999999', 'sku_8_7: 0.5597222146875002', 'sku_9_7: 0.5261904717380952', 'sku_10_7: 0.4977142842900571'], ['sku_1_8: 0.010588235999999996', 'sku_2_8: 0.040000001999999986', 'sku_3_8: 0.06', 'sku_4_8: 0.07846153799999998', 'sku_5_8: 0.095844156', 'sku_6_8: 0.11142857399999999', 'sku_7_8: 0.125714286', 'sku_9_8: 0.5640553023594469', 'sku_10_8: 0.5335299595724681'], ['sku_1_9: 0.010588235999999996', 'sku_2_9: 0.040000001999999986', 'sku_3_9: 0.06', 'sku_4_9: 0.07846153799999998', 'sku_5_9: 0.095844156', 'sku_6_9: 0.11142857399999999', 'sku_7_9: 0.125714286', 'sku_8_9: 0.139078344', 'sku_10_9: 0.5675294149428706'], ['sku_1_10: 0.010588235999999996', 'sku_2_10: 0.040000001999999986', 'sku_3_10: 0.06', 'sku_4_10: 0.07846153799999998', 'sku_5_10: 0.095844156', 'sku_6_10: 0.11142857399999999', 'sku_7_10: 0.125714286', 'sku_8_10: 0.139078344', 'sku_9_10: 0.151764708']]]
# for lo in ko:
#     for row in lo:
#         for item in row:
#             key, value = item.split(':')
#             column_name = key.split('_')[0]
#             df.loc[df[column_name] == key, 'res_index_promo'] *= float(value)
#
#

import pandas as pd
import random
ko = [['sku_9_1: 0.42352940752941187', 'sku_2_1: 0.3529411799999999', 'sku_4_1: 0.3058823570196078', 'sku_5_1: 0.27176470832823524', 'sku_6_1: 0.24705882229411769', 'sku_7_1: 0.22805430057222412', 'sku_8_1: 0.2127450969754902', 'sku_9_1: 0.19999999999999998', 'sku_10_1: 0.1891764716476235'], ['sku_1_2: 0.010588235999999996', 'sku_3_2: 0.5000000099999999', 'sku_4_2: 0.43333334344444446', 'sku_5_2: 0.38500000731499995', 'sku_6_2: 0.35000000174999996', 'sku_7_2: 0.3230769290414201', 'sku_8_2: 0.3013888903958333', 'sku_9_2: 0.28333333616666656', 'sku_10_2: 0.26800000418079994']]

print('llllllllllllllllll')
# Create a list of sku values
skus = ['sku_{}_1'.format(i) for i in range(1, 11)]
sales_reg = [i for i in range(1, 11)]

# Shuffle the list randomly
random.shuffle(skus)

# Create a DataFrame with the 'index_promo' column
df = pd.DataFrame({'index_promo': skus})

# Print the DataFrame


# df['res_index_promo'] = [item.split(': ')[1] for sublist in ko for item in sublist]
df['res_index_promo'] = ''  # Создаем новую колонку с пустыми значениями

for sublist in ko:
    for item in sublist:
        key, value = item.split(': ')
        df.loc[df['index_promo'] == key, 'res_index_promo'] = float(value)

df['sales_reg'] = sales_reg
# df['res_index_promo'].replace('', df['sales_reg'], inplace=True)

df['result'] = df['sales_reg'] * df['res_index_promo']
df['sales_reg'] = df['sales_reg'] * df['res_index_promo']
# df['result'].fillna(df['sales_reg'], inplace=True)



print(df,'____')
