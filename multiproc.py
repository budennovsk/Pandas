# import pandas as pd
#
# # Your lists of lists
# data_list = [
#     [
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Bonfesto Моцарелла Pizza Пленка 250 г 40,0%', 74888.44, 13320.119999999999],
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Cheerussi Дой-пак 100 г 45,0%', 298.69, 148.37]
#     ],
#     [
#         ['X6', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Bonfesto Моцарелла Pizza Пленка 250 г 40,0%', 74888.44, 13320.119999999999],
#         ['X6', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Cheerussi Дой-пак 100 г 45,0%', 298.69, 148.37]
#     ]
#     # Add more lists of lists here if needed
# ]
#
# # Define the column names
# columns = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6']
#
# # Create a list of DataFrames
# df_list = [pd.DataFrame(data, columns=columns) for data in data_list]
#
#
# # Concatenate all the DataFrames in the list
# df = pd.concat(df_list, ignore_index=True)
#
# print(df)
#
# import pandas as pd
#
# # Your lists of lists
# data_list = [
#     [
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Bonfesto Моцарелла Pizza Пленка 250 г 40,0%', 74888.44, 13320.119999999999],
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Cheerussi Дой-пак 100 г 45,0%', 298.69, 148.37]
#     ],
#     [
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Bonfesto Моцарелла Pizza Пленка 250 г 40,0%', 74888.44, 13320.119999999999],
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Cheerussi Дой-пак 100 г 45,0%', 298.69, 148.37]
#     ]
#     # Add more lists of lists here if needed
# ]
#
# # Transpose each list of lists and create a DataFrame
# df_list = [pd.DataFrame(data) for data in data_list]
#
# # Concatenate all the DataFrames along the columns axis
# df = pd.concat(df_list, ignore_index=True)
#
# print(df.to_string())
#
# import pandas as pd
#
# data_list = [
#     [
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Bonfesto Моцарелла Pizza Пленка 250 г 40,0%', 74888.44, 13320.119999999999],
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Cheerussi Дой-пак 100 г 45,0%', 298.69, 148.37]
#     ],
#     [
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Bonfesto Моцарелла Pizza Пленка 250 г 40,0%', 74888.44, 13320.119999999999],
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Cheerussi Дой-пак 100 г 45,0%', 298.69, 148.37]
#     ]
# ]
#
# # Create two DataFrames from the two lists in data_list
# df1 = pd.DataFrame(data_list[0], columns=['Brand', 'Type', 'Store', 'Product', 'Value1', 'Count1'])
# df2 = pd.DataFrame(data_list[1], columns=['Brand', 'Type', 'Store', 'Product', 'Value2', 'Count2'])
#
# # Merge the two DataFrames on ['Brand', 'Type', 'Store', 'Product']
# merged_df = pd.merge(df1, df2, on=['Brand', 'Type', 'Store', 'Product'])
#
#
#
# print(merged_df.to_string())
# print('___________')
# import pandas as pd
#
# data_list = [
#     [
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Bonfesto Моцарелла Pizza Пленка 250 г 40,0%', 74888.44, 13320.119999999999],
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Cheerussi Дой-пак 100 г 45,0%', 298.69, 148.37]
#     ],
#     [
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Bonfesto Моцарелла Pizza Пленка 250 г 40,0%', 74888.44, 13320.119999999999],
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Cheerussi Дой-пак 100 г 45,0%', 298.69, 148.37]
#     ]
#     ,
#     [
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Bonfesto Моцарелла Pizza Пленка 250 г 40,0%', 74888.44, 13320.119999999999],
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Cheerussi Дой-пак 100 г 45,0%', 298.69, 148.37]
#     ]
#     # Add more lists of lists here if needed
# ]
#
# # Create a list of DataFrames
# df_list = [pd.DataFrame(data, columns=['Brand', 'Type', 'Store', 'Product', 'Value'+str(i+1), 'Count'+str(i+1)]) for i, data in enumerate(data_list)]
#
# # Initialize merged_df as the first DataFrame in the list
# merged_df = df_list[0]
#
# # Loop through the rest of the DataFrames and merge them with merged_df
# for i in range(1, len(df_list)):
#     merged_df = pd.merge(merged_df, df_list[i], on=['Brand', 'Type', 'Store', 'Product'])
#
# print(merged_df.T.to_string())

# cc = [
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Bonfesto Моцарелла Pizza Пленка 250 г 40,0%', 74888.44, 13320.119999999999],
#         ['X5', 'Сыр полутвёрдый', 'Дискаунтер', 'Сыр полутвёрдый Cheerussi Дой-пак 100 г 45,0%', 298.69, 148.37]
#     ]
#
# for i in cc:
#     print(i+['1111'])


import pandas as pd
import numpy as np

# Assuming df is your DataFrame
df = pd.DataFrame({

    'Product': ['Сыр полутвёрдый Bonfesto Моцарелла Pizza Пленка 250 г 40,0%', 'Сыр полутвёрдый Cheerussi Дой-пак 100 г 45,0%'],
    'Value1': [74888.44, 298.69],
    'Count1': [13320.12, 148.37],
    'Value2': [74888.44, 298.69],
    'Count2': [13320.12, 148.37],
    'Value3': [74888.44, 298.69],
    'Count3': [13320.12, 148.37]
})
r1= df.T
# Ваш новый столбец
month = [0,1,1,2,2,3,3]

# Вставка столбца 'month' на первое место
r1.insert(0, 'month', month)

print(r1)

print(r1.to_string())

# Исходный список
lst = [10, 9, 8]

# Повторяем каждый элемент из исходного списка 2 раза
repeated_lst = [item for item in lst for _ in range(2)]

# Добавляем 4 нуля в начало списка
result = [0]*4 + repeated_lst

print(result)