import pandas as pd

for i in range(5):
    print("Внешний цикл:", i)
    for j in range(3):
        print("Второй цикл:", j)
        for k in range(2):
            print("Третий цикл:", k)
            if k == 1:
                break
        else:
            continue




print('_________')


exit_condition = False

for i in range(5):
    print("Внешний цикл:", i)
    for j in range(3):
        exit_condition = False
        print("Второй цикл:", j)
        if exit_condition == True:
            continue

        for k in range(2):
            print("Третий цикл:", k)
            if k == 1:
                exit_condition = True
                break



print('_________')

import re


ver = "_all_"

if re.fullmatch(r"\_all_\d*", ver) is not None:
    print('ok')
if re.fullmatch(r"\_all_\d*", ver) is None:

    print('ok1')
else:
    print('no')

print('------------------')

ver = "_a"
if ver.startswith("_all_") and (ver == "_all_" or re.match(r"_all_\d*_\d{4}", ver)) is not None:
    print('ok')
else:
    print('no')


vf ='_all_1_2022'
print([vf[5:]])

print('++++++++++++++++++++==')

data = [[4,3,2,1],[44,55,66,77,88,99,77],[1,2,3,4]]


col = range(len(max(data)))
print(col)

er =pd.DataFrame(data, columns=col)
print(er)

print('))))))))')
for i in [10]:
    count = 10
    go = 'ress'
    if go == 'res':
        print('10')
    else:
        i +=2
qw = [[None, 1614551.7999999998], ['STM TANDER', 940055.47], ['SAVUSHKIN PRODUKT', 138869.89], ['BELSYR', 111510.79000000001], ['BELEBEEVSKIY MK', 83861.94000000002], ['VIMM-BILL-DANN', 81284.27], ['GREYT FUDZ INK', 71629.69], ['OTHER', 65683.25], ['POSTAVSKIY MZ', 37176.61000000001], ['MOLVEST', 28002.13], ['PRUZHANSKIY MK', 22488.5], ['SK LENINGRADSKIY', 19105.800000000003], ['RTK SYRNYY DOM', 7649.55], ['BELOVEZHSKIE SYRY', 2079.87], ['PIENO ZVAIGZDES', 2007.04], ['SYRNAYA DOLINA', 1417.24], ['MILKOM', 1019.97], ['BOBROVSKIY SZ', 709.79], ['AB VILKYSKIU PIENINE', None], ['ABINSKIY MZ', None], ['ADAMS FOODS LTD', None], ['AGRIVOLGA', None], ['AGROFERMA INSKIE PROSTORY', None], ['AGROFIRMA PRIVOL`E', None], ['AGROHOLDING BELOZORIE', None], ['AGROKHOLDING PORECHE', None], ['AGROSILA', None], ['AGROSVET', None], ['AGRO-TREYD', None], ['AGROVAL SA', None], ['AKH ANUYSKOE', None], ['AKSENOVA A.O', None], ['ALAPAEVSKIY MK', None], ['ALDA UNIVERSAL', None], ['ALEJSKIJ MSK', None], ['ALEV', None], ['ALEYSKIY MSK', None], ['ALIGOREKS', None], ['ALLGOY', None], ['ALLORIS', None]]
count = 0
for i in qw:
    print(i[0],'p')
    if i[0] != 'BELEBEEVSKIY MK':
        count += 1
    else:
        break
print(count)

# print('1111')
#
# print([i if i[0] != 'BELEBEEVSKIY MK' else None for i in qw])

import pandas as pd

data = [[None, 15427379.669999992], ['OTHER', 1911378.1099999999], ['VIMM-BILL-DANN', 1440507.94], ['FRESH FUDS', 1429532.58], ['BOBROVSKIY SZ', 1323507.72], ['NATURA PRO', 1159006.84], ['BELSYR', 1123024.5399999996], ['PRUZHANSKIY MK', 761223.46], ['VOKHOMSKIY SZ', 611548.27], ['VIOLA', 545640.55]]

df = pd.DataFrame(data)

# Select rows up to the row containing 'BELSYR'
truncated_df = df.iloc[:df.index[df[0] == 'VIMM-BILL-DANN'][0]+2]

print(truncated_df)


print("))))))))))))))))))")
for i in range(2):
    print(i)
    num_rows_1 = ''
    num_rows = '10'
    query = f'''pervoe mesto {num_rows if num_rows_1 !='' else num_rows}'''
    print(query)
    if i ==1:
        num_rows_1 = '20'
        print(query)



print('BBBBBBBBbb')

for i in range(2):
    print(i)
    num_rows = 10
    query = lambda: f'''pervoe mesto {num_rows}'''
    print(query())
    if i == 1:
        num_rows = 20
        print(type(query()))

print('____f________f_________f__')


fip = pd.DataFrame(data=[[233, 22], [3, 4], [5,6]], columns=['a', 'b'])
print(len(fip))
print(fip)
if 233 in fip.iloc[:, 0].values:
    print('ok')
else:
    print('no')


import logging

# Настройка логгера
logging.basicConfig(filename='errors.log',filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ваш код с циклами и условием для отлова ошибок
for i in range(10):

    if i < 5:
        logging.info(str(i))




