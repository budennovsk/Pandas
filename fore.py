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