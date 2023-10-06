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





