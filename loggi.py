import logging

# Настройка логгера
# logging.basicConfig(filename='errors.log', filemode='w', level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
#
# # Ваш код с циклами и условием для отлова ошибок
#
# logging.info(str('eeeeeeeeeeeeeeeee'))

print('______________________-')

data = [['NATURA PRO', 1], ['SAVUSHKIN PRODUKT', None], ['VIMM-BILL-DANN', None], ['PIR-PAK', None], ['СТМ OZON', None],
        ['VIOLA', None], ['PERVAYA LINIYA', None], ['PIENO ZVAIGZDES', None], ['NEVSKIE SYRY', None],
        ['EKONIVAAGRO', None], ['HOCHLAND', None], ['TUROVSKIY MK', None], ['SER DZHON', None], ['OTHER', None],
        ['VKUSVILL', None], ['BELOVEZHSKIE SYRY', None], ['MILKOM', None], ['GREYT FUDZ INK', None],
        ['NALCHIKSKIY MK', None]]

all_none = all(value is None for _, value in data)
print(all_none)

if all_none:
    print(True)
else:
    print(False)

data = [['None', None], ['STM X5', 1975316546.8999887], ['SAVUSHKIN PRODUKT', 674867780.4400012],
        ['VIMM-BILL-DANN', 250587005.6600004]]

filtered_data = [item for item in data if item[0] is not None]

print(filtered_data)

data = [[None, 3764156098.98999], ['STM X5', 1975316546.8999887], ['SAVUSHKIN PRODUKT', 674867780.4400012],
        ['VIMM-BILL-DANN', 250587005.6600004]]

data11 = [['ff', 3764156098.98999], ['STM X5', 1975316546.8999887], ['SAVUSHKIN PRODUKT', 674867780.4400012],
          ['VIMM-BILL-DANN', 250587005.6600004]]

data2 = [['STM X5', None], ['SAVUSHKIN PRODUKT', 674867780.4400012], ['VIMM-BILL-DANN', 250587005.6600004]]
data3 = [['STM X5', None], ['SAVUSHKIN PRODUKT', None], ['VIMM-BILL-DANN', None]]

data1 = []

print('___________')


def got():
    for item in data11:

        if item[0] is not None:
            data1.append(item)
            print('NONO item')
            return data1
        elif item[1] is None:
            item[1] = 0.001
            data1.append(item)
            print('NONE =0.01')
            return data1

    if all(item[1] is None for item in data):
        print('NONO_ALL')
        return []
    return data11


print(got())
print('qqqqq')

[[None, 3764156098.98999], ['STM X5', 1975316546.8999887], ['SAVUSHKIN PRODUKT', 674867780.4400012],
 ['VIMM-BILL-DANN', 250587005.6600004], ['ff', 3764156098.98999], ['STM X5', 1975316546.8999887],
 ['SAVUSHKIN PRODUKT', 674867780.4400012], ['VIMM-BILL-DANN', 250587005.6600004], ['STM X5', None],
 ['SAVUSHKIN PRODUKT', 674867780.4400012], ['VIMM-BILL-DANN', 250587005.6600004], ['STM X5', None],
 ['SAVUSHKIN PRODUKT', None], ['VIMM-BILL-DANN', None]]

print('____________')
data = [['None', 'None'], ['SAVUSHKIN PRODUKT', 'None'], ['None', 'None']]


def process_data(data):
    if all(item[1] is None for item in data):
        print('ii')
        return []

    else:
        print('q2222222')

        for sublist in data:

            if sublist[0] is None:
                print('jj')
                sublist = [item for item in data if item[0] is not None]
                return sublist

            elif None in sublist[1:]:
                print('pp')

                sublist = [[item[0], 0.001] if item[1] is None else item for item in data]

                return sublist
        else:
            print('ee')
            return data

        return result


print(process_data(data))

# logging.basicConfig(filename='egggggg.log', filemode='w', level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

def fo(name):
    print("eeeeeeeee")
    if name == 9:
        logger1 = logging.getLogger('logger1')
        logger1.setLevel(logging.INFO)
        # Настройте обработчик для первого логгера
        file_handler1 = logging.FileHandler('log_file1.log')
        file_handler1.setLevel(logging.INFO)
        logger1.addHandler(file_handler1)
        logger1.info('wwwwwwwwwwwwwwwwww 1')


        # Ваш код с циклами и условием для отлова ошибок




def ierty():
    for i in range(10):
        if i == 5:
            logger2 = logging.getLogger('logger2')
            logger2.setLevel(logging.INFO)
            file_handler2 = logging.FileHandler('log_file2.log')
            file_handler2.setLevel(logging.INFO)
            logger2.addHandler(file_handler2)
            logger2.info('gggggggggggggggggg 2')


            # Ваш код с циклами и условием для отлова ошибок



        fo(i)



ierty()

import logging


def create_logger(logger_name, log_file):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def function1():
    logger = create_logger('logger1', 'log_file1.log')
    logger.info('Сообщение из функции 1')


def function2():
    logger = create_logger('logger2', 'log_file2.log')
    logger.info('Сообщение из функции 2')


# Вызов функций
function1()
function2()
