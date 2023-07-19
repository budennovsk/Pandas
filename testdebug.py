import re
import datetime
import pytz
import os

def test():
    # result_top_m = [('X5 Retail Group Private Label', 1996707.6139999633), ('MONDELEZE', 1549518.00800006),
    #                 ('Сладкая Слобода', 561580.9600000036)]
    # result_top_m_dict = dict(result_top_m) | {1: 'ddd'}
    # print(result_top_m_dict)
    list_dv = []
    res = {'kg': 'SUM(Sales_kg)', 'rub': 'SUM(Sales_rub)', 'Price': 'SUM(Sales_rub)/SUM(Sales_kg)'}

    ff = [{'kg': 'Sales_kg'}, {'rub': 'Sales_rub'}, {'Price': 'Sales_Price'},
          {'kg': 'Sales_Difference_kg'}, {'kg': 'Sales_Precentage_Change_kg'},
          {'rub': 'Sales_Difference_rub'}, {'rub': 'Sales_Precentage_Change_rub'},
          {'Price': 'Sales_Difference_Price'}, {'Price': 'Sales_Precentage_Change_Price'}]
    i = 0
    for i in res:
        print(i)

        # print(res.get(i))


        qwer = ([k.get(i) for k in ff if k.get(i)])
        for h in qwer:
            print(h, 'fff')

            list_dv.append(h)

    print(list_dv)

def ew():
    for i in range(10):
        if i >=3:
            continue
        print(i)
def req():
    name_dir = datetime.datetime.now(pytz.timezone('Europe/Moscow')).strftime('%m-%d-%y %H-%M')
    if not os.path.isdir(name_dir):
        os.mkdir(name_dir)

    # os.chdir(name_dir)
    c = os.getcwd()
    print(c)
    os.chdir(f'{c}/{name_dir}')
    print(os.getcwd())

    if not os.path.exists('sasa'):
        os.mkdir('sasa')
if __name__ == "__main__":
    # test()
    # ew()
    req()