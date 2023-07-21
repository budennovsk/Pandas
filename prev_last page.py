last_year, last_month = get_last_month(cursor)

# print(last_year, last_month)
last_month = 3
print('входные данные', last_year, last_month)
PERIOD_REQUEST = 'QY'
dict_request = {'M': 1, 'Q': 3, 'P': 6, 'Y': 12, 'MY': 1, 'QY': 3, 'PY': 6}
num_period = dict_request[PERIOD_REQUEST]
if len(PERIOD_REQUEST) == 1:
    print(PERIOD_REQUEST, dict_request[PERIOD_REQUEST])
    num_period = dict_request[PERIOD_REQUEST]
    # Получаем предыдущий месяц
    prev_year, prev_month = last_year, last_month - num_period
    if prev_month < 0:
        print('-1.1')
        prev_month = 12 - abs(last_month - num_period) + 1
        prev_year -= 1
    elif prev_month == 0 and num_period == 1:
        print('0,0')
        prev_month = 12 - abs(last_month - num_period)
        prev_year -= 1
    elif num_period == last_month:
        print('0=0')
        prev_month = last_month - num_period + 1
    elif prev_month > 0 and num_period != 1:
        print('1')
        prev_month = last_month - num_period + 1

    print('last', last_year, last_month)
    print('pref', prev_year, prev_month)
else:
    prev_year, prev_month = last_year, last_month - num_period
    print(PERIOD_REQUEST)
    print(dict_request[PERIOD_REQUEST])

    if prev_month < 0:
        print('-1.1 -t')
        prev_month = 12 - abs(last_month - num_period) + 1
        prev_year -= 1
    elif prev_month == 0 and num_period == 1:
        print('0,0 -t')
        prev_month = 12 - abs(last_month - num_period)
        prev_year -= 1
    elif num_period == last_month:
        print('0=0-t')
        prev_month = last_month - num_period + 1
    elif prev_month > 0 and num_period != 1:
        print('1-t')
        prev_month = last_month - num_period + 1

    between_last = [(prev_year, prev_month), (last_year, last_month)]
    between_prev = [(prev_year - 1, prev_month), (last_year - 1, last_month)]
    print('pref year', between_last)
    print('___')
    print('last year', between_prev)
