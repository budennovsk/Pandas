for meas in measures_list:
    # Получение первых элементов каждого кортежа
    top_m = [x[0] for x in result_top_m2]
    print('top_m', top_m)
    # ['MARS', 'Объединенные кондитеры', 'STORCK', 'FERRERO', 'KONTI', 'АККОНД', 'СЛАВЯНКА', 'МОК-производство', 'MONDELEZE', 'PERFETTI VAN MELLE']

    # SUM(Sales_kg) YearNo = 2022 AND MonthNo = 12  AND Articul3 = 'HYPER'  AND Group_name = 'Конфеты'  AND Format_TT = 'Дискаунтер'
    result_top_m = get_monthly_data_top_m(cursor, last_year, last_month, '', meas,
                                          first_three_pairs, top_m)
    result_top_m_prev = get_monthly_data_top_m(cursor, prev_year, prev_month, '', meas,
                                               first_three_pairs, top_m)
    param_1 = 'NW'
    result_top_m_NW_prev= get_monthly_data_top_m(cursor, prev_year, prev_month, '', param_1,
                                               first_three_pairs, top_m)


    # дата фреим создаю по NW prev
    data_dicts_prev_NW = [{'Articul3': x[0], 'Group_name': x[1], 'Format_TT': x[2],
                           f'Prev_Month_{ver}': x[3], 'Manufacture': x[4]}
                          for x in result_top_m_NW_prev]

    # Создание DataFrame
    prev_month_data_NW = pd.DataFrame(data_dicts_prev_NW)
    print('prev_month_data_NW', prev_month_data_NW)

    result_top_m_NW_last = get_monthly_data_top_m(cursor, last_year, last_month, '', param_1,
                                                  first_three_pairs, top_m)


    # дата фреим создаю по NW last
    data_dicts_last_NW = [
        {'Articul3': x[0], 'Group_name': x[1], 'Format_TT': x[2], f'Last_Month_{ver}': x[3],
         'Manufacture': x[4]}
        for x
        in
        result_top_m_NW_last]

    # Создание DataFrame NW
    last_month_data_NW = pd.DataFrame(data_dicts_last_NW)
    print('last_month_data_NW', last_month_data_NW)

    # Соедините два DataFrame NW  с использованием outer join
    merged_data_top_m_NW = pd.merge(last_month_data_NW, prev_month_data_NW,
                             on=['Articul3', 'Group_name', 'Format_TT','Manufacture'], how='outer')

    # Заполните пропущенные значения нулями
    merged_data_top_m_NW.fillna(0, inplace=True)
    print('merged_data_top_m_NW',merged_data_top_m_NW)

    # join df1 & df2
    merged_data_NM = merged_data.drop('Sales_Difference', axis=1)
    merged_data_NM = merged_data_NM.drop('Sales_Percentage_Change', axis=1)
    print('fd1',merged_data_NM)
    merged_data_2m_NM = merged_data_top_m_NW.merge(merged_data_NM, on=['Articul3', 'Group_name', 'Format_TT'],
                                         how='left',
                                         suffixes=('_m', '_r'))
    print('merged_data_2m_NM')
    print(merged_data_2m_NM)
