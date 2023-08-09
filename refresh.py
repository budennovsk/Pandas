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
    result_top_m_NW_prev = get_monthly_data_top_m(cursor, prev_year, prev_month, '', param_1,
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
                                    on=['Articul3', 'Group_name', 'Format_TT', 'Manufacture'], how='outer')

    # Заполните пропущенные значения нулями
    merged_data_top_m_NW.fillna(0, inplace=True)
    print('merged_data_top_m_NW', merged_data_top_m_NW)

    # join df1 & df2
    merged_data_NM = merged_data.drop('Sales_Difference', axis=1)
    merged_data_NM = merged_data_NM.drop('Sales_Percentage_Change', axis=1)
    print('fd1', merged_data_NM)
    merged_data_2m_NM = merged_data_top_m_NW.merge(merged_data_NM, on=['Articul3', 'Group_name', 'Format_TT'],
                                                   how='left',
                                                   suffixes=('_m', '_r'))
    print('merged_data_2m_NM')
    print(merged_data_2m_NM)

if meas == 'SUM(Sales_rub)':
    last_result_market = get_items_market(cursor=cursor,
                                          year=last_year,
                                          month=last_month,
                                          meas=meas,
                                          first_three_pairs=first_three_pairs,
                                          manufactur=Manufacture_m)
    last_result_market_dict = [
        {'Articul3': x[0], 'Group_name': x[1], 'Format_TT': x[2],
         f'MERA': x[3]}
        for x in last_result_market]
    result_market_df = pd.DataFrame(last_result_market_dict)

    # ______
    last_month_result_measures_name = get_monthly_data(cursor, last_year, last_month, Manufacture,
                                                       meas)
    prev_month_result_measures_name = get_monthly_data(cursor, prev_year, prev_month, Manufacture,
                                                       meas)
    # _____

    data_dicts = [
        {'Articul3': x[0], 'Group_name': x[1], 'Format_TT': x[2],
         f'Last_Month_{name_measures_list[i]}': x[3]}
        for x in last_month_result_measures_name]
    data_dicts_prev = [
        {'Articul3': x[0], 'Group_name': x[1], 'Format_TT': x[2],
         f'Prev_Month_{name_measures_list[i]}': x[3]}
        for x in prev_month_result_measures_name]
    prev_month_data = pd.DataFrame(data_dicts_prev)
    last_month_data = pd.DataFrame(data_dicts)
    merged_data_VER = pd.merge(last_month_data, prev_month_data,
                               on=['Articul3', 'Group_name', 'Format_TT'], how='outer')

    merged_data_2m_SA = result_market_df.merge(merged_data_VER,
                                               on=['Articul3', 'Group_name', 'Format_TT'],
                                               how='left',
                                               suffixes=('_m', '_r'))

    market_al = pd.DataFrame(merged_data_2m_SA['Last_Month_Sales_rub'])
    market_al.columns = ['Value']

    csa = len(merged_result_top_m_difference.index)
    repat = pd.concat([market_al] * csa, ignore_index=True)

    two_in = pd.concat([repat, get_list_name], axis=1)
    print('two_in', two_in)
    two_in = two_in.applymap(
        lambda s: s.lower().title() if type(s) == str else s)
    df_transonse_set_index(
        'Measures',
        'MARKET_SALES',
        df=two_in,
        name_field_index=name_measures_list[i],
        list_app=list_df,
        result_top_m2_col=result_top_m2_df.columns
    )
