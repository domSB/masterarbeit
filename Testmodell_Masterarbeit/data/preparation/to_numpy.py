import os

import numpy as np
import pandas as pd


def extend_list(list_of_str, length):
    list_copy = list_of_str.copy()
    for i in range(1, length):
        list_of_str.extend([name + '_' + str(i) for name in list_copy])
    return list_of_str


def concat(df, length):
    cols = list(df.columns)
    cols = extend_list(cols, length)
    df_s = df.copy()
    for i in range(1, length):
        df = pd.concat((df, df_s.shift(-i)), axis=1)
    df.columns = cols
    df.dropna(axis=0, inplace=True)
    for i in range(1, length):
        df = df[df['Artikel'] == df['Artikel_' + str(i)]]
    for i in range(1, length):
        df = df[df['Markt'] == df['Markt_' + str(i)]]
    x_cols = ['Menge', 'MaxTemp_1D', 'MinTemp_1D', 'Wolken_1D', 'Regen_1D',
              'MaxTemp_2D', 'MinTemp_2D', 'Wolken_2D', 'Regen_2D', 'Preis', 'relRabatt', 'absRabatt']
    x_cols = extend_list(x_cols, length)
    weekday_col = ['Wochentag']
    weekday_col = extend_list(weekday_col, length)
    yearweek_col = ['Kalenderwoche']
    yearweek_col = extend_list(yearweek_col, length)
    y_cols = ['in1', 'in2', 'in3', 'in4', 'in5']
    x_arr = df[x_cols].to_numpy(dtype=np.float32).reshape(-1, length, int(len(x_cols) / length))
    weekday_arr = df[weekday_col].to_numpy(dtype=np.float32).reshape(-1, length, 1)
    weekday_arr = to_categorical(weekday_arr, num_classes=7)
    yearweek_arr = df[yearweek_col].to_numpy(dtype=np.float32).reshape(-1, length, 1)
    yearweek_arr = to_categorical(yearweek_arr, num_classes=54)
    big_x_arr = np.concatenate((x_arr, weekday_arr, yearweek_arr), axis=2)
    y_arr = df[y_cols].to_numpy(dtype=np.float32)
    stat_df = df['Artikel']
    return y_arr, big_x_arr, stat_df


def create_numpy_from_frame(params, absatz, artikelstamm):
    """
    Erstellt 3 Numpy-Arrays für den Prädiktor, speichert diese in einer .npz-Datei und gibt sie zum direkten Verarbeiten
    auch weiter.
    :param params:
    :param absatz: Absatz-Frame
    :param artikelstamm Artikelstamm-Frame
    :return:
    """
    absatz.drop(columns=['Datum'], inplace=True)
    print('INFO - Concatenating dynamic states')
    lab, dyn, stat_df = concat(absatz, 6)
    print('INFO - Reindexing static state')
    stat_df = artikelstamm.reindex(stat_df)
    assert not stat_df.isna().any().any(), 'NaNs im Artikelstamm'
    print('INFO - Creating categorical states')
    stat = stat_df.loc[:, params.stat_state_scalar_cols].to_numpy(dtype=np.int8)
    for category, class_numbers in params.stat_state_category_cols.items():
        category_state = to_categorical(stat_df.loc[:, category], num_classes=class_numbers).astype(np.int8)
        stat = np.concatenate((stat, category_state), axis=1)
    print('INFO - Speichere NPZ-Dateien')
    filename = str(params.warengruppenmaske) + ' .npz'
    path = os.path.join(params.output_dir, filename)
    np.savez(path, lab=lab, dyn=dyn, stat=stat)
    return lab, dyn, stat