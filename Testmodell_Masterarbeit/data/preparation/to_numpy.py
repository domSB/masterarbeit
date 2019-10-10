import os

import numpy as np
import pandas as pd
from keras.utils import to_categorical


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
              'MaxTemp_2D', 'MinTemp_2D', 'Wolken_2D', 'Regen_2D', 'Preis', 'relRabatt', 'absRabatt',
              'j', 'q1', 'q2', 'q_m', 'w1', 'w2', 't1', 't2', 't3']
    x_cols = extend_list(x_cols, length)
    y_cols = ['in1', 'in2', 'in3', 'in4', 'in5', 'in6']
    x_arr = df[x_cols].to_numpy(dtype=np.float32).reshape(-1, length, int(len(x_cols) / length))
    y_arr = df[y_cols].to_numpy(dtype=np.float32)
    stat_df = df['Artikel']
    return y_arr, x_arr, stat_df


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
    stat_state_scalar_cols = ['Eigenmarke', 'GuG', 'OSE', 'Saisonal', 'Kern', 'Bio', 'Glutenfrei', 'Laktosefrei']
    stat = stat_df.loc[:, stat_state_scalar_cols].to_numpy(dtype=np.int8)
    for category, class_numbers in params.stat_state_category_cols.items():
        category_state = to_categorical(stat_df.loc[:, category], num_classes=class_numbers).astype(np.int8)
        stat = np.concatenate((stat, category_state), axis=1)
    print('INFO - Speichere NPZ-Dateien')
    return lab, dyn, stat
