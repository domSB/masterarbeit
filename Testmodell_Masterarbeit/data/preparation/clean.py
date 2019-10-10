import os
import numpy as np
import pandas as pd
import json
from calender.german_holidays import get_german_holiday_calendar
from keras.utils import to_categorical


def quarters(q):
    first = int(q/2.5)
    if 1 < q < 4:
        second = 1
    else:
        second = 0
    return first, second


def quarter_month(m):
    """ Erstellt ein One-Hot-Encoding für das Quartal"""
    q_m = m % 3 - 2
    if q_m == -2:
        q_m = 1
    return q_m


def week_of_month(date):
    """ Erstellt One-Hot-Encoding für die Woche des Monats"""
    months_first = pd.Timestamp(year=date.year, month=date.month, day=1)
    month_week = date.weekofyear - months_first.weekofyear
    if month_week == 0:
        return -1, -1
    elif month_week == 1:
        return -1, 0
    elif month_week == 2:
        return 0, 0
    elif month_week == 3:
        return 0, 1
    else:
        return 1, 1


def day_of_week(day):
    """ Erstellt One-Hot-Encoding für den Wochentag"""
    if day == 0:
        return -1, -1, -1
    elif day == 1:
        return -1, -1, 0
    elif day == 2:
        return -1, 0, 0
    elif day == 3:
        return 0, 0, 0
    elif day == 4:
        return 1, 0, 0
    elif day == 5:
        return 1, 1, 0
    else:
        return 1, 1, 1


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


def create_frame_from_raw_data(params):
    """
    Returns Absatz-Frame, Bewegung-Frame & Artikelstamm-Frame und speichert die Frames in einem HDF-Store
    :param params:
    :return:
    """
    warengruppenstamm = pd.read_csv(
        os.path.join(params.input_dir, '0 Warengruppenstamm.csv'),
        header=1,
        names=['WG', 'WGNr', 'WGBez', 'Abt', 'AbtNr', 'AbtBez']
    )

    artikelstamm = pd.read_csv(
        os.path.join(params.input_dir, '0 ArtikelstammV4.csv'),
        header=0,
        names=['Artikel', 'Warengruppe', 'Detailwarengruppe', 'Bezeichnung',
               'Eigenmarke', 'Einheit', 'Verkaufseinheit', 'MHD',
               'GuG', 'OSE', 'OSEText', 'Saisonal',
               'Kern', 'Bio', 'Glutenfrei',
               'Laktosefrei', 'MarkeFK', 'Region']
    )
    artikelstamm = artikelstamm[artikelstamm.Warengruppe.isin(params.warengruppenmaske)]
    artikelmaske = pd.unique(artikelstamm.Artikel)
    # endregion

    # region Warenbewegung
    warenausgang = pd.read_csv(
        os.path.join(params.input_dir, '0 Warenausgang.Markt.csv'),
        header=1,
        names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
    )
    warenausgang = warenausgang[warenausgang.Artikel.isin(artikelmaske)]
    warenausgang['Datum'] = pd.to_datetime(warenausgang['Datum'], format='%d.%m.%y')
    absatz = warenausgang[warenausgang.Belegtyp.isin(['UMSATZ_SCANNING', 'UMSATZ_AKTION'])]
    absatz = absatz.groupby(['Markt', 'Artikel', 'Datum'], as_index=False).sum()

    # Verwerfe Artikel, die nicht jedes Jahr verkauft wurden
    absatz['Jahr'] = absatz.Datum.dt.year
    absatz_jahre = absatz.groupby(['Artikel', 'Markt'])['Jahr'].nunique()
    max_abs_jahre = max(absatz_jahre)
    artikel_mit_durchgehendem_abs = absatz_jahre[absatz_jahre == max_abs_jahre].index.values
    absatz.set_index(['Artikel', 'Markt'], drop=False, inplace=True)
    absatz = absatz.loc[artikel_mit_durchgehendem_abs]
    absatz.reset_index(drop=True, inplace=True)
    # Verwerfe Artikel, die seltener als an 10 Tage p.a. verkauft werden.
    absatz_vorgaenge = absatz.groupby(['Artikel', 'Jahr', 'Markt'])['Menge'].count()
    art_abs_unter_10_art = absatz_vorgaenge[absatz_vorgaenge > 10].index.get_level_values('Artikel').values
    art_abs_unter_10_mkt = absatz_vorgaenge[absatz_vorgaenge > 10].index.get_level_values('Markt').values
    art_abs_unter_10 = [(art, mkt) for art, mkt in zip(art_abs_unter_10_art, art_abs_unter_10_mkt)]
    art_abs_unter_10 = set(art_abs_unter_10)
    absatz.set_index(['Artikel', 'Markt'], drop=False, inplace=True)
    absatz = absatz.loc[list(art_abs_unter_10)]
    absatz.reset_index(drop=True, inplace=True)

    # Artikelmaske auf relevante beschränken
    artikelmaske = absatz.Artikel.unique()
    warenausgang = warenausgang[warenausgang.Artikel.isin(artikelmaske)]
    artikelstamm = artikelstamm[artikelstamm.Artikel.isin(artikelmaske)]

    wareneingang = pd.read_csv(
        os.path.join(params.input_dir, '0 Wareneingang.Markt.csv'),
        header=1,
        names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
    )
    wareneingang = wareneingang[wareneingang.Artikel.isin(artikelmaske)]
    wareneingang['Datum'] = pd.to_datetime(wareneingang['Datum'], format='%d.%m.%y')

    bestand = pd.read_csv(
        os.path.join(params.input_dir, '0 Warenbestand.Markt.csv'),
        header=1,
        names=['Markt', 'Artikel', 'Bestand', 'EK', 'VK', 'Anfangsbestand', 'Datum']
    )
    bestand['Datum'] = pd.to_datetime(bestand['Datum'], format='%d.%m.%y')

    warenausgang['Menge'] = -warenausgang['Menge']

    bewegung = pd.concat([warenausgang, wareneingang])
    # endregion

    # region Preise
    preise = pd.read_csv(
        os.path.join(params.input_dir, '0 Preise.Markt.csv'),
        header=1,
        names=['Preis', 'Artikel', 'Datum']
    )
    preise = preise[preise.Artikel.isin(artikelmaske)]
    preise.drop_duplicates(inplace=True)
    preise = preise[(preise.Preis < 100) & (preise.Preis != 99.99)]
    # TODO: Preise mit relativem Vergleich in der Gruppe sortieren
    preise['Datum'] = pd.to_datetime(preise['Datum'], format='%d.%m.%Y')
    if 'Markt' not in preise.columns:
        preise['Markt'] = 5

    aktionspreise = pd.read_csv(
        os.path.join(params.input_dir, '0 Aktionspreise.Markt.csv'),
        header=1,
        names=['Artikel', 'DatumAb', 'DatumBis', 'Aktionspreis']
    )
    aktionspreise = aktionspreise[aktionspreise.Artikel.isin(artikelmaske)]
    aktionspreise['DatumAb'] = pd.to_datetime(aktionspreise['DatumAb'], format='%d.%m.%Y')
    aktionspreise['DatumBis'] = pd.to_datetime(aktionspreise['DatumBis'], format='%d.%m.%Y')
    # endregion

    # region Wetter
    wetter = pd.read_csv(
        os.path.join(params.input_dir, '1 Wetter.csv')
    )
    wetter.drop(columns=['Unnamed: 0'], inplace=True)
    # endregion

    artikelstamm = artikelstamm.set_index('Artikel')

    # TODO: Feiertage Hinweis in State aufnehmen
    # region fehlende Detailwarengruppen auffüllen
    wg_group = artikelstamm.loc[
               :,
               ['Warengruppe', 'Detailwarengruppe']
               ].groupby('Warengruppe').median()
    detail_warengruppen_nan_index = wg_group.to_dict()['Detailwarengruppe']
    artikelstamm['DetailwarengruppeBackup'] = artikelstamm['Warengruppe'].map(
        detail_warengruppen_nan_index
    )
    artikelstamm['Detailwarengruppe'].fillna(
        value=artikelstamm['DetailwarengruppeBackup'],
        inplace=True
    )
    artikelstamm.drop(columns=['DetailwarengruppeBackup'], inplace=True)
    # endregion

    # region numerisches MHD in kategoriale Variable transformieren
    mhd_labels = [0, 1, 2, 3, 4, 5, 6]
    mhd_bins = [0, 1, 7, 14, 28, 100, 1000, 100000]
    artikelstamm['MHDgroup'] = pd.cut(artikelstamm.MHD, mhd_bins, right=False, labels=mhd_labels)
    # endregion

    #  region Lückenhafte Fremdschlüssel durch eine durchgehende ID ersetzen
    detail_warengruppen_index = {
        int(value): index for index, value in enumerate(np.sort(pd.unique(artikelstamm.Detailwarengruppe)))
    }
    params.stat_state_category_cols['Detailwarengruppe'] = len(detail_warengruppen_index)
    warengruppen_index = {
        int(value): index for index, value in enumerate(np.sort(pd.unique(artikelstamm.Warengruppe)))
    }
    einheit_index = {
        int(value): index for index, value in enumerate(np.sort(pd.unique(artikelstamm.Einheit)))
    }
    params.stat_state_category_cols['Einheit'] = len(einheit_index)
    mapping = {
        'Detailwarengruppe': detail_warengruppen_index,
        'Warengruppe': warengruppen_index,
        'Einheit': einheit_index
    }
    filename = str(params.warengruppenmaske) + ' ValueMapping.json'
    with open(os.path.join(params.output_dir, filename), 'w') as file:
        json.dump(mapping, file)
    artikelstamm['Detailwarengruppe'] = artikelstamm['Detailwarengruppe'].map(
        detail_warengruppen_index)
    artikelstamm['Warengruppe'] = artikelstamm['Warengruppe'].map(warengruppen_index)
    artikelstamm['Einheit'] = artikelstamm['Einheit'].map(einheit_index)
    # endregion

    # region überflüssige Spalten löschen und OSE&Saisonal Kennzeichen auffüllen
    artikelstamm.drop(columns=['MHD', 'Region', 'MarkeFK', 'Verkaufseinheit', 'OSEText'], inplace=True)
    artikelstamm['OSE'].fillna(0, inplace=True)
    artikelstamm['Saisonal'].fillna(0, inplace=True)
    # endregion

    # region Reindexieren des Absatzes
    cal_cls = get_german_holiday_calendar('SL')
    cal = cal_cls()
    sl_bd = pd.tseries.offsets.CustomBusinessDay(calendar=cal, weekmask='Mon Tue Wed Tue Fri Sat')
    zeitraum = pd.date_range(
        pd.to_datetime('2018-01-01'),
        pd.to_datetime('2019-07-01') + pd.DateOffset(7),
        freq=sl_bd
    )
    absatz.set_index('Datum', inplace=True)
    absatz = absatz.groupby(['Markt', 'Artikel']).apply(lambda x: x.reindex(zeitraum, fill_value=0))
    absatz.drop(columns=['Markt', 'Artikel'], inplace=True)
    absatz.reset_index(inplace=True)
    absatz.rename(columns={'level_2': 'Datum'}, inplace=True)

    absatz['j'] = absatz.Datum.dt.year - 2018
    absatz['q1'], absatz['q2'] = zip(*absatz.Datum.dt.quarter.apply(quarters))
    absatz['q_m'] = absatz.Datum.dt.quarter.apply(quarter_month)
    absatz['w1'], absatz['w2'] = zip(*absatz.Datum.apply(week_of_month))
    absatz['t1'], absatz['t2'], absatz['t3'] = zip(*absatz.Datum.dt.dayofweek.apply(day_of_week))

    # absatz['Wochentag'] = absatz.Datum.dt.dayofweek
    # absatz['Kalenderwoche'] = absatz.Datum.dt.weekofyear
    absatz["UNIXDatum"] = absatz["Datum"].astype(np.int64) / (1000000000 * 24 * 3600)
    # endregion

    # region Wetter anfügen
    wetter["date_shifted_oneday"] = wetter["date"] - 1
    wetter["date_shifted_twodays"] = wetter["date"] - 2
    absatz = pd.merge(
        absatz,
        wetter,
        left_on='UNIXDatum',
        right_on='date_shifted_oneday'
    )
    absatz = pd.merge(
        absatz,
        wetter,
        left_on='UNIXDatum',
        right_on='date_shifted_twodays',
        suffixes=('_1D', '_2D')
    )
    absatz.drop(
        columns=["date_shifted_oneday_1D",
                 "date_shifted_twodays_1D",
                 "date_shifted_oneday_2D",
                 "date_shifted_twodays_2D",
                 "date_1D",
                 "date_2D"
                 ],
        inplace=True
    )
    # endregion

    # region reguläre Preise aufbereiten
    preise.sort_values(by=['Datum', 'Artikel'], inplace=True)
    # pd.merge_asof ist ein Left Join mit dem nächsten passenden Key.
    # Standardmäßig wird in der rechten Tabelle der Gleiche oder nächste Kleinere gesucht.
    absatz = pd.merge_asof(
        absatz,
        preise.loc[:, ["Preis", "Artikel", "Datum"]].copy(),
        left_on='Datum',
        right_on='Datum',
        by='Artikel'
    )
    absatz['Preis'] = absatz.groupby(['Markt', 'Artikel'])['Preis'].fillna(method='bfill')
    neuere_preise = preise.groupby('Artikel').last()
    neuere_preise.drop(columns=['Datum', 'Markt'], inplace=True)
    neuere_preise_index = neuere_preise.to_dict()['Preis']
    absatz['PreisBackup'] = absatz['Artikel'].map(
        neuere_preise_index
    )
    absatz['Preis'].fillna(
        value=absatz['PreisBackup'],
        inplace=True
    )
    absatz.drop(columns=['PreisBackup'], inplace=True)
    print('{:.2f} % der Daten aufgrund fehlender Preise verworfen.'.format(np.mean(absatz.Preis.isna()) * 100))
    preis_mean, preis_std = np.mean(absatz.Preis), np.std(absatz.Preis)
    absatz['Preis'] = (absatz['Preis'] - preis_mean) / preis_std
    filename = str(params.warengruppenmaske) + ' PreisStd.json'
    with open(os.path.join(params.output_dir, filename), 'w') as file:
        json.dump({'PreisStandardDerivation': preis_std, 'PreisMean': preis_mean}, file)
    absatz.dropna(inplace=True)
    # endregion

    # region Aktionspreise aufbereiten
    aktionspreise.sort_values(by=['DatumAb', 'DatumBis', 'Artikel'], inplace=True)
    len_vor = absatz.shape[0]
    absatz = pd.merge_asof(
        absatz,
        aktionspreise,
        left_on='Datum',
        right_on='DatumAb',
        tolerance=pd.Timedelta('9d'),
        by='Artikel')
    len_nach = absatz.shape[0]
    assert len_vor == len_nach, 'Anfügen der Aktionspreise hat zu einer Verlängerung der Absätze geführt.'
    absatz['Aktionspreis'].where(~(absatz.DatumBis < absatz.Datum), inplace=True)
    absatz['absRabatt'] = absatz.Preis - absatz.Aktionspreis
    absatz['relRabatt'] = absatz.absRabatt / absatz.Preis
    absatz.relRabatt.fillna(0., inplace=True)
    absatz.absRabatt.fillna(0., inplace=True)
    absatz.drop(columns=['DatumAb', 'DatumBis', 'Aktionspreis'], inplace=True)
    # endregion

    # region Targets erzeugen
    absatz['in1'] = absatz.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-1)
    absatz['in2'] = absatz.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-2)
    absatz['in3'] = absatz.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-3)
    absatz['in4'] = absatz.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-4)
    absatz['in5'] = absatz.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-5)
    absatz['in6'] = absatz.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-6)
    absatz.dropna(axis=0, inplace=True)
    absatz.sort_values(['Markt', 'Artikel', 'Datum'], inplace=True)
    # endregion
    filename = str(params.warengruppenmaske) + ' store.h5'
    store = pd.HDFStore(os.path.join(params.output_dir, filename))
    store.put('Artikelstamm', artikelstamm)
    store.put('Absatz', absatz)
    store.put('Bewegung', bewegung)
    store.close()
    # TODO: Bestand und weitere Stammdaten für Statistiken zurückgeben
    # TODO: Checken, ob das Updaten der Dicts zu kategorialen Variable Einfluss auf den Dateinamen hat.
    return absatz, bewegung, artikelstamm


