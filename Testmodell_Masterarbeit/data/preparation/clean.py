import os
import numpy as np
import pandas as pd
import json
from calender.german_holidays import get_german_holiday_calendar


def quarters(q):
    """ Erstellt ein One-Hot-Encoding für das Quartal"""
    first = int(q/2.5)
    if 1 < q < 4:
        second = 1
    else:
        second = 0
    return first, second


def quarter_month(m):
    """ Erstellt ein One-Hot-Encoding für den Monat des Quartals"""
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


def create_frame_from_raw_data(params):
    """
    Returns Absatz-Frame, Bewegung-Frame & Artikelstamm-Frame
    :param params:
    :return:
    """
    print('Lese Artikelstammdaten')
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
    print('Lese Warenbewegungen ein')
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
    print('Lese Preise ein')
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
    print('Lese Wetter ein')
    wetter = pd.read_csv(
        os.path.join(params.input_dir, '1 Wetter.csv')
    )
    wetter.drop(columns=['Unnamed: 0'], inplace=True)
    # endregion

    artikelstamm = artikelstamm.set_index('Artikel')

    # region fehlende Detailwarengruppen auffüllen
    print('Beginne mit Aufbereitungsmaßnahmen')
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
    absatz["UNIXDatum"] = absatz["Datum"].astype(np.int64) / (1000000000 * 24 * 3600)
    # TODO: Feiertage Hinweis in State aufnehmen
    # endregion

    # region Wetter anfügen
    print('Starte mit dem Zusammenfügen der Tabellen')
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
    print('Frames sind erstellt')
    # TODO: Bestand und weitere Stammdaten für Statistiken zurückgeben
    # TODO: Checken, ob das Updaten der Dicts zu kategorialen Variable Einfluss auf den Dateinamen hat.
    return absatz, bewegung, artikelstamm


