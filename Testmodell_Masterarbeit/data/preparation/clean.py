import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from calender.german_holidays import get_german_holiday_calendar
from keras.utils import to_categorical
if os.path.split(os.getcwd())[1] == 'preparation':
    # Falls clean.py nicht mit wdir-Parameter gestartet wird.
    print('Changed current directory')
    os.chdir(os.path.join('..', '..'))
    print(os.getcwd())
else:
    print(os.getcwd())


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


class Datapipeline(object):
    def __init__(self, **kwargs):
        self.input_path = kwargs['InputPath']
        self.output_path = kwargs['OutputPath']
        self.type = kwargs['Type']
        self.artikelstamm = None
        self.warengruppenstamm = None
        self.artikelmaske = None
        self.warengruppenmaske = kwargs['ZielWarengruppen']
        self.preise = None
        self.aktionspreise = None
        self.bestand = None
        self.absatz = None
        self.bewegung = None
        self.wetter = None
        self.dyn_state_scalar_cols = kwargs['DynStateScalarCols']
        self.dyn_state_label_cols = kwargs['DynStateLabelCols']
        self.dyn_state_category_cols = kwargs['DynStateCategoryCols']
        self.stat_state_scalar_cols = kwargs['StatStateScalarCols']
        self.stat_state_category_cols = kwargs['StatStateCategoryCols']

    def read_files(self):
        # region Stammdaten
        self.warengruppenstamm = pd.read_csv(
            os.path.join(self.input_path, '0 Warengruppenstamm.csv'),
            header=1,
            names=['WG', 'WGNr', 'WGBez', 'Abt', 'AbtNr', 'AbtBez']
        )

        artikelstamm = pd.read_csv(
            os.path.join(self.input_path, '0 ArtikelstammV4.csv'),
            header=0,
            names=['Artikel', 'Warengruppe', 'Detailwarengruppe', 'Bezeichnung',
                   'Eigenmarke', 'Einheit', 'Verkaufseinheit', 'MHD',
                   'GuG', 'OSE', 'OSEText', 'Saisonal',
                   'Kern', 'Bio', 'Glutenfrei',
                   'Laktosefrei', 'MarkeFK', 'Region']
            )
        self.artikelstamm = artikelstamm[artikelstamm.Warengruppe.isin(self.warengruppenmaske)]
        self.artikelmaske = pd.unique(self.artikelstamm.Artikel)
        # endregion

        # region Preise
        self.preise = pd.read_csv(
            os.path.join(self.input_path, '0 Preise.' + self.type + '.csv'),
            header=1,
            names=['Preis', 'Artikel', 'Datum']
        )
        self.preise = self.preise[self.preise.Artikel.isin(self.artikelmaske)]
        self.preise.drop_duplicates(inplace=True)
        self.preise = self.preise[(self.preise.Preis < 100) & (self.preise.Preis != 99.99)]
        # TODO: Preise mit relativem Vergleich in der Gruppe sortieren
        self.preise['Datum'] = pd.to_datetime(self.preise['Datum'], format='%d.%m.%Y')
        if 'Markt' not in self.preise.columns:
            self.preise['Markt'] = 5

        self.aktionspreise = pd.read_csv(
            os.path.join(self.input_path, '0 Aktionspreise.' + self.type + '.csv'),
            header=1,
            names=['Artikel', 'DatumAb', 'DatumBis', 'Aktionspreis']
        )
        self.aktionspreise = self.aktionspreise[self.aktionspreise.Artikel.isin(self.artikelmaske)]
        self.aktionspreise['DatumAb'] = pd.to_datetime(self.aktionspreise['DatumAb'], format='%d.%m.%Y')
        self.aktionspreise['DatumBis'] = pd.to_datetime(self.aktionspreise['DatumBis'], format='%d.%m.%Y')
        # endregion

        # region Warenbewegung
        warenausgang = pd.read_csv(
            os.path.join(self.input_path, '0 Warenausgang.' + self.type + '.csv'),
            header=1,
            names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
        )
        warenausgang = warenausgang[warenausgang.Artikel.isin(self.artikelmaske)]
        warenausgang['Datum'] = pd.to_datetime(warenausgang['Datum'], format='%d.%m.%y')

        wareneingang = pd.read_csv(
            os.path.join(self.input_path, '0 Wareneingang.' + self.type + '.csv'),
            header=1,
            names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
        )
        wareneingang = wareneingang[wareneingang.Artikel.isin(self.artikelmaske)]
        wareneingang['Datum'] = pd.to_datetime(wareneingang['Datum'], format='%d.%m.%y')

        self.bestand = pd.read_csv(
            os.path.join(self.input_path, '0 Warenbestand.' + self.type + '.csv'),
            header=1,
            names=['Markt', 'Artikel', 'Bestand', 'EK', 'VK', 'Anfangsbestand', 'Datum']
        )
        self.bestand['Datum'] = pd.to_datetime(self.bestand['Datum'], format='%d.%m.%y')

        self.absatz = warenausgang[warenausgang.Belegtyp.isin(['UMSATZ_SCANNING', 'UMSATZ_AKTION'])]
        self.absatz = self.absatz.groupby(['Markt', 'Artikel', 'Datum'],  as_index=False).sum()

        warenausgang['Menge'] = -warenausgang['Menge']

        self.bewegung = pd.concat([warenausgang, wareneingang])
        # endregion

        # region Wetter
        self.wetter = pd.read_csv(
            os.path.join(self.input_path, '1 Wetter.csv')
        )
        self.wetter.drop(columns=['Unnamed: 0'], inplace=True)
        # endregion

    def prepare_for_simulation(self):
        """
        Clean the data and prepare it, to be used in the Simulation.
        """
    pass

    def prepare_for_regression(self, **kwargs):
        # TODO: Feiertage Hinweis in State aufnehmen
        # region fehlende Detailwarengruppen auffüllen
        wg_group = self.artikelstamm.loc[
                   :,
                   ['Warengruppe', 'Detailwarengruppe']
                   ].groupby('Warengruppe').median()
        detail_warengruppen_nan_index = wg_group.to_dict()['Detailwarengruppe']
        self.artikelstamm['DetailwarengruppeBackup'] = self.artikelstamm['Warengruppe'].map(
            detail_warengruppen_nan_index
        )
        self.artikelstamm['Detailwarengruppe'].fillna(
            value=self.artikelstamm['DetailwarengruppeBackup'],
            inplace=True
        )
        self.artikelstamm.drop(columns=['DetailwarengruppeBackup'], inplace=True)
        # endregion

        # region numerisches MHD in kategoriale Variable transformieren
        mhd_labels = [0, 1, 2, 3, 4, 5, 6]
        mhd_bins = [0, 1, 7, 14, 28, 100, 1000, 100000]
        self.artikelstamm['MHDgroup'] = pd.cut(self.artikelstamm.MHD, mhd_bins, right=False, labels=mhd_labels)
        # endregion

        #  region Lückenhafte Fremdschlüssel durch eine durchgehende ID ersetzen
        detail_warengruppen_index = {
            int(value): index for index, value in enumerate(np.sort(pd.unique(self.artikelstamm.Detailwarengruppe)))
        }
        self.stat_state_category_cols['Detailwarengruppe'] = len(detail_warengruppen_index)
        warengruppen_index = {
            int(value): index for index, value in enumerate(np.sort(pd.unique(self.artikelstamm.Warengruppe)))
        }
        einheit_index = {
            int(value): index for index, value in enumerate(np.sort(pd.unique(self.artikelstamm.Einheit)))
        }
        self.stat_state_category_cols['Einheit'] = len(einheit_index)
        mapping = {
            'Detailwarengruppe': detail_warengruppen_index,
            'Warengruppe': warengruppen_index,
            'Einheit': einheit_index
        }
        filename = '-'.join([
            self.type,
            kwargs['StartDatum'],
            kwargs['EndDatum'],
            str(kwargs['StepSize']),
            'ValueMapping.json'
        ])
        with open(os.path.join(self.output_path, filename), 'w') as file:
            json.dump(mapping, file)
        self.artikelstamm['Detailwarengruppe'] = self.artikelstamm['Detailwarengruppe'].map(
            detail_warengruppen_index)
        self.artikelstamm['Warengruppe'] = self.artikelstamm['Warengruppe'].map(warengruppen_index)
        self.artikelstamm['Einheit'] = self.artikelstamm['Einheit'].map(einheit_index)
        # endregion

        # region überflüssige Spalten löschen und OSE&Saisonal Kennzeichen auffüllen
        self.artikelstamm.drop(columns=['MHD', 'Region', 'MarkeFK', 'Verkaufseinheit', 'OSEText'], inplace=True)
        self.artikelstamm['OSE'].fillna(0, inplace=True)
        self.artikelstamm['Saisonal'].fillna(0, inplace=True)
        # endregion

        # region Reindexieren des Absatzes
        cal_cls = get_german_holiday_calendar('SL')
        cal = cal_cls()
        sl_bd = pd.tseries.offsets.CustomBusinessDay(calendar=cal, weekmask='Mon Tue Wed Tue Fri Sat')
        zeitraum = pd.date_range(
            pd.to_datetime(kwargs['StartDatum']),
            pd.to_datetime(kwargs['EndDatum']) + pd.DateOffset(7),
            freq=sl_bd
        )
        self.absatz.set_index('Datum', inplace=True)
        self.absatz = self.absatz.groupby(['Markt', 'Artikel']).apply(lambda x: x.reindex(zeitraum, fill_value=0))
        self.absatz.drop(columns=['Markt', 'Artikel'], inplace=True)
        self.absatz.reset_index(inplace=True)
        self.absatz.rename(columns={'level_2': 'Datum'}, inplace=True)

        self.absatz['Wochentag'] = self.absatz.Datum.dt.dayofweek
        self.absatz['Kalenderwoche'] = self.absatz.Datum.dt.weekofyear
        self.absatz["UNIXDatum"] = self.absatz["Datum"].astype(np.int64)/(1000000000 * 24 * 3600)
        # endregion

        # region Wetter anfügen
        self.wetter["date_shifted_oneday"] = self.wetter["date"] - 1
        self.wetter["date_shifted_twodays"] = self.wetter["date"] - 2
        self.absatz = pd.merge(
            self.absatz,
            self.wetter,
            left_on='UNIXDatum',
            right_on='date_shifted_oneday'
        )
        self.absatz = pd.merge(
            self.absatz,
            self.wetter,
            left_on='UNIXDatum',
            right_on='date_shifted_twodays',
            suffixes=('_1D', '_2D')
        )
        self.absatz.drop(
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
        self.preise.sort_values(by=['Datum', 'Artikel'], inplace=True)
        # pd.merge_asof ist ein Left Join mit dem nächsten passenden Key.
        # Standardmäßig wird in der rechten Tabelle der Gleiche oder nächste Kleinere gesucht.
        self.absatz = pd.merge_asof(
            self.absatz,
            self.preise.loc[:, ["Preis", "Artikel", "Datum"]].copy(),
            left_on='Datum',
            right_on='Datum',
            by='Artikel'
        )
        self.absatz['Preis'] = self.absatz.groupby(['Markt', 'Artikel'])['Preis'].fillna(method='bfill')
        neuere_preise = self.preise.groupby('Artikel').last()
        neuere_preise.drop(columns=['Datum', 'Markt'], inplace=True)
        neuere_preise_index = neuere_preise.to_dict()['Preis']
        self.absatz['PreisBackup'] = self.absatz['Artikel'].map(
            neuere_preise_index
        )
        self.absatz['Preis'].fillna(
            value=self.absatz['PreisBackup'],
            inplace=True
        )
        self.absatz.drop(columns=['PreisBackup'], inplace=True)
        print('{:.2f} % der Daten aufgrund fehlender Preise verworfen.'.format(np.mean(time.absatz.Preis.isna()) * 100))
        preis_mean, preis_std = np.mean(self.absatz.Preis), np.std(self.absatz.Preis)
        self.absatz['Preis'] = (self.absatz['Preis'] - preis_mean) / preis_std
        filename = '-'.join([
            self.type,
            kwargs['StartDatum'],
            kwargs['EndDatum'],
            str(kwargs['StepSize']),
            'PreisStd.json'
        ])
        with open(os.path.join(self.output_path, filename), 'w') as file:
            json.dump({'PreisStandardDerivation': preis_std, 'PreisMean': preis_mean}, file)
        self.absatz.dropna(inplace=True)
        # endregion

        # region Aktionspreise aufbereiten
        self.aktionspreise.sort_values(by=['DatumAb', 'DatumBis', 'Artikel'], inplace=True)
        len_vor = self.absatz.shape[0]
        self.absatz = pd.merge_asof(
            self.absatz,
            self.aktionspreise,
            left_on='Datum',
            right_on='DatumAb',
            tolerance=pd.Timedelta('9d'),
            by='Artikel')
        len_nach = self.absatz.shape[0]
        assert len_vor == len_nach, 'Anfügen der Aktionspreise hat zu einer Verlängerung der Absätze geführt.'
        self.absatz['Aktionspreis'].where(~(self.absatz.DatumBis < self.absatz.Datum), inplace=True)
        self.absatz['absRabatt'] = self.absatz.Preis - self.absatz.Aktionspreis
        self.absatz['relRabatt'] = self.absatz.absRabatt / self.absatz.Preis
        self.absatz.relRabatt.fillna(0., inplace=True)
        self.absatz.absRabatt.fillna(0., inplace=True)
        self.absatz.drop(columns=['DatumAb', 'DatumBis', 'Aktionspreis'], inplace=True)
        # endregion

        # region Targets erzeugen
        self.absatz['in1'] = self.absatz.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-1)
        self.absatz['in2'] = self.absatz.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-2)
        self.absatz['in3'] = self.absatz.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-3)
        self.absatz['in4'] = self.absatz.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-4)
        self.absatz['in5'] = self.absatz.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-5)
        self.absatz.dropna(axis=0, inplace=True)
        self.absatz.sort_values(['Markt', 'Artikel', 'Datum'], inplace=True)
        # endregion
        assert not self.absatz.isna().any().any(), 'NaNs im Datensatz gefunden'

    def create_regression_numpy(self, **kwargs):
        self.prepare_for_regression(**kwargs)
        self.absatz.drop(columns=['Datum'], inplace=True)
        print('INFO - Concatenating dynamic states')
        y, x, stat_df = concat(self.absatz, kwargs['StepSize'])
        print('INFO - Reindexing static state')
        self. artikelstamm = self.artikelstamm.set_index('Artikel')
        stat_df = self.artikelstamm.reindex(stat_df)
        assert not stat_df.isna().any().any(), 'NaNs im Artikelstamm'
        print('INFO - Creating categorical states')
        stat_state = stat_df.loc[:, self.stat_state_scalar_cols].to_numpy(dtype=np.int8)
        for category, class_numbers in self.stat_state_category_cols.items():
            category_state = to_categorical(stat_df.loc[:, category], num_classes=class_numbers).astype(np.int8)
            stat_state = np.concatenate((stat_state, category_state), axis=1)
        return y, x, stat_state

    def save_regression_numpy(self, **kwargs):
        lab, dyn, stat = self.create_regression_numpy(**kwargs)
        print('INFO - Speichere NPZ-Dateien')
        filename = '-'.join([self.type, kwargs['StartDatum'], kwargs['EndDatum'], str(kwargs['StepSize'])])
        path = os.path.join(self.output_path, filename)
        np.savez(path, lab=lab, dyn=dyn, stat=stat)


data_dir = os.path.join('files', 'raw')
output_dir = os.path.join('files', 'prepared')
warengruppen_maske = [1, 12, 55, 80, 17, 77, 71, 6, 28]
dyn_state_scalar_cols = ['Menge', 'MaxTemp_1D', 'MinTemp_1D', 'Wolken_1D',
                         'Regen_1D', 'MaxTemp_2D', 'MinTemp_2D', 'Wolken_2D', 'Regen_2D',
                         'Preis', 'relRabatt', 'absRabatt']
dyn_state_label_cols = ['in1', 'in2', 'in3', 'in4', 'in5']
dyn_state_category_cols = {'Wochentag': 7, 'Kalenderwoche': 54}
stat_state_scalar_cols = ['Eigenmarke', 'GuG', 'OSE', 'Saisonal', 'Kern', 'Bio', 'Glutenfrei',
                          'Laktosefrei']
stat_state_category_cols = {'MHDgroup': 7, 'Warengruppe': 9, 'Detailwarengruppe': None, 'Einheit': None}

# region Markt.Train
markt = Datapipeline(
    InputPath=data_dir,
    OutputPath=output_dir,
    ZielWarengruppen=warengruppen_maske,
    Type='Markt',
    DynStateScalarCols=dyn_state_scalar_cols,
    DynStateLabelCols=dyn_state_label_cols,
    DynStateCategoryCols=dyn_state_category_cols,
    StatStateScalarCols=stat_state_scalar_cols,
    StatStateCategoryCols=stat_state_category_cols
)
markt.read_files()
markt.save_regression_numpy(
    StartDatum='2017-01-01',
    EndDatum='2017-12-31',
    StepSize=6
)
# endregion

# region Markt.Test
markt = Datapipeline(
    InputPath=data_dir,
    OutputPath=output_dir,
    ZielWarengruppen=warengruppen_maske,
    Type='Markt',
    DynStateScalarCols=dyn_state_scalar_cols,
    DynStateLabelCols=dyn_state_label_cols,
    DynStateCategoryCols=dyn_state_category_cols,
    StatStateScalarCols=stat_state_scalar_cols,
    StatStateCategoryCols=stat_state_category_cols
)
markt.read_files()
markt.save_regression_numpy(
    StartDatum='2018-01-01',
    EndDatum='2018-12-31',
    StepSize=6
)
# endregion

# region Time.Train
time = Datapipeline(
    InputPath=data_dir,
    OutputPath=output_dir,
    ZielWarengruppen=warengruppen_maske,
    Type='Time',
    DynStateScalarCols=dyn_state_scalar_cols,
    DynStateLabelCols=dyn_state_label_cols,
    DynStateCategoryCols=dyn_state_category_cols,
    StatStateScalarCols=stat_state_scalar_cols,
    StatStateCategoryCols=stat_state_category_cols
)
time.read_files()
time.save_regression_numpy(
    StartDatum='2016-01-01',
    EndDatum='2017-12-31',
    StepSize=6
)
# endregion

# region Time.Test
time = Datapipeline(
    InputPath=data_dir,
    OutputPath=output_dir,
    ZielWarengruppen=warengruppen_maske,
    Type='Time',
    DynStateScalarCols=dyn_state_scalar_cols,
    DynStateLabelCols=dyn_state_label_cols,
    DynStateCategoryCols=dyn_state_category_cols,
    StatStateScalarCols=stat_state_scalar_cols,
    StatStateCategoryCols=stat_state_category_cols
)
time.read_files()
time.save_regression_numpy(
    StartDatum='2018-01-01',
    EndDatum='2018-12-31',
    StepSize=3
)
# endregion
# TODO: Baseline in Klasse einbauen
# bewegung = self.dynamic_state.loc[:, ['Markt', 'Artikel', 'Datum', 'Menge', 'UNIXDatum']].copy()
# bewegung.reset_index(inplace=True, drop=True)
# bewegung['Prediction'] = bewegung.groupby(['Markt', 'Artikel'])['Menge'].shift(1)
# bewegung['AError'] = np.abs(bewegung['Menge'] - bewegung['Prediction'])
# bewegung['SError'] = np.square(bewegung['AError'])
# bewegung.dropna(inplace=True)
# bewegung['MAE'] = bewegung['AError'].rolling(prediction_days).mean()
# bewegung['MSE'] = bewegung['SError'].rolling(prediction_days).mean()
# self.mae = np.mean(bewegung['MAE'])
# self.mse = np.mean(bewegung['MSE'])
# print('BASELINE\n---\nMean Average Error: {mae} \nMean Squared Error: {mse}'.format(
#     mae=self.mae,
#     mse=self.mse
# ))
