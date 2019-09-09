import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from calender.german_holidays import get_german_holiday_calendar


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
            names=['Artikel', 'DatumAb', 'DatumBis', 'Preis']
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
        # TODO: Datapipeline aus regression.py kopieren
        # Fehlende Detailwarengruppen mit wahrscheinlich richtiger DetailWarengruppe füllen
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

        # Restlaufzeit von Anzahl-Tage in kategoriale Gruppen sortieren
        mhd_labels = [0, 1, 2, 3, 4, 5, 6]
        mhd_bins = [0, 1, 7, 14, 28, 100, 1000, 100000]
        self.artikelstamm['MHDgroup'] = pd.cut(self.artikelstamm.MHD, mhd_bins, right=False, labels=mhd_labels)

        # Lückenhafte Fremdschlüssel durch eine durchgehende ID ersetzen
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
        with open(os.path.join(self.output_path, 'ValueMapping.json'), 'w') as file:
            json.dump(mapping, file)
        self.artikelstamm['Detailwarengruppe'] = self.artikelstamm['Detailwarengruppe'].map(
            detail_warengruppen_index)
        self.artikelstamm['Warengruppe'] = self.artikelstamm['Warengruppe'].map(warengruppen_index)
        self.artikelstamm['Einheit'] = self.artikelstamm['Einheit'].map(einheit_index)
        self.artikelstamm.drop(columns=['MHD', 'Region', 'MarkeFK', 'Verkaufseinheit', 'OSEText'], inplace=True)
        self.artikelstamm['OSE'].fillna(0, inplace=True)
        self.artikelstamm['Saisonal'].fillna(0, inplace=True)

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
        self.preise = self.preise.sort_values(by=['Datum', 'Artikel'])
        # self.preise['Next'] = self.preise.groupby(['Artikel'], as_index=True)['Datum'].shift(-1)
        # self.preise = self.preise.where(
        #     ~self.preise.isna(),
        #     pd.to_datetime(kwargs['EndDatum']) + pd.DateOffset(7)
        # )
        # pd.merge_asof ist ein Left Join mit dem nächsten passenden Key.
        # Standardmäßig wird in der rechten Tabelle der Gleiche oder nächste Kleinere gesucht.
        self.absatz = pd.merge_asof(
            self.absatz,
            self.preise.loc[:, ["Preis", "Artikel", "Datum"]].copy(),
            left_on='Datum',
            right_on='Datum',
            by='Artikel'
        )
        # Fehlerhafte Daten: Häufig exitieren
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
        print('{:.2f} % der Daten verworfen, aufgrund fehlender Preise'.format(np.mean(time.absatz.Preis.isna()) * 100))
        self.absatz.dropna(inplace=True)


data_dir = os.path.join('files', 'raw')
output_dir = os.path.join('files', 'prepared')
warengruppen_maske = [1, 12, 55, 80, 17, 77, 71, 6, 28]
dyn_state_scalar_cols = ['Menge', 'MaxTemp_1D', 'MinTemp_1D', 'Wolken_1D',
                         'Regen_1D', 'MaxTemp_2D', 'MinTemp_2D', 'Wolken_2D', 'Regen_2D',
                         'Preis', 'relRabat', 'absRabat', 'vDauer']
dyn_state_label_cols = ['in1', 'in2', 'in3', 'in4', 'in5']
dyn_state_category_cols = {'Wochentag': 7, 'Kalenderwoche': 54}
stat_state_scalar_cols = ['Eigenmarke', 'GuG', 'OSE', 'Saisonal', 'Kern', 'Bio', 'Glutenfrei',
                          'Laktosefrei']
stat_state_category_cols = {'MHDgroup': 7, 'Warengruppe': 9, 'Detailwarengruppe': None, 'Einheit': None}
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
# markt.read_files()

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
time.prepare_for_regression(
    StartDatum='2016-01-01',
    EndDatum='2018-12-31'
)

