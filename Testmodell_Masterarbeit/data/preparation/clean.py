import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


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

    def read_files(self):
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

    def prepare_for_simulation(self):
        pass

    def prepare_for_regression(self):
        # TODO: Datapipeline aus regression.py kopieren
        pass


data_dir = os.path.join('files', 'raw')
output_dir = os.path.join('files', 'prepared')
warengruppen_maske = [1, 12, 55, 80, 17, 77, 71, 6, 28]
markt = Datapipeline(
    InputPath=data_dir,
    OutputPath=output_dir,
    ZielWarengruppen=warengruppen_maske,
    Type='Markt'
)
markt.read_files()

time = Datapipeline(
    InputPath=data_dir,
    OutputPath=output_dir,
    ZielWarengruppen=warengruppen_maske,
    Type='Time'
)
time.read_files()
