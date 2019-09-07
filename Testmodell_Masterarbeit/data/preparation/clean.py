import numpy as np
import pandas as pd
import os
data_dir = os.path.join('files', 'raw')
warengruppenstamm = pd.read_csv(
    os.path.join(data_dir, '0 Warengruppenstamm.csv'),
    header=1,
    names=['WG', 'WGNr', 'WGBez', 'Abt', 'AbtNr', 'AbtBez']
)

artikelstamm = pd.read_csv(
    os.path.join(data_dir, '0 ArtikelstammV4.csv'),
    header=0,
    names=['Artikel', 'Warengruppe', 'Detailwarengruppe', 'Bezeichnung',
           'Eigenmarke', 'Einheit', 'Verkaufseinheit', 'MHD',
           'GuG', 'OSE', 'OSEText', 'Saisonal',
           'Kern', 'Bio', 'Glutenfrei',
           'Laktosefrei', 'MarkeFK', 'Region']
)
warengruppen_maske = [1, 12, 55, 80, 17, 77, 71, 6, 28]
artikelstamm = artikelstamm[artikelstamm.Warengruppe.isin(warengruppen_maske)]

preise = pd.read_csv(
    os.path.join(data_dir, '0 Preise.Markt.csv'),
    header=1,
    names=['Preis', 'Artikel', 'Datum']
)
preise['Datum'] = pd.to_datetime(preise['Datum'], format='%d.%m.%Y')
if 'Markt' not in preise.columns:
    preise['Markt'] = 5

aktionspreise = pd.read_csv(
    os.path.join(data_dir, '0 Aktionspreise.Markt.csv'),
    header=1,
    names=['Artikel', 'DatumAb', 'DatumBis', 'Preis']
)
aktionspreise['DatumAb'] = pd.to_datetime(aktionspreise['DatumAb'], format='%d.%m.%Y')
aktionspreise['DatumBis'] = pd.to_datetime(aktionspreise['DatumBis'], format='%d.%m.%Y')
