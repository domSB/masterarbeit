"""
Hier sollen die Inputdaten auf den Absatz regressiert werden, um die Vorhersagekraft zu bestimmen.
"""
#TODO: Regression von State auf Absatz
import os

import numpy as np
import pandas as pd
from keras.utils import to_categorical

from tqdm import tqdm


from calender import get_german_holiday_calendar

DATA_PATH = os.path.join('F:', 'OneDrive', 'Dokumente', '1 Universität - Master', '6. Semester', 'Masterarbeit', 'Implementation', 'Echtdaten')

Artikelstamm = pd.read_csv(
    os.path.join(DATA_PATH, '0 ArtikelstammV4.csv'),
    header=0,
    names=['Artikel', 'Warengruppe', 'Detailwarengruppe', 'Bezeichnung',
       'Eigenmarke', 'Einheit', 'Verkaufseinheit', 'MHD',
       'GuG', 'OSE', 'OSEText', 'Saisonal',
       'Kern', 'Bio', 'Glutenfrei',
       'Laktosefrei', 'MarkeFK', 'Region'],
    memory_map=True
    )
warengruppen_maske = [1, 12, 55, 80, 17, 77, 71, 6, 28 ]
Artikelstamm = Artikelstamm[Artikelstamm.Warengruppe.isin(warengruppen_maske)]
artikel_maske = pd.unique(Artikelstamm.Artikel)
Artikelstamm = Artikelstamm.set_index('Artikel')

warenausgang = pd.read_csv(
    os.path.join(DATA_PATH, '0 Warenausgang.time.csv'),
    header=0,
    names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
    )

warenausgang = warenausgang[warenausgang.Artikel.isin(artikel_maske)]

warenausgang['Datum'] = pd.to_datetime(warenausgang['Datum'], format='%d.%m.%y')
warenausgang['Belegtyp'] = warenausgang['Belegtyp'].astype('category')
warenausgang.drop(columns=['Markt'], inplace=True)
warenausgang = warenausgang.loc[(warenausgang.Belegtyp=='UMSATZ_SCANNING') | (warenausgang.Belegtyp=='UMSATZ_AKTION')].copy()
warenausgang = warenausgang.groupby(["Datum", "Artikel"],  as_index=False).sum()

cal_cls = get_german_holiday_calendar('SL')
cal = cal_cls()
sl_bd = pd.tseries.offsets.CustomBusinessDay(calendar=cal, weekmask='Mon Tue Wed Tue Fri Sat')
zeitraum = pd.date_range('2018-01-01', '2018-12-31', freq=sl_bd)

warenausgang.set_index('Datum', inplace=True)
warenausgang = warenausgang.groupby('Artikel').apply(lambda x: x.reindex(zeitraum, fill_value=0))
warenausgang.drop(columns=['Artikel'], inplace=True)
warenausgang.reset_index(inplace=True)
warenausgang.rename(columns={'level_1': 'Datum'}, inplace=True)

warenausgang.set_index('Artikel', inplace=True)
"""

Zwischenstand

"""

warenausgang['Wochentag'] = warenausgang.Datum.dt.dayofweek
# dataframe["Wochentag"] = dataframe["Wochentag"].apply(lambda x: to_categorical(x, num_classes = 7))
warenausgang['Kalenderwoche'] = warenausgang.Datum.dt.weekofyear
warenausgang["UNIXDatum"] = warenausgang["Datum"].astype(np.int64)/(1000000000 * 24 * 3600)

mhd_labels = [0, 1, 2, 3, 4, 5, 6]
mhd_bins = [0, 1, 7, 14, 28, 100, 1000, 100000]
artikelstamm['MHDgroup'] = pd.cut(artikelstamm.MHD, mhd_bins, right=False, labels=mhd_labels)
dataframe = pd.merge(dataframe, artikelstamm.loc[:,["Warengruppe", "Einheit", "Eigenmarke", "GuG", "MHDgroup"]], how='left', left_on=["Artikel"], right_index=True)

wetter = pd.read_csv(
    'F:/OneDrive/Dokumente/1 Universität - Master/6. Semester/Masterarbeit/Implementation/Echtdaten/1 Wetter.csv',
    memory_map=True
    )
wetter["date_shifted_oneday"] = wetter["date"] + 1
wetter["date_shifted_twodays"] = wetter["date"] + 2
wetter = wetter.drop(columns=['Unnamed: 0', 'NebenGruppe', 'HauptGruppe'])
dataframe = pd.merge(dataframe, wetter, left_on='UNIXDatum', right_on='date_shifted_oneday')
dataframe = pd.merge(dataframe, wetter, left_on='UNIXDatum', right_on='date_shifted_twodays')
dataframe.drop(columns=["date_shifted_oneday_x", "date_shifted_twodays_x", "date_shifted_oneday_y", "date_shifted_twodays_y"], inplace=True)

preise = pd.read_csv(
    'F:/OneDrive/Dokumente/1 Universität - Master/6. Semester/Masterarbeit/Implementation/Echtdaten/0 RegulärePreise.csv',
    memory_map=True
    )
preise = preise.drop(columns=['Unnamed: 0'])
preise['Datum'] = pd.to_datetime(preise['Datum'], format='%Y-%m-%d')
preise.drop_duplicates(inplace=True)
preise = preise.sort_values(by=['Datum', 'Artikel'])
preise['Next'] = preise.groupby(['Artikel'])['Datum'].shift(-1)
preise = preise.where(~preise.isna(), pd.to_datetime('2019-07-01'))

dataframe = pd.merge_asof(dataframe, preise.loc[:,["Preis", "Artikel", "Datum"]], left_on='Datum', right_on='Datum', by='Artikel')

aktionspreise = pd.read_csv(
    'F:/OneDrive/Dokumente/1 Universität - Master/6. Semester/Masterarbeit/Implementation/Echtdaten/0 Preisaktionen.csv',
    decimal =',',
    memory_map=True
    )
aktionspreise['DatumAb'] = pd.to_datetime(aktionspreise['DatumAb'], format='%d.%m.%Y')
aktionspreise['DatumBis'] = pd.to_datetime(aktionspreise['DatumBis'], format='%d.%m.%Y')
aktionspreise = aktionspreise.sort_values(by=['DatumAb', 'Artikel'])
aktionspreise.drop_duplicates(inplace=True)

aktionspreise = pd.merge_asof(aktionspreise, preise, left_on='DatumAb', right_on='Datum', by='Artikel')
aktionspreise = aktionspreise.dropna() # Falls für Aktionszeitraum kein regulärer Preis vorhanden => regulärer Preis muss vor Aktionsstart liegen
aktionspreise = aktionspreise.drop(columns=['Datum', 'Next'])
aktionspreise['relAbweichung'] = (aktionspreise.Preis - aktionspreise.AktionsPreis)/aktionspreise.Preis

artikel = pd.unique(dataframe.Artikel)
aktionspreise = aktionspreise[aktionspreise.Artikel.isin(artikel)]
start = min(dataframe.Datum)

aktionspreise = aktionspreise[aktionspreise.DatumBis >= start]

# ACHTUNG: Extrem langsam, aber macht Matching einfacher
#
# 5 min Laufzeit bei 90.000 Aktionen
test = aktionspreise.copy()
test["GroupID"] = test.groupby("Artikel")["DatumBis"].rank()
test = pd.melt(test, id_vars=['Artikel', 'relAbweichung', 'GroupID'], value_vars=['DatumAb', 'DatumBis'])
test = test.set_index(['value'])
drop = lambda df: df.index.drop_duplicates()
res = lambda df: df[~df.index.duplicated()].resample(rule='1D').ffill()
# test = test[(test.Artikel==5774) | (test.Artikel==5711)]
test = test.groupby(['Artikel', 'GroupID'], as_index=False).apply(res)
test = test.reset_index().drop(["level_0"], axis=1)
test["Aktion"] = 1
dataframe = pd.merge(dataframe, test.loc[:,["value","Artikel", "Aktion", "relAbweichung"]], left_on=['Aktion', 'Datum', 'Artikel'], right_on=['Aktion', 'value', 'Artikel'], how='left')
dataframe.relAbweichung.fillna(0, inplace=True)
dataframe.drop(columns=["value"], inplace=True)
store = pd.HDFStore('./data/time.h5')

store.put('dataframe', dataframe)
store.close()

""" 
Hier mit fertigen Daten weiterarbeiten 
"""
store = pd.HDFStore('./data/time.h5')
dataframe = store.get('dataframe')
store.close()
dataframe = dataframe.dropna() # Artikel ohne Preise => nur aktuell ein Problem


