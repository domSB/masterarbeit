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

artikelstamm = pd.read_csv(
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
artikelstamm = artikelstamm[artikelstamm.Warengruppe.isin(warengruppen_maske)]
artikel_maske = pd.unique(artikelstamm.Artikel)
# artikelstamm = artikelstamm.set_index('Artikel')

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

warenausgang['Wochentag'] = warenausgang.Datum.dt.dayofweek
warenausgang['Kalenderwoche'] = warenausgang.Datum.dt.weekofyear
warenausgang["UNIXDatum"] = warenausgang["Datum"].astype(np.int64)/(1000000000 * 24 * 3600)

mhd_labels = [0, 1, 2, 3, 4, 5, 6]
mhd_bins = [0, 1, 7, 14, 28, 100, 1000, 100000]
artikelstamm['MHDgroup'] = pd.cut(artikelstamm.MHD, mhd_bins, right=False, labels=mhd_labels)
warenausgang = pd.merge(warenausgang, artikelstamm.loc[:,["Artikel", "Warengruppe", "Einheit", "Eigenmarke", "GuG", "MHDgroup"]], how='left', on='Artikel', validate='many_to_one')

wetter = pd.read_csv(
    os.path.join(DATA_PATH, '1 Wetter.csv'),
    memory_map=True
    )
wetter["date_shifted_oneday"] = wetter["date"] - 1
wetter["date_shifted_twodays"] = wetter["date"] - 2
wetter.drop(columns=['Unnamed: 0'], inplace=True)
warenausgang = pd.merge(warenausgang, wetter, left_on='UNIXDatum', right_on='date_shifted_oneday')
warenausgang = pd.merge(warenausgang, wetter, left_on='UNIXDatum', right_on='date_shifted_twodays', suffixes=('_1D', '_2D'))
warenausgang.drop(columns=["date_shifted_oneday_1D", "date_shifted_twodays_1D", "date_shifted_oneday_2D", "date_shifted_twodays_2D"], inplace=True)

preise = pd.read_csv(
     os.path.join(DATA_PATH, '1 Preise.csv'),
    memory_map=True
    )
preise.drop(columns=['Unnamed: 0'], inplace=True)
preise['Datum'] = pd.to_datetime(preise['Datum'], format='%Y-%m-%d')
preise.drop_duplicates(inplace=True)
preise = preise.sort_values(by=['Datum', 'Artikel'])
preise['Next'] = preise.groupby(['Artikel'], as_index=True)['Datum'].shift(-1)
preise = preise.where(~preise.isna(), pd.to_datetime('2019-07-01'))
# preise.set_index('Artikel', inplace=True)
warenausgang.reset_index(inplace=True)

warenausgang = pd.merge_asof(warenausgang, preise.loc[:,["Preis", "Artikel", "Datum"]], left_on='Datum', right_on='Datum', by='Artikel')

aktionspreise = pd.read_csv(
    os.path.join(DATA_PATH, '1 Preisaktionen.csv'),
    decimal =',',
    memory_map=True
    )
aktionspreise.drop(columns=['Unnamed: 0'], inplace=True)
aktionspreise['DatumAb'] = pd.to_datetime(aktionspreise['DatumAb'], format='%Y-%m-%d')
aktionspreise['DatumBis'] = pd.to_datetime(aktionspreise['DatumBis'], format='%Y-%m-%d')
aktionspreise.drop_duplicates(inplace=True)
aktionspreise = aktionspreise.sort_values(by=['DatumAb', 'DatumBis', 'Artikel'])


warenausgang = pd.merge_asof(warenausgang, aktionspreise, left_on='Datum', right_on='DatumAb', by='Artikel')
warenausgang.relRabat.fillna(0., inplace=True)
warenausgang.absRabat.fillna(0., inplace=True)
warenausgang.vDauer.fillna(0, inplace=True)
warenausgang.relRabat = warenausgang.relRabat.astype(np.float64)
warenausgang.absRabat = warenausgang.absRabat.astype(np.float64)
warenausgang.vDauer = warenausgang.vDauer.astype(np.int)
warenausgang.drop(columns=['DatumAb', 'DatumBis'], inplace=True)
warenausgang.head(1).T.squeeze()

warenausgang['in1'] = warenausgang.groupby(['Artikel'], as_index=False)['Menge'].shift(-1)
warenausgang.in1.fillna(0., inplace=True)
warenausgang['in2'] = warenausgang.groupby(['Artikel'], as_index=False)['Menge'].shift(-2)
warenausgang.in2.fillna(0., inplace=True)
warenausgang['in3'] = warenausgang.groupby(['Artikel'], as_index=False)['Menge'].shift(-3)
warenausgang.in3.fillna(0., inplace=True)
warenausgang['in4'] = warenausgang.groupby(['Artikel'], as_index=False)['Menge'].shift(-4)
warenausgang.in4.fillna(0., inplace=True)
warenausgang['in5'] = warenausgang.groupby(['Artikel'], as_index=False)['Menge'].shift(-5)
warenausgang.in5.fillna(0., inplace=True)
store = pd.HDFStore('./data/time.h5')

store.put('warenausgang', warenausgang, format='table')
store.close()
"""

Zwischenstand

"""
""" 
Hier mit fertigen Daten weiterarbeiten 
"""
#store = pd.HDFStore('./data/time.h5')
#dataframe = store.get('dataframe')
#store.close()
#dataframe = dataframe.dropna() # Artikel ohne Preise => nur aktuell ein Problem


