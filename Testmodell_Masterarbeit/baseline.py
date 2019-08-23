import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from calender import get_german_holiday_calendar

DATA_PATH = os.path.join('F:', 'OneDrive', 'Dokumente', '1 Universität - Master', '6. Semester', 'Masterarbeit', 'Implementation', 'Echtdaten')

""" 
Zum Testen nur mit einem Artikel
"""

def reward(x):
    if x >= 27.5:
        reward = 0.004992 - (x-27.5)/1000
    elif x >= 1:
        reward = np.exp((1-x)/5)
    else:
        reward = np.exp((x-1)*1.5)-1
        
    return reward

Wareneingang = pd.read_csv(
    os.path.join(DATA_PATH, '0 Wareneingang.time.csv'),
    header=0,
    names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
    )

Warenausgang = pd.read_csv(
    os.path.join(DATA_PATH, '0 Warenausgang.time.csv'),
    header=0,
    names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
    )

Warenbestand = pd.read_csv(
    os.path.join(DATA_PATH, '0 Warenbestand.time.csv'),
    header=0,
    names=['Markt', 'Artikel', 'Bestand', 'MengeEK', 'MengeVK', 'Anfangsbestand', 'Datum']
    )
Wareneingang['Datum'] =  pd.to_datetime(Wareneingang['Datum'], format='%d.%m.%y')
Warenausgang['Datum'] =  pd.to_datetime(Warenausgang['Datum'], format='%d.%m.%y')

Warenausgang = Warenausgang.set_index('Artikel')
Wareneingang = Wareneingang.set_index('Artikel')
Warenbestand = Warenbestand.set_index('Artikel')

alle_artikel = pd.unique(Warenausgang.index.get_values())

cal_cls = get_german_holiday_calendar('SL')
cal = cal_cls()
sl_bd = pd.tseries.offsets.CustomBusinessDay(calendar=cal, weekmask='Mon Tue Wed Tue Fri Sat')
zeitraum = pd.date_range('2016-01-01', '2019-06-30', freq=sl_bd)

def get_reward(artikel_nr, plot=False):
    global Warenausgang, Wareneingang, Warenbestand, zeitraum
    ausgang = Warenausgang.loc[artikel_nr].copy()
    eingang = Wareneingang.loc[artikel_nr].copy()
    bestand = Warenbestand.loc[artikel_nr].copy()

    eingang.Menge = -eingang.Menge

    bewegung = pd.concat([eingang, ausgang]).drop(columns=['Markt'])

    bewegung_agg = bewegung.groupby('Datum').sum()
    bewegung_agg = bewegung_agg.reindex(zeitraum, fill_value=0)
    bewegung_agg.sort_index(ascending=False, inplace=True)
    bewegung_agg.iloc[0] += bestand.Bestand
    bewegung_agg['Bestand'] = bewegung_agg['Menge'].cumsum()
    bewegung_agg['Reward'] = bewegung_agg['Bestand'].apply(lambda x: reward(x))
    ges_reward = np.sum(bewegung_agg['Reward'])
    if plot:
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        f.suptitle('Analyse Artikel: {}'.format(artikel_nr), fontsize=14)
        bewegung_agg.plot(y='Bestand', title='Bestandsentwicklung', use_index=True, ax=ax1)
        bewegung_agg.plot(y='Reward', title='Belohnung {:0.2f}'.format(ges_reward), use_index=True, ax=ax2)
        plt.show()
    return ges_reward, bewegung_agg

for artikel in np.random.choice(alle_artikel, 4):
    ges_reward, bewegung = get_reward(artikel, True)