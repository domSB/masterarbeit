import os
import datetime
from multiprocessing import Process, Pool, Queue, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def convert_series_to_frame(series):
    frame = pd.DataFrame(
        [
            {
            'Markt': series.Markt,
            'Belegtyp': series.Belegtyp,
            'Menge': series.Menge,
            'Datum': series.Datum
                }
            ],
        index=[series.name]
        )
    return frame

def get_reward(artikel_nr, plot=False):
    global Warenausgang, Wareneingang, Warenbestand, zeitraum
    ausgang = Warenausgang.loc[artikel_nr].copy()
    if type(ausgang) == pd.core.series.Series:
        ausgang = convert_series_to_frame(ausgang)

    try:
        eingang = Wareneingang.loc[artikel_nr].copy()
        eingang.Menge = -eingang.Menge
    except KeyError: 
        """ 
        Entsteht durch nicht geschlossenes Warensystem. 
        Wareneingänge ggf. nie elektronisch erfasst.
        """
        eingang = None
    if type(eingang) == pd.core.series.Series:
        eingang = convert_series_to_frame(eingang)

    bestand = Warenbestand.loc[artikel_nr].copy()   

    bewegung = pd.concat([eingang, ausgang], sort=True).drop(columns=['Markt'])

    bewegung_agg = bewegung.groupby('Datum').sum()
    bewegung_agg = bewegung_agg.reindex(zeitraum, fill_value=0)
    aktive_tage = bewegung_agg[bewegung_agg.Menge != 0].shape[0]
    aktivitaet = aktive_tage / zeitraum.shape[0]
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
    return ges_reward, aktivitaet, bewegung_agg

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

Wareneingang = pd.read_csv(
    os.path.join(DATA_PATH, '0 Wareneingang.time.csv'),
    header=0,
    names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
    )
Wareneingang = Wareneingang[Wareneingang.Artikel.isin(artikel_maske)]

Warenausgang = pd.read_csv(
    os.path.join(DATA_PATH, '0 Warenausgang.time.csv'),
    header=0,
    names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
    )
Warenausgang = Warenausgang[Warenausgang.Artikel.isin(artikel_maske)]

Warenbestand = pd.read_csv(
    os.path.join(DATA_PATH, '0 Warenbestand.time.csv'),
    header=0,
    names=['Markt', 'Artikel', 'Bestand', 'MengeEK', 'MengeVK', 'Anfangsbestand', 'Datum']
    )
Warenbestand = Warenbestand[Warenbestand.Artikel.isin(artikel_maske)]

Wareneingang['Datum'] =  pd.to_datetime(Wareneingang['Datum'], format='%d.%m.%y')
Warenausgang['Datum'] =  pd.to_datetime(Warenausgang['Datum'], format='%d.%m.%y')

Warenausgang = Warenausgang.set_index('Artikel')
Wareneingang = Wareneingang.set_index('Artikel')
Warenbestand = Warenbestand.set_index('Artikel')

alle_artikel = pd.unique(Warenausgang.index.get_values())

cal_cls = get_german_holiday_calendar('SL')
cal = cal_cls()
sl_bd = pd.tseries.offsets.CustomBusinessDay(calendar=cal, weekmask='Mon Tue Wed Tue Fri Sat')
zeitraum = pd.date_range('2018-01-01', '2018-12-31', freq=sl_bd)

rewards = []
for artikel in tqdm(alle_artikel):
    ges_reward, aktivitaet, bewegung = get_reward(artikel, False)
    rewards.append([artikel, ges_reward, aktivitaet])

rewards = np.array(rewards)
rewards = pd.DataFrame(rewards[:,1:3], index=rewards[:,0], columns=['Reward', 'Aktivitaet'])
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('Rewards der Mitarbeiter in 2018', fontsize=14)
ax1.set_title('Verteilung der Rewards', fontsize=11)
ax1.hist(rewards.Reward, bins=range(-300, 300, 10))
ax2.set_title('Reward zu prozentualer Aktivität der Artikel', fontsize=11)
ax2.scatter(rewards[rewards.Reward>-300].Reward, rewards[rewards.Reward>-300].Aktivitaet)
plt.show()
