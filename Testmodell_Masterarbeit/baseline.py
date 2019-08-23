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
Wareneingang.rename(columns={'MARKT': 'Markt', 'oldName2': 'newName2'}, inplace=True)
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

ausgang = Warenausgang.loc[22299].copy()
eingang = Wareneingang.loc[22299].copy()
bestand = Warenbestand.loc[22299].copy()

eingang.Menge = -eingang.Menge

bewegung = pd.concat([eingang, ausgang]).drop(columns=['Markt'])

bewegung_agg = bewegung.groupby('Datum').sum()

cal_cls = get_german_holiday_calendar('SL')
cal = cal_cls()

sl_bd = pd.tseries.offsets.CustomBusinessDay(calendar=cal, weekmask='Mon Tue Wed Tue Fri Sat')

zeitraum = pd.date_range('2016-01-01', '2019-06-30', freq=sl_bd)

bewegung_agg = bewegung_agg.reindex(zeitraum, fill_value=0)

bewegung_agg.sort_index(ascending=False, inplace=True)

bewegung_agg.iloc[0] += bestand.Bestand

bewegung_agg['Test'] = bewegung_agg['Menge'].cumsum()
bewegung_agg['Reward'] = bewegung_agg['Test'].apply(lambda x: reward(x))

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
bewegung_agg.plot(y='Test', title='Bestandsentwicklung', use_index=True, ax=ax1)
bewegung_agg.plot(y='Reward', title='Belohnung {:0.2f}'.format(np.sum(bewegung_agg['Reward'])), use_index=True, ax=ax2)
plt.show()










alle_artikel = Warenbestand.Artikel.to_numpy()

for artikel in alle_artikel:
    date = datetime.datetime(2018,1,1)

    artikel_wareneingang = Wareneingang[Wareneingang.Artikel == artikel]
    artikel_warenausgang = Warenausgang[Warenausgang.Artikel == artikel]
    artikel_warenbestand = Warenbestand[Warenbestand.Artikel == artikel]

    we_gesamt = sum(artikel_wareneingang.loc[date:artikel_warenbestand.Datum.iat[0]].Menge)
    wa_gesamt = sum(artikel_warenausgang.loc[date:artikel_warenbestand.Datum.iat[0]].Menge)

    initialbestand = artikel_warenbestand.Anfangsbestand.iat[0] - we_gesamt + wa_gesamt
    bestand = initialbestand
    bestandslinie = []
    zeitlinie = []
    i = 0
    while True:
        if date == datetime.datetime(2019, 6, 2):
            break
        bestandslinie.append(bestand)
        zeitlinie.append(date)
        try:
            we = artikel_wareneingang.loc[date]
        except KeyError:
            we = None
        try:
            wa = artikel_warenausgang.loc[date]
        except KeyError:
            wa = None
        if wa is not None:
            if wa.index.has_duplicates:
                for idx, wa_action in wa.iterrows():
                    bestand -= wa_action.Menge
            else:    
                bestand -= wa.Menge

        if we is not None:
            if we.index.has_duplicates:
                for idx, we_action in we.iterrows():
                    bestand += we_action.Menge
            else:    
                bestand += we.Menge

        

        date += datetime.timedelta(days=1)
        i += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(zeitlinie, bestandslinie)
    ax1.set_title('Bestandsentwicklung')
    ax2.plot(zeitlinie, [reward(x) for x in bestandslinie])
    ax2.set_title('Belohnungen')
    fig.suptitle('Artikel: ' + str(artikel_warenbestand.Artikel.iat[0]), fontsize='large')
    plt.show()