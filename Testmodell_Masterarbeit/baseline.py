import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    os.path.join(DATA_PATH, '99 Art_WE.csv')
    )

Warenausgang = pd.read_csv(
    os.path.join(DATA_PATH, '99 Art_WA.csv')
    )

Warenbestand = pd.read_csv(
    os.path.join(DATA_PATH, '99 Art_WB.csv')
    )
Wareneingang['Datum'] =  pd.to_datetime(Wareneingang['Datum'], format='%Y-%m-%d')
Warenausgang['Datum'] =  pd.to_datetime(Warenausgang['Datum'], format='%Y-%m-%d')

#Warenausgang = Warenausgang[Warenausgang.Datum.dt.year == 2018]
#Wareneingang = Wareneingang[Wareneingang.Datum.dt.year == 2018]

Warenausgang = Warenausgang.set_index(Warenausgang.Datum)
Wareneingang = Wareneingang.set_index(Wareneingang.Datum)

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