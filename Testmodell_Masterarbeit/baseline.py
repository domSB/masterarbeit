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

Wareneingang['Datum'] =  pd.to_datetime(Wareneingang['Datum'], format='%Y-%m-%d')
Warenausgang['Datum'] =  pd.to_datetime(Warenausgang['Datum'], format='%Y-%m-%d')

#Warenausgang = Warenausgang[Warenausgang.Datum.dt.year == 2018]
#Wareneingang = Wareneingang[Wareneingang.Datum.dt.year == 2018]

Warenausgang = Warenausgang.set_index(Warenausgang.Datum)
Wareneingang = Wareneingang.set_index(Wareneingang.Datum)


date = datetime.datetime(2018,1,1)
initialbestand = 10
bestand = initialbestand
bestandslinie = []
zeitlinie = []
i = 0
while True:
    if date == datetime.datetime(2019, 6, 2):
        break
    print(i)
    bestandslinie.append(bestand)
    zeitlinie.append(date)
    try:
        we = Wareneingang.loc[date]
    except KeyError:
        we = None
    try:
        wa = Warenausgang.loc[date]
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


plt.plot(zeitlinie, bestandslinie)
plt.show()
plt.plot(zeitlinie, [reward(x) for x in bestandslinie])
plt.show()