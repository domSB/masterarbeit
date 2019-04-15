import numpy as np
import csv
import pandas as pd

df = pd.read_csv('F:/OneDrive/Dokumente/1 Universität - Master/6. Semester/Masterarbeit/Implemenation/Echtdaten/altforweiler_vorbereitet.csv')
df['DATUM_BELEG'] = pd.to_datetime(df['DATUM_BELEG'])
df.dropna(how='any', inplace=True)
df = df.drop(labels="Unnamed: 0", axis=1) 
df['ARTIKELNUMMER'] = df['ARTIKELNUMMER'].astype(np.int32)
df['Tagesabsatz'] = df['Tagesabsatz'].astype(np.float32)
df['WARENGRUPPENNUMMER'] = df['WARENGRUPPENNUMMER'].astype(np.uint8)
df['ABTEILUNGSNUMMER'] = df['ABTEILUNGSNUMMER'].astype(np.uint8)
df.rename(index=str, columns={"DATUM_BELEG": "Datum", "ARTIKELNUMMER": "Artikelnummer", "Tagesabsatz": "Absatz", "WARENGRUPPENNUMMER": "Warengruppe", "ABTEILUNGSNUMMER": "Abteilung"})
df.to_pickle('F:/OneDrive/Dokumente/1 Universität - Master/6. Semester/Masterarbeit/Implemenation/Echtdaten/altforweiler.pkl')