import numpy as np
import csv
import pandas as pd

def prepare(in_path, out_path, absatzart):
    df = pd.read_csv(in_path)
    df['DATUM_BELEG'] = pd.to_datetime(df['DATUM_BELEG'])
    df.dropna(how='any', inplace=True)
    df = df.drop(labels="Unnamed: 0", axis=1) 
    df['ARTIKELNUMMER'] = df['ARTIKELNUMMER'].astype(np.int32)
    df[absatzart] = df[absatzart].astype(np.float32)
    df['WARENGRUPPENNUMMER'] = df['WARENGRUPPENNUMMER'].astype(np.uint8)
    df['ABTEILUNGSNUMMER'] = df['ABTEILUNGSNUMMER'].astype(np.uint8)
    if absatzart == 'Tagesabsatz':
        new_name = 'Absatz'
    elif absatzart == 'Tagespreisabschriften':
        new_name = 'Preisabschrift'
    else:
        new_name = 'Vollabschrift'
    df = df.rename(index=str, columns={"DATUM_BELEG": "Datum", "ARTIKELNUMMER": "Artikel", absatzart:new_name, "WARENGRUPPENNUMMER": "Warengruppe", "ABTEILUNGSNUMMER": "Abteilung"})
    df.to_pickle(out_path)
    return



for file, absatzart in [("absatz_altforweiler", "Tagesabsatz"), ("preisabschrift_altforweiler", "Tagespreisabschriften"), ("vollabschrift_altforweiler", "Tagesvollabschriften")]:
    in_path = 'F:/OneDrive/Dokumente/1 Universität - Master/6. Semester/Masterarbeit/Implemenation/Echtdaten/3 ' + file + '.csv'
    out_path = 'F:/OneDrive/Dokumente/1 Universität - Master/6. Semester/Masterarbeit/Implemenation/Echtdaten/4 ' + file + '.pkl'
    prepare(in_path, out_path, absatzart)