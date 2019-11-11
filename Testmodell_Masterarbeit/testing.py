"""
Datei für Testzwecke während der Entwicklungsphase
"""
import numpy as np
import matplotlib.pyplot as plt
from data.access import DataPipeLine
import random


simulation_params = {
    'ZielWarengruppen': [55],
    'DetailWarengruppe': [2363]
}
markt_index = {
    27: 0,
    67: 1,
    87: 2,
    128: 3,
    129: 4,
    147: 5
}
inv_markt_index = {k: v for v, k in markt_index.items()}

pipeline = DataPipeLine(**simulation_params)
absatz, bewegung, artikelstamm = pipeline.get_statistics_data()
aggr = absatz.groupby(['Artikel', 'Markt']).sum()['Menge']*8
kleine = aggr[aggr < 40]
kleine_artikel = kleine.index.values
grosse = aggr[aggr > 40]
grosse_artikel = grosse.index.values

test_artikel = random.choice(kleine_artikel)
artikel_bewegung = bewegung[(bewegung.Artikel == test_artikel[0]+1) & (bewegung.Markt == inv_markt_index[test_artikel[1]])]
pivot = artikel_bewegung.pivot_table(index='Datum', values='Menge', columns=['Belegtyp'], aggfunc=np.sum)
pivot.plot(subplots=True, style=['.', '.', '.', '.', '.', '.', '.'], title=artikelstamm.loc[test_artikel[0]].Bezeichnung)
plt.show()

# 180471
# 180472