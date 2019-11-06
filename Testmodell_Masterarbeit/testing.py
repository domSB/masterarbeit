"""
Datei für Testzwecke während der Entwicklungsphase
"""
import numpy as np
import matplotlib.pyplot as plt


bestand = np.array([
    3,
    3,
    4,
    14,
    14,
    14,
    14,
    14
])
absatz = np.array([
    8,
])
kum_absatz = absatz.cumsum()
bestand_max_reichweite = bestand[-1]
tage, mengen = np.unique(bestand, return_counts=True)
cum_bestand = np.zeros(kum_absatz.shape, dtype=np.int64)  # brauche den Bestand für die zu betrachtende Absatzperiode
for tag, menge in zip(tage, mengen):
    cum_bestand[:tag] += menge
    print(cum_bestand)
print('diff', (cum_bestand - kum_absatz))
print(np.argwhere((cum_bestand - kum_absatz) < 0))
break_even = np.argwhere((cum_bestand - kum_absatz) < 0)
if break_even.size > 0:
    bestandsreichweite = break_even[0, 0]
else:
    bestandsreichweite = len(absatz) - 1
abschriften = cum_bestand[0] - cum_bestand[bestandsreichweite-1]
absatz_ausfall = absatz[bestandsreichweite + 1:].sum()
# Sonderanalyse Tag des Umbruchs
tages_absatz = absatz[bestandsreichweite]
vor_tages_bestand = cum_bestand[bestandsreichweite - 1] - kum_absatz[bestandsreichweite - 1]
tages_abschrift = cum_bestand[bestandsreichweite - 1] - cum_bestand[bestandsreichweite]

# Annahme, das Abschrift zuerst eintritt
restmenge_nach_abschrift = vor_tages_bestand - tages_abschrift
if restmenge_nach_abschrift > 0:  # Nach Abschrift noch Artikel im Bestand
    abschriften += tages_abschrift
    absatz_ausfall += max(tages_absatz - restmenge_nach_abschrift, 0)
else:
    abschriften += vor_tages_bestand  # Wenn Restmenge negativ, kann nur der Bestand abgeschrieben werden
    absatz_ausfall += tages_absatz  # Absatz fällt komplett aus

plt.plot(cum_bestand, label='Bestand')
plt.plot(kum_absatz, label='Absatz')
plt.legend()
plt.show()