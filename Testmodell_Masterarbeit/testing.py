"""
Datei für Testzwecke während der Entwicklungsphase
"""
import numpy as np
import matplotlib.pyplot as plt


def belohnung_bestandsreichweite(_bestand, _absatz, order_zyklus, rohertrag=0.3, ek_preis=0.7, kap_kosten=0.05/365):
    """
    Berechnet die Belohnung für den kommenden Bestand
    """
    assert len(_absatz) > 1
    analyse_tage = len(_absatz)
    anfangsbestand = _bestand.shape[0]
    kum_absatz = _absatz.cumsum()
    tage, mengen = np.unique(_bestand, return_counts=True)
    abschr_bestand = np.zeros(analyse_tage, dtype=np.int64)
    for tag, menge in zip(tage, mengen):
        abschr_bestand[:tag] += menge

    absatz_bestand = anfangsbestand - kum_absatz
    abschriften = np.clip(abschr_bestand - absatz_bestand, None, 0)

    real_bestand = absatz_bestand + abschriften

    break_even = np.argwhere(real_bestand < 0)
    if break_even.size > 0:
        bestandsreichweite = break_even[0, 0]
    else:
        bestandsreichweite = analyse_tage - 1
    anz_abschriften = -abschriften[:bestandsreichweite].sum()
    absatz_ausfall = _absatz[bestandsreichweite + 1:].sum()
    # Sonderanalyse Tag des Umbruchs
    tages_absatz = _absatz[bestandsreichweite]
    vor_tages_bestand = real_bestand[bestandsreichweite - 1]
    tages_abschrift = abschr_bestand[bestandsreichweite - 1] - abschr_bestand[bestandsreichweite]

    # Annahme, das Abschrift zuerst eintritt
    restmenge_nach_abschrift = vor_tages_bestand - tages_abschrift
    if restmenge_nach_abschrift > 0:  # Nach Abschrift noch Artikel im Bestand
        anz_abschriften += tages_abschrift
        absatz_ausfall += max(tages_absatz - restmenge_nach_abschrift, 0)
    else:
        anz_abschriften += vor_tages_bestand  # Wenn Restmenge negativ, kann nur der Bestand abgeschrieben werden
        absatz_ausfall += tages_absatz  # Absatz fällt komplett aus

    rew_abschrift = anz_abschriften * - ek_preis
    if bestandsreichweite > order_zyklus - 1:
        rew_bestand = - kap_kosten * real_bestand[order_zyklus]
        rew_fehlmenge = 0
    else:
        rew_bestand = 0
        fehlmenge = absatz_ausfall - _absatz[order_zyklus:].sum()
        rew_fehlmenge = fehlmenge * - rohertrag

    return np.array([rew_abschrift, rew_bestand, rew_fehlmenge])


bestand = np.array([
    2,
    2,
    2,
    14,
    14,
    14,
    14,
    14
])
absatz = np.array([
    1,
    1,
    0,
    3,
    2,
    1,
])
reward = belohnung_bestandsreichweite(bestand, absatz, 3)

# kum_absatz = absatz.cumsum()
# tage, mengen = np.unique(bestand, return_counts=True)
# abschr_bestand = np.zeros(kum_absatz.shape,
#                           dtype=np.int64)  # brauche den Bestand für die zu betrachtende Absatzperiode
# for tag, menge in zip(tage, mengen):
#     abschr_bestand[:tag] += menge
#
# absatz_bestand = 8 - kum_absatz
# abschriften = np.clip(abschr_bestand - absatz_bestand, None, 0)
#
# real_bestand = absatz_bestand + abschriften
#
# plt.bar(np.arange(0, 6) - 0.375, abschr_bestand, width=1/4, fill=True, label='Abschriftsbestand', color='red', alpha=0.3)
# plt.bar(np.arange(0, 6) - 0.125, absatz_bestand, width=1/4, fill=True, label='Absatzbestand', color='blue', alpha=0.3)
# plt.bar(np.arange(0, 6) + 0.125, abschriften, width=1/4, fill=True, label='Wahre Abschriften', color='green', alpha=0.3)
# plt.bar(np.arange(0, 6) + 0.375, real_bestand, width=1/4, fill=True, label='Realbestand', color='orange', alpha=0.3)
#
# plt.legend()
# plt.show()
