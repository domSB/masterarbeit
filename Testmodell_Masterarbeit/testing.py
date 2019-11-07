"""
Datei für Testzwecke während der Entwicklungsphase
"""
import numpy as np
import matplotlib.pyplot as plt


def belohnung_bestandsreichweite(bestand, absatz, order_zyklus, rohertrag=0.3, ek_preis=0.7,
                                 kap_kosten=0.05 / 365):
    """
    Berechnet die Belohnung für den kommenden Bestand
    """
    assert len(absatz) > 1, "Belohnungsfunktion benötigt mehr als einen Absatztag für den Horizont."
    analyse_tage = len(absatz)
    anfangsbestand = bestand.shape[0]
    kum_absatz = absatz.cumsum()
    tage, mengen = np.unique(bestand, return_counts=True)
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
        bestandsreichweite = analyse_tage

    # Fallunterscheidungen:
    # 1. Fall Bestandsreichweite == Orderzyklus
    # ==> Perfekter Bestand
    if bestandsreichweite == order_zyklus:
        reward = 0.1 + 0.3 * kum_absatz[order_zyklus - 1]**2
        # Exponentielle steigende Belohnung für Treffen von hohen Absätzen

    # 2. Fall Bestandsreichweite < Orderzyklus
    # ==> Unterbestand mit Fehlmenge
    elif bestandsreichweite < order_zyklus:
        fehlmenge = real_bestand[order_zyklus - 1]
        reward = -rohertrag * fehlmenge**2

    # 3. Fall Bestandsreichweite > Orderzyklus & Bestand bei Orderzyklus == 0
    # ==> Bestandsreichweite per Definition höher als Orderzyklus, aber Bestellmenge optimal
    else:
        end_bestand = real_bestand[order_zyklus - 1]
        if end_bestand == 0:
            reward = 0.1 + 0.3 * kum_absatz[order_zyklus - 1]**2

    # 4. Fall Bestandsreichweite > Orderzyklus & Bestand bei Orderzyklus > 0
    # ==> Noch Bestand bei nächstem Liefereingang.
        else:
            unvermeidbare_abschriften = -abschriften[order_zyklus-1:].sum()
            # Bei MHD > 2x Orderzyklus werden Abschriften ggf. mehrfach bestraft.
            verkaufbarer_mehrbestand = end_bestand - unvermeidbare_abschriften
            reward = (unvermeidbare_abschriften * -ek_preis) + (verkaufbarer_mehrbestand * -ek_preis * kap_kosten)

    return reward

    # plt.bar(np.arange(0, analyse_tage) - 0.375, abschr_bestand, width=1 / 4, fill=True, label='Abschriftsbestand', color='red', alpha=0.3)
    # plt.bar(np.arange(0, analyse_tage) - 0.125, absatz_bestand, width=1/4, fill=True, label='Absatzbestand', color='blue', alpha=0.3)
    # plt.bar(np.arange(0, analyse_tage) + 0.125, abschriften, width=1/4, fill=True, label='Wahre Abschriften', color='green', alpha=0.3)
    # plt.bar(np.arange(0, analyse_tage) + 0.375, real_bestand, width=1/4, fill=True, label='Realbestand', color='orange', alpha=0.3)
    #
    # plt.legend()
    # plt.show()


bestand = np.array([2, 2, 4, 4, 4, 4, 14])
absatz = np.array([1, 1, 0, 1, 1, 1])
# bestand = np.array([14, 14, 14, 14, 14])
# absatz = np.array([3, 2, 1])
reward = belohnung_bestandsreichweite(bestand, absatz, 3)
