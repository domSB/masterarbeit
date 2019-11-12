import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import entropy


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

    reward = 0
    # Fallunterscheidungen:
    # 1. Fall Bestandsreichweite == Orderzyklus
    # ==> Perfekter Bestand
    if bestandsreichweite == order_zyklus:
        reward = 0.1 + 0.3 * kum_absatz[order_zyklus - 1]**2
        # Exponentielle steigende Belohnung für Treffen von hohen Absätzen

    # 2. Fall Bestandsreichweite < Orderzyklus
    # ==> Unterbestand mit Fehlmenge
    # elif bestandsreichweite < order_zyklus:
    #     fehlmenge = real_bestand[order_zyklus - 1]
    #     reward = -rohertrag * fehlmenge
    #
    # # 3. Fall Bestandsreichweite > Orderzyklus & Bestand bei Orderzyklus == 0
    # # ==> Bestandsreichweite per Definition höher als Orderzyklus, aber Bestellmenge optimal
    # else:
    #     end_bestand = real_bestand[order_zyklus - 1]
    #     if end_bestand == 0:
    #         reward = 0.1 + 0.3 * kum_absatz[order_zyklus - 1]**2
    #
    # # 4. Fall Bestandsreichweite > Orderzyklus & Bestand bei Orderzyklus > 0
    # # ==> Noch Bestand bei nächstem Liefereingang.
    #     else:
    #         unvermeidbare_abschriften = -abschriften[order_zyklus-1:].sum()
    #         # Bei MHD > 2x Orderzyklus werden Abschriften ggf. mehrfach bestraft.
    #         verkaufbarer_mehrbestand = end_bestand - unvermeidbare_abschriften
    #         reward = (unvermeidbare_abschriften * -ek_preis) + (verkaufbarer_mehrbestand * -ek_preis * kap_kosten)

    return reward


def gradienten_belohnung(ausfall, abschrift):
    z = np.log(3/(ausfall**2+abschrift**2+1))/4 + 3
    return z


class Statistics(object):
    def __init__(self):
        self.data = {}
        self.artikel = -1

    def set_artikel(self, artikel):
        """
        Legt den Artikel fest, für den die aktuellen Statistiken erfasst werden sollen
        :param artikel: Artikelnummer
        :return:
        """
        self.artikel = artikel
        self.data[self.artikel] = np.zeros((0, 7))

    def add(self, other):
        """
        Fügt einen neuen Statistikdatensatz für einen Tag hinzu.
        self.vergangene_tage, action, absatz, reward, self.bestand, self.fehlmenge, self.abschriften
        :param other: np.array([day, action, reward, bestand, fehlmenge, abschrift])
        :return:
        """
        assert other.shape == (7, )
        self.data[self.artikel] = np.concatenate((self.data[self.artikel], other.reshape(1, 7)), axis=0)

    def tage(self, artikel=None):
        if artikel is None:
            artikel = self.artikel
        return self.data[artikel][:, 0]

    def actions(self, artikel=None):
        if artikel is None:
            artikel = self.artikel
        return self.data[artikel][:, 1]

    def absaetze(self, artikel=None):
        if artikel is None:
            artikel = self.artikel
        return self.data[artikel][:, 2]

    def rewards(self, artikel=None):
        if artikel is None:
            artikel = self.artikel
        return self.data[artikel][:, 3]

    def bestand(self, artikel=None):
        if artikel is None:
            artikel = self.artikel
        return self.data[artikel][:, 4]

    def fehlmenge(self, artikel=None):
        if artikel is None:
            artikel = self.artikel
        return self.data[artikel][:, 5]

    def abschrift(self, artikel=None):
        if artikel is None:
            artikel = self.artikel
        return self.data[artikel][:, 6]

    def action_entropy(self, artikel=None):
        if artikel is None:
            artikel = self.artikel
            a_values, a_count = np.unique(self.actions(artikel), return_counts=True)
            abs_values, abs_count = np.unique(self.absaetze(artikel), return_counts=True)
            actions_df = pd.DataFrame(data={'Actions': a_count}, index=a_values)
            absatz_df = pd.DataFrame(data={'Absatz': abs_count}, index=abs_values)
            probas = pd.merge(actions_df, absatz_df, how='outer', left_index=True, right_index=True)
            probas.fillna(0, inplace=True)
            return entropy(probas.Actions, qk=probas.Absatz)

    def plot(self, artikel):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='all')
        x = self.tage(artikel)
        ax1.plot(x, self.bestand(artikel))
        ax1.set_title('Bestandsentwicklung')
        ax2.bar(x, self.actions(artikel), label='Bestellmenge')
        ax2.bar(x, self.absaetze(artikel), label='Absatz')
        ax2.set_title('Getroffene Bestellaktionen')
        ax3.bar(x, self.abschrift(artikel), label='Abschriften')
        ax3.bar(x, self.fehlmenge(artikel), label='Fehlmengen')
        ax3.set_title('Abschriften und Fehlmengen')
        ax4.bar(x, self.rewards(artikel))
        ax4.set_title('Belohungen')
        plt.show()


class StockSimulation(object):
    def __init__(self, simulation_data, pred, hparams):
        """

        :param simulation_data: 4er Tupel aus Labels, dynamischem Zustand, statischem Zustand und der Ids zum zuordnen
        :param pred: Vorberechnete Predictions für schnelleres Training
        :param state_flag: Gibt an, welcher Zustand zurückgegeben werden soll
        0 Nur prediction und Bestandsdaten, 1 mit Zeitstempel, 2 mit statischen Artikelinformationen
        :param reward_flag: Gibt an, welche Belohnungsfunktion verwendet werden soll
        Wählen aus Bestandsreichweite, Bestand, MCGewinn, TDGewinn & Bestandsreichweite V2
        """
        self.lab, self.dyn, self.stat, self.ids = simulation_data
        self.pred = pred
        self.possibles = np.unique(self.ids)
        self.state_flag = hparams.sim_state_group
        self.reward_flag = hparams.reward_func
        self.aktueller_markt = None
        self.aktueller_artikel = None
        self.artikel_absatz = None
        self.vergangene_tage = None
        self.static_state = None
        self.dynamic_state = None
        self.predicted_state = None
        self.kristall_glas = None
        self.tage = None
        self.bestand = None
        self.bestands_frische = None
        self.break_bestand = None
        self.abschriften = 0
        self.fehlmenge = 0
        self.optimal_flag = None
        self.gesamt_belohnung = None
        self.anz_abschriften = 0
        self.anz_fehlmengen = 0
        self.artikel_einkaufspreis = 0.7
        self.artikel_verkaufspreis = 1
        self.artikel_rohertrag = 0.3
        self.kap_kosten = 0.05/365
        self.optimaler_reward = None
        self.placeholder_mhd = 14
        self.bestellrythmus = hparams.bestell_zyklus
        # TODO: Lookup für MHD und OSE, Preise
        self.statistics = Statistics()

    @property
    def state(self):
        state = np.concatenate(
            (
                np.argmax(self.predicted_state[self.vergangene_tage], axis=1),
                np.array([self.bestand, self.fehlmenge / 8, self.abschriften / 8])
            ), axis=0
        )
        if self.state_flag >= 1:
            state = np.concatenate(
                (
                    self.dynamic_state[self.vergangene_tage, 0, -9:],
                    state
                ), axis=0
            )
        if self.state_flag >= 2:
            state = np.concatenate(
                (
                    self.static_state[0, :],
                    state
                )
            )
        return state

    @property
    def info(self):
        return {'Artikel': self.aktueller_artikel, 'Markt': self.aktueller_markt, 'Optimal': self.optimaler_reward}

    def reset(self, artikel_markt=None):
        """
        ...
        """
        if type(artikel_markt) == tuple:
            name_wahl = artikel_markt[0] + artikel_markt[1] * 1000000
            self.aktueller_artikel = artikel_markt[0]
            self.aktueller_markt = artikel_markt[1]
            ids_wahl = np.argwhere(np.isin(self.ids, [name_wahl])).reshape(-1)
            if len(ids_wahl) == 0:
                raise AssertionError('Keine Ids mit diesen Eigenschaften {name}'.format(name=name_wahl))
        elif artikel_markt is not None:
            self.aktueller_markt = int('0' + str(artikel_markt)[:-6])
            self.aktueller_artikel = int(str(artikel_markt)[-6:])
            ids_wahl = np.argwhere(np.isin(self.ids, [artikel_markt])).reshape(-1)
        else:
            position_wahl = np.random.choice(len(self.possibles))
            self.aktueller_markt = int('0' + str(self.possibles[position_wahl])[:-6])
            self.aktueller_artikel = int(str(self.possibles[position_wahl])[-6:])
            ids_wahl = np.argwhere(np.isin(self.ids, self.possibles[position_wahl])).reshape(-1)
        self.static_state = self.stat[ids_wahl]
        self.dynamic_state = self.dyn[ids_wahl]
        self.predicted_state = self.pred[ids_wahl]
        self.kristall_glas = self.lab[ids_wahl]
        self.artikel_absatz = (self.dyn[ids_wahl, 0, 0] * 8).astype(np.int64)
        # Zufälliger Bestand mit maximaler Reichweite von 6 Tagen.
        start_absatz = np.sum(self.artikel_absatz[0:6]).astype(int)
        if start_absatz > 0:
            self.bestand = np.random.choice(start_absatz)
        else:
            self.bestand = 0
        self.bestands_frische = np.ones((self.bestand,), dtype=np.int64) * self.placeholder_mhd
        self.break_bestand = np.sum(self.artikel_absatz) * 2

        self.vergangene_tage = 0
        self.tage = self.dynamic_state.shape[0]

        self.abschriften = 0
        self.fehlmenge = 0
        self.optimal_flag = True
        self.gesamt_belohnung = 0
        self.anz_abschriften = 0
        self.anz_fehlmengen = 0

        self.optimaler_reward = self.artikel_absatz.sum() * self.artikel_rohertrag
        self.statistics.set_artikel(self.aktueller_artikel)
        return self.state, self.info

    def make_action(self, action):
        self.vergangene_tage += self.bestellrythmus
        self.abschriften = 0  # Werden in self.state verwendet, daher keine lokale Var. der Methode
        self.fehlmenge = 0
        action = int(action)
        absatz = self.artikel_absatz[self.vergangene_tage - self.bestellrythmus:self.vergangene_tage].sum()
        done = self.tage <= self.vergangene_tage + self.bestellrythmus
        # BUG: Nach aktueller Implementierung trifft Bestellung am Tag der Bestellung ein und nicht mit 1 Tag Verzug
        # Bei Beachtung der OrderLeadTime, muss der folgende Teil des Skriptes 2-fach ausgeführt werden:
        # - Betrachtung bis Bestellungseingang
        # - Betrachtung ab Bestellungseingang
        self.bestand += action
        self.bestands_frische = np.concatenate(
            (self.bestands_frische, np.ones((action,), dtype=np.int64) * self.placeholder_mhd))

        # Produkte sind ein Tag älter
        self.bestands_frische -= self.bestellrythmus
        # BUG: Wenn ein Feier- oder Sonntag zwischen den Absatztagen lag, altern die Produkte trotzdem nur um einen Tag
        abgelaufene = np.argwhere(self.bestands_frische <= 0).reshape(-1)
        if len(abgelaufene) > 0:
            self.bestands_frische = np.delete(self.bestands_frische, abgelaufene)
            self.abschriften = len(abgelaufene)
            self.bestand -= self.abschriften
            self.optimal_flag = False

        # Absatz abziehen und bewerten:
        if absatz > 0:
            if absatz <= self.bestand:
                self.bestands_frische = self.bestands_frische[absatz:]
                self.bestand -= absatz
            else:
                self.bestands_frische = np.ones((0,), dtype=np.int64)
                self.fehlmenge = absatz - self.bestand
                self.optimal_flag = False
                self.bestand = 0

        # Rewardberechnung
        if self.reward_flag == 'MCGewinn' or self.reward_flag == 'TDGewinn':
            # Abschrift
            r_abschrift = self.abschriften * -self.artikel_einkaufspreis
            # Umsatzausfall
            r_ausfall = self.fehlmenge * -self.artikel_rohertrag
            # Umsatz
            r_umsatz = absatz * self.artikel_rohertrag
            # Kapitalbindung
            r_bestand = -(self.bestand * self.artikel_einkaufspreis) * self.kap_kosten
            # Belohnung für optimale Bestell-Strategien
            if self.reward_flag == 'TDGewinn':
                # Temporal Difference Gewinn gibt jeden Tag eine Belohnung
                reward = r_abschrift + r_ausfall + r_bestand + r_umsatz
                if done and self.optimal_flag:
                    reward += 30
            else:
                # Monte Carlo Gewinn summiert alle Gewinne auf und gibt die Summe am Ende der Episode zurück
                self.gesamt_belohnung += (r_abschrift + r_ausfall + r_bestand + r_umsatz)
                if done:
                    reward = self.gesamt_belohnung
                    if self.optimal_flag:
                        reward += 30
                else:
                    reward = 0

        elif self.reward_flag == 'MCGewinn V2':
            self.anz_abschriften += self.abschriften
            self.anz_fehlmengen += self.fehlmenge
            if done:
                reward = gradienten_belohnung(self.anz_fehlmengen, self.anz_abschriften)
            else:
                reward = 0

        elif self.reward_flag == 'TDGewinn V2':
            reward = gradienten_belohnung(self.fehlmenge*10, self.abschriften*10)

        elif self.reward_flag == 'Bestandsreichweite':
            kommende_absaetze = np.sum(
                self.artikel_absatz[self.vergangene_tage+1:self.vergangene_tage+1+self.bestellrythmus]
            )
            reichweite = self.bestand - kommende_absaetze
            if reichweite == 0:
                if kommende_absaetze > 0:  # Eine Art Importance Sampling für A3C
                    reward = 3
                else:
                    reward = 0.1
            elif reichweite > 0:
                reward = reichweite * - 0.1
            else:
                reward = reichweite * 0.3  # fürs Erste fixe Bestrafung
            reward = np.clip(reward, -3, 3)

        elif self.reward_flag == 'Bestandsreichweite V2':
            if done:
                reward = self.bestand * self.kap_kosten * -self.artikel_einkaufspreis
            else:
                analyse_start = self.vergangene_tage+1
                analyse_stop = min(self.tage + 1, analyse_start + self.placeholder_mhd)
                kommende_absaetze = self.artikel_absatz[analyse_start:analyse_stop]
                if kommende_absaetze.shape[0] < self.placeholder_mhd:
                    kommende_absaetze = np.concatenate(
                        (
                            kommende_absaetze,
                            np.zeros(self.placeholder_mhd - kommende_absaetze.shape[0])
                        )
                    )
                reward = belohnung_bestandsreichweite(
                    self.bestands_frische,
                    kommende_absaetze,
                    self.bestellrythmus,
                    rohertrag=0.3,
                    ek_preis=0.7,
                    kap_kosten=0.05 / 365
                )
                reward = np.clip(reward, -7, 7)

        elif self.reward_flag == 'Bestand':
            # TODO: Ursprüngliche Belohnungsfunktion implementieren
            raise NotImplementedError('Muss noch gecoded werden')

        else:
            raise NotImplementedError('Unbekannte Belohnungsart')

        # Abbruch der Episode bei hoffnungslosem Überbestand
        # if self.bestand > self.break_bestand:
        #     reward = -300
        #     done = True

        self.statistics.add(
            np.array([self.vergangene_tage, action, absatz, reward, self.bestand, self.fehlmenge, self.abschriften])
        )

        return reward,  done, self.state
