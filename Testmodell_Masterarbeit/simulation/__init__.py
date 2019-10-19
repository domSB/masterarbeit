import numpy as np
import matplotlib.pyplot as plt
from agents import Predictor


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
        if artikel in self.data.keys():
            print('|', end='')
        self.data[self.artikel] = np.zeros((0, 7))

    def add(self, other):
        """
        Fügt einen neuen Statistikdatensatz für einen Tag hinzu.
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

    def plot(self, artikel):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
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
    def __init__(self, simulation_data, _predictor_path):
        self.lab, self.dyn, self.stat, self.ids = simulation_data
        self.possibles = np.unique(self.ids)
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
        self.artikel_einkaufspreis = None
        self.artikel_verkaufspreis = None
        self.artikel_rohertrag = None
        self.placeholder_mhd = 14
        # TODO: Lookup für MHD und OSE, Preise
        self.statistics = Statistics()
        # To Speed up Online-Learning
        self.predictor = Predictor()
        self.predictor.build_model(
            dynamic_state_shape=simulation_data[1].shape[2],
            static_state_shape=simulation_data[2].shape[1]
        )
        self.predictor.load_from_weights(_predictor_path)

    @property
    def state(self):
        state = np.concatenate(
            (
                self.predicted_state[self.vergangene_tage].reshape(-1),
                np.array([self.bestand, self.fehlmenge / 8, self.abschriften / 8])
            ), axis=0
        )
        return state

    @property
    def info(self):
        return {'Artikel': self.aktueller_artikel, 'Markt': self.aktueller_markt, 'Kristallglas': self.kristall_glas}

    def reset(self, artikel_markt_tupel=None):
        """
        wahl = np.random.choice(len(possibles), int(len(possibles) * percentage))
            args_test = np.argwhere(np.isin(idx, possibles[wahl])).reshape(-1)
            idx = self.split_helper[:, 0] + self.split_helper[:, 1] * 1000000
        """
        if artikel_markt_tupel:
            id_wahl = artikel_markt_tupel[0] + artikel_markt_tupel[1] * 1000000
            self.aktueller_artikel = artikel_markt_tupel[0]
            self.aktueller_markt = artikel_markt_tupel[1]
        else:
            id_wahl = np.random.choice(len(self.possibles))
            self.aktueller_markt = int('0' + str(self.possibles[id_wahl])[:-6])
            self.aktueller_artikel = int(str(self.possibles[id_wahl])[-6:])
        ids_wahl = np.argwhere(np.isin(self.ids, self.possibles[id_wahl])).reshape(-1)
        self.static_state = self.stat[ids_wahl]
        self.dynamic_state = self.dyn[ids_wahl]
        self.predicted_state = self.predictor.predict(
            {
                'dynamic_input': self.dynamic_state,
                'static_input': self.static_state
            }
        )
        self.kristall_glas = self.lab[ids_wahl]
#        self.artikel_absatz = self.dyn[ids_wahl, 0, 0] * 8
        self.artikel_absatz = self.dyn[ids_wahl, 0, 0]
        # Zufälliger Bestand mit maximaler Reichweite von 6 Tagen.
        start_absatz = np.sum(self.artikel_absatz[0:6]).astype(int)
        if start_absatz > 0:
            self.bestand = np.random.choice(start_absatz)
        else:
            self.bestand = 0
        self.bestands_frische = np.ones((self.bestand,)) * self.placeholder_mhd
        self.break_bestand = np.sum(self.artikel_absatz) * 2

        self.vergangene_tage = 0
        self.tage = self.dynamic_state.shape[0]

        self.abschriften = 0
        self.fehlmenge = 0
        self.optimal_flag = True

        self.artikel_einkaufspreis = 0.7
        self.artikel_verkaufspreis = 1
        self.artikel_rohertrag = self.artikel_verkaufspreis - self.artikel_einkaufspreis

        self.statistics.set_artikel(self.aktueller_artikel)
        return self.state, self.info

    def make_action(self, action):
        self.vergangene_tage += 1
        self.abschriften = 0
        self.fehlmenge = 0
        absatz = self.artikel_absatz[self.vergangene_tage]
        done = self.tage <= self.vergangene_tage + 1
        # Produkte sind ein Tag älter
        # BUG: Wenn ein Feier- oder Sonntag zwischen den Absatztagen lag, altern die Produkte trotzdem nur um einen Tag
        self.bestands_frische -= 1
        abgelaufene = np.argwhere(self.bestands_frische <= 0).reshape(-1)
        if len(abgelaufene) > 0:
            self.bestands_frische = np.delete(self.bestands_frische, abgelaufene)
            self.abschriften = len(abgelaufene)
            self.bestand -= self.abschriften
            self.optimal_flag = False
        # Tagsüber Absatz abziehen und bewerten:
        if absatz > 0:
            if absatz <= self.bestand:
                self.bestands_frische = self.bestands_frische[int(absatz):]
                self.bestand -= absatz
            else:
                self.bestands_frische = np.ones((0,))
                self.fehlmenge = absatz - self.bestand
                self.optimal_flag = False
                self.bestand = 0

        self.bestand += action
        self.bestands_frische = np.concatenate((self.bestands_frische, np.ones((action,)) * self.placeholder_mhd))

        # Rewardberechnung
        # Abschrift
        r_abschrift = self.abschriften * -self.artikel_einkaufspreis
        # Umsatzausfall
        r_ausfall = self.fehlmenge * -self.artikel_rohertrag
        # Umsatz
        # r_umsatz = absatz * self.artikel_rohertrag
        # Kapitalbindung
        r_bestand = -(self.bestand * self.artikel_einkaufspreis) * 0.05/365
        # Belohnung für optimale Bestell-Strategien
        r_optimal = 0
        if done:
            if self.optimal_flag:
                r_optimal = 30
        # Abbruch der Episode
        if self.bestand > self.break_bestand:
            reward = -300
            done = True
        else:
            reward = r_abschrift + r_ausfall + r_bestand + r_optimal  # + r_umsatz

        self.statistics.add(
            np.array([self.vergangene_tage, action, absatz, reward, self.bestand, self.fehlmenge, self.abschriften])
        )

        return reward,  done, self.state


def test(simulation, lenght):
    for i in range(lenght):
        print("Run {}".format(i))
        _ = simulation.reset()
        done = False
        k = 0
        while not done:
            k += 1
            reward, done, state = simulation.make_action(3)
            if simulation.aktueller_tag.date() in simulation.feiertage:
                print("Feiertag")
                print(simulation.aktueller_tag)
        print(' %s Tage durchlaufen' % k)
