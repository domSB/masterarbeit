import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
from collections import deque
import os
import pickle
from calender import get_german_holiday_calendar

import matplotlib.pyplot as plt


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
            print("Statistik für diesen Artikel schon vorhanden. Überschreibe Daten.")
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
        ax2.legent()
        ax3.bar(x, self.abschrift(artikel), label='Abschriften')
        ax3.bar(x, self.fehlmenge(artikel), label='Fehlmengen')
        ax3.set_title('Abschriften und Fehlmengen')
        ax3.legent()
        ax4.bar(x, self.rewards(artikel))
        ax4.set_title('Belohungen')
        plt.show()


class StockSimulation(object):
    def __init__(self, simulation_data):
        self.lab, self.dyn, self.stat, self.ids = simulation_data
        self.possibles = np.unique(self.ids)
        self.aktueller_markt = None
        self.aktueller_artikel = None
        self.artikel_absatz = None
        self.vergangene_tage = None
        self.static_state = None
        self.dynamic_state = None
        self.tage = None
        self.bestand = None
        self.bestands_frische = None
        self.break_bestand = None
        self.artikel_einkaufspreis = None
        self.artikel_verkaufspreis = None
        self.artikel_rohertrag = None
        # TODO: Lookup für MHD und OSE, Preise
        self.statistics = Statistics()

    @property
    def state(self):
        state = {
            'RegressionState': {
                'dynamic_input': np.expand_dims(self.dyn[self.vergangene_tage], axis=0),
                'static_input': np.expand_dims(self.stat[self.vergangene_tage], axis=0)
            },
            'AgentState': np.array([self.bestand])
        }
        return state

    @property
    def info(self):
        return {'Artikel': self.aktueller_artikel, 'Markt': self.aktueller_markt}

    def reset(self, artikel_markt_tupel=None):
        """
        wahl = np.random.choice(len(possibles), int(len(possibles) * percentage))
            args_test = np.argwhere(np.isin(idx, possibles[wahl])).reshape(-1)
            idx = self.split_helper[:, 0] + self.split_helper[:, 1] * 1000000
        """
        if artikel_markt_tupel:
            id_wahl = artikel_markt_tupel[0] + artikel_markt_tupel[1] * 1000000
        else:
            id_wahl = np.random.choice(len(self.possibles))
        self.aktueller_markt, self.aktueller_artikel = int('0' + str(id_wahl)[:-6]), int(str(id_wahl)[-6:])
        ids_wahl = np.argwhere(np.isin(self.ids, self.possibles[id_wahl])).reshape(-1)
        self.static_state = self.stat[ids_wahl]
        self.dynamic_state = self.dyn[ids_wahl]
#        self.artikel_absatz = self.dyn[ids_wahl, 0, 0] * 8
        self.artikel_absatz = self.dyn[ids_wahl, 0, 0]
        # Zufälliger Bestand mit maximaler Reichweite von 6 Tagen.
        start_absatz = np.sum(self.artikel_absatz[0:6]).astype(int)
        if start_absatz > 0:
            self.bestand = np.random.choice(start_absatz)
        else:
            self.bestand = 0
        placeholder_mhd = 14
        self.bestands_frische = np.ones((self.bestand,)) * placeholder_mhd
        self.break_bestand = np.sum(self.artikel_absatz) * 2

        self.vergangene_tage = 0
        self.tage = self.dynamic_state.shape[0]

        self.artikel_einkaufspreis = 0.7
        self.artikel_verkaufspreis = 1
        self.artikel_rohertrag = self.artikel_verkaufspreis - self.artikel_einkaufspreis

        self.statistics.set_artikel(self.aktueller_artikel)
        return self.state, self.info

    def make_action(self, action):
        self.vergangene_tage += 1
        absatz = self.artikel_absatz[self.vergangene_tage]
        done = self.tage <= self.vergangene_tage + 1
        anz_fehlartikel = 0
        # Produkte sind ein Tag älter
        self.bestands_frische -= 1
        abgelaufene = np.argwhere(self.bestands_frische <= 0).reshape(-1)
        if len(abgelaufene) > 0:
            self.bestands_frische = np.delete(self.bestands_frische, abgelaufene)
        # Tagsüber Absatz abziehen und bewerten:
        if absatz > 0:
            if absatz <= self.bestand:
                self.bestands_frische = self.bestands_frische[int(absatz):]
                self.bestand -= absatz
            else:
                self.bestands_frische = np.ones((0,))
                anz_fehlartikel = absatz - self.bestand
                self.bestand = 0

        self.bestand += action

        # Rewardberechnung
        # Abschrift
        r_abschrift = len(abgelaufene) * -self.artikel_einkaufspreis
        # Umsatzausfall
        r_ausfall = anz_fehlartikel * -self.artikel_rohertrag
        # Umsatz
        r_umsatz = absatz * self.artikel_rohertrag
        # Kapitalbindung
        r_bestand = -(self.bestand * self.artikel_einkaufspreis) * 0.05/365
        # Abbruch der Episode
        if self.bestand > self.break_bestand:
            reward = -30
            done = True
        else:
            reward = r_abschrift + r_ausfall + r_umsatz + r_bestand

        self.statistics.add(
            np.array([self.vergangene_tage, action, absatz, reward, self.bestand, anz_fehlartikel, len(abgelaufene)])
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
