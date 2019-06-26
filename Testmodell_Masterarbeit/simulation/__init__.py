# normaler Datafram
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
import multiprocessing
from collections import deque
import os
import pickle

def load_artikel(path):
    df = pd.read_csv(
        path,  
        memory_map=True,
        index_col="Artikel"
        )
    return df

def load_weather(path, start, ende):
    df = pd.read_csv(
        path, 
        index_col="date", 
        memory_map=True

        )
    df = df.drop(columns=["Unnamed: 0", "HauptGruppe", "NebenGruppe"])
    df = df[df.index.isin(range(start, ende +3))]
    # Plus 2 Tage, da Wetter von morgen und übermorgen verwendet wird
    return df.to_numpy()
    
def load_prices(path):
    df = pd.read_csv(
        path, 
        header=0,
        names = ['Zeile', 'Preis', 'Artikel', 'Datum'],
        index_col='Artikel', 
        parse_dates=['Datum'],
        memory_map=True
        )
    df = df.sort_index()
    df = df.drop(columns=["Zeile"])
    return df

def load_sales(path, artikel_maske):
    #TODO: Statische Artikelinfo aus der Absatztabelle rausnehmen. (Warengruppe, Abteilung)
    """
     for artikel in train_data["Artikel"].unique():
         warengruppen.append([artikel, train_data.loc[(slice(None), slice(5550,5550)),:].iloc[0].Warengruppe])
    """
    df = pd.read_csv(
        path, 
        index_col=['Markt', 'Datum', 'Artikel'], 
        parse_dates=['Datum'], 
        memory_map=True
        )
    df = df.drop(columns=['Unnamed: 0'])
    df["Datum"] = df.index.get_level_values('Datum')
    df["UNIXDatum"] = df["Datum"].astype(np.int64)/(1000000000 * 24 * 3600)
    df["Artikel"] = df.index.get_level_values('Artikel')

    df = df[df.Artikel.isin(artikel_maske)]
    df = df.sort_index()

    maerkte = pd.unique(df.index.get_level_values('Markt'))


    test_data = {}
    train_data = {}
    train = df[df.Datum.dt.year.isin([2017, 2018])]
    test = df[df.Datum.dt.year.isin([2019])]
    timeline = {
    "Train": {
        "Start": int(min(train.UNIXDatum)),
        "Ende": int(max(train.UNIXDatum))
        },
    "Test": {
        "Start": int(min(test.UNIXDatum)),
        "Ende": int(max(test.UNIXDatum))
        }
    }
    for markt in maerkte:
        temp_train = train.loc[(slice(markt,markt),slice(None),slice(None)),:]
        temp_test = test.loc[(slice(markt,markt),slice(None),slice(None)),:]

        #
        # Die Timeline ist je Markt unterschiedlich. 
        # Mit neuen Daten testen, ob SQL-Script Error oder andere Fehlerherkunft.
        #

        print("ping .")
        test_data[markt] = {
            artikel: copy_data_to_numpy(
                temp_test, artikel, 
                timeline['Test']['Start'], 
                timeline['Test']['Ende']
                ) for artikel in artikel_maske}
        print("ping ..")
        train_data[markt] = {
            artikel: copy_data_to_numpy(
                temp_train, artikel, 
                timeline['Train']['Start'], 
                timeline['Train']['Ende']
                ) for artikel in artikel_maske}
        print("ping ...")

    return test_data, train_data, timeline


def copy_data_to_numpy(big_df, artikel, start, end):
    """Returns a numpy array with lenght = self.kalendertage. Days without Sales are filled with zeros"""
    s = big_df[big_df.Artikel == artikel].copy()
    s.set_index(s.UNIXDatum, inplace=True)
    s = s.drop(columns=["Datum", "Artikel", "UNIXDatum"])
    s = s.reindex(range(int(start), int(end+1)), fill_value=0)

    return s.to_numpy()




class StockSimulation:
    def __init__(self, data_dir, time_series_lenght, use_pickled, save_pickled, is_trainer, test_data=None, timeline=None):
        """
        Lädt Daten selbstständig aus Data_dir und erstellt das Simulationsmodell. 
        1. Episode entspricht einem Durchlauf mit einem Artikel.
        
        """
        #TODO: Laden der Absatzdaten an neue .csv-Dateien anpassen
        self.artikelstamm = load_artikel(os.path.join(data_dir, '1 Artikelstamm.csv'))

        # Warengruppen auswählen
        # 13 Frischmilch
        # 14 Joghurt
        # 69 Tabak
        # 8 Obst Allgemen
        warengruppen_maske = [8, 13, 14, 69 ]
        artikel_maske = self.artikelstamm[self.artikelstamm.Warengruppe.isin(warengruppen_maske)].index.values

        if is_trainer:
            # FÜr schnelleres Ausführen, wenn sich die Daten nicht ändern.
            if use_pickled:
                with open("data/train_data.pickle", "rb") as file:
                    train_data = pickle.load(file)

                with open("data/test_data.pickle", "rb") as file:
                    test_data = pickle.load(file)

                with open("data/timeline.pickle", "rb") as file:
                    timeline = pickle.load(file)
            else:
                test_data, train_data, timeline = load_sales(os.path.join(data_dir, '1 Absatz.csv'), artikel_maske)

            if save_pickled:
                with open("data/train_data.pickle", "wb") as file:
                    pickle.dump(train_data, file)

                with open("data/test_data.pickle", "wb") as file:
                    pickle.dump(test_data, file)

                with open("data/timeline.pickle", "wb") as file:
                    pickle.dump(timeline, file)
            self.absatz_data = train_data
            self.start_tag =timeline['Train']['Start']
            self.end_tag = timeline['Train']['Ende']
            self.test_data = test_data
            self.timeline_for_validator = timeline
        else:
            self.absatz_data = test_data
            self.start_tag =timeline['Test']['Start']
            self.end_tag = timeline['Test']['Ende']


        self.kalender_tage = self.end_tag - self.start_tag + 1

        preise = load_prices(os.path.join(data_dir, '1 Preise.csv'))
        #TODO: 1 Preise.csv für Artikel aus Altenkessel anpassen

        self.wetter = load_weather(os.path.join(data_dir, '1 Wetter.csv'), self.start_tag, self.end_tag)
        
        """
        Bin bis hier hin gekommen

        """
        self.warengruppen = warengruppen_maske
        self.anz_wg = len(self.warengruppen)
        self.artikel = artikel_maske
        self.maerkte = list(self.absatz_data.keys())

        self.time_series_lenght = time_series_lenght

        olt = 1  # Fürs erste
        self.fertig = None
        self.anfangsbestand = None
        self.vergangene_tage = None
        self.aktuelles_produkt = None
        self.aktueller_markt = None
        self.akt_prod_bestand = None
        self.akt_prod_absatz = None
        self.akt_prod_wg = None
        self.akt_prod_preis = None
        self.akt_prod_olt = None
        self.time_series_state = None
        self.stat_theo_bestand = None
        self.stat_fakt_bestand = None

        self.static_state_data = {}
        for artikel in tqdm(self.artikel):
            warengruppennummer = self.artikelstamm.loc[artikel].Warengruppe
            warengruppen_index = self.warengruppen.index(warengruppennummer)
            warengruppen_state = to_categorical(warengruppen_index, num_classes=self.anz_wg)

            try:
                artikel_preis = preise.loc[artikel]
            except KeyError: # Weil .csv aktuell nur für einen Markt
                artikel_preis = 0
            # Work Around aufgrund schlechter Ausgangsdaten
            if type(artikel_preis) == pd.core.frame.DataFrame:
                artikel_preis = np.array(
                    [artikel_preis[artikel_preis.Datum == max(artikel_preis.Datum)]["Preis"].iat[0]]
                )
            elif type(artikel_preis) == pd.core.series.Series:
                artikel_preis = np.array([artikel_preis["Preis"]])
            elif type(artikel_preis) == int:
                artikel_preis = np.array([artikel_preis])
            else:
                raise AssertionError("Unknown Type for Price: {}".format(type(artikel_preis)))
            self.static_state_data[artikel] = {
                "Warengruppe": warengruppen_state, 
                "OrderLeadTime": olt, 
                "Preis": artikel_preis
                }

        self.aktueller_tag = self.start_tag
        
    def get_test_data(self):
        """ Methode, um die Testdaten aus dem Trainer zu kopieren."""
        return self.test_data, self.timeline_for_validator

    def del_test_data(self):
        """ 
        Methode, um die Testdaten aus dem Trainer zu löschen, 
        sobald diese in eine neue Instanz der Simulation kopiert wurden.
        """
        del self.test_data
        del self.timeline_for_validator
        return

    def create_new_state(self, wochentag):
        new_state = np.concatenate(
            [
                np.array([self.akt_prod_bestand]), 
                wochentag, 
                self.akt_prod_wg, 
                self.akt_prod_preis, 
                self.wetter[self.vergangene_tage], 
                self.wetter[self.vergangene_tage+1]
                ]
            )
        return new_state

    def reset(self, artikel=None, markt=None):
        """ 
        Methode für das Zurücksetzen der Simulation. Parameter artikel ist optional. 
        Falls weggelassen, wird ein Artikel zufällig gewählt (Für Training).
        Für eine Evaluation des Agenten, können eigene Artikel festgesetzt werden.
        """
        self.fertig = False
        self.stat_theo_bestand = []
        self.stat_fakt_bestand = []
        self.anfangsbestand = np.random.randint(0, 10)
        self.aktueller_tag = self.start_tag
        self.vergangene_tage = 0
        if artikel:
            assert artikel in self.artikel, "Simulation kennt diesen Artikel nicht."
            self.aktuelles_produkt = artikel
        else:
            self.aktuelles_produkt = np.random.choice(self.artikel, 1)[0]
        
        if markt:
            assert markt in self.maerkte, "Simulation kennt diesen Markt nicht."
            self.aktueller_markt = markt
        else:
            self.aktueller_markt = np.random.choice(self.maerkte, 1)[0]

        self.akt_prod_bestand = self.anfangsbestand
        self.akt_prod_absatz = self.absatz_data[self.aktueller_markt][self.aktuelles_produkt]
        self.akt_prod_wg = self.static_state_data[self.aktuelles_produkt]["Warengruppe"]
        self.akt_prod_preis = self.static_state_data[self.aktuelles_produkt]["Preis"]
        self.akt_prod_olt = self.static_state_data[self.aktuelles_produkt]["OrderLeadTime"]

        wochentag = self.aktueller_tag % 7

        wochentag = to_categorical(wochentag, num_classes=7)

        new_state = self.create_new_state(wochentag)
        
        self.time_series_state = deque(maxlen=self.time_series_lenght)
        for _ in range(self.time_series_lenght):
            self.time_series_state.append(new_state)
        return np.array(self.time_series_state), {"Artikel": self.aktuelles_produkt}

    def make_action(self, action):
        if self.fertig:
            raise AssertionError("Simulation für diesen Artikel fertig. Simulation zurücksetzen")

        absatz = self.akt_prod_absatz[self.vergangene_tage][0]

        self.aktueller_tag += 1
        self.vergangene_tage += 1

        if self.aktueller_tag % 7 == 3: # Sonntag
            self.aktueller_tag += 1
            self.vergangene_tage += 1
        
        wochentag = self.aktueller_tag % 7

        #TODO: OrderLeadTime durch eine action-Deque realisieren

        # Action ist die Bestellte Menge an Artikeln
        # Tagsüber Absatz abziehen und bewerten:
        self.akt_prod_bestand -= absatz
        self.stat_theo_bestand.append(self.akt_prod_bestand)

        if self.akt_prod_bestand >= 27.5:
            reward = 0.004992 - (self.akt_prod_bestand-27.5)/1000
        elif self.akt_prod_bestand >= 1:
            reward = np.exp((1-self.akt_prod_bestand)/5)
        else:
            reward = np.exp((self.akt_prod_bestand-1)*1.5)-1
            # Nichtnegativität des Bestandes
            self.akt_prod_bestand = 0
        self.stat_fakt_bestand.append(self.akt_prod_bestand)

        # Nachmittag: Bestellung kommt an und wird verräumt
        self.akt_prod_bestand += action

        wochentag = to_categorical(wochentag, num_classes=7)
        
        new_state = self.create_new_state(wochentag)

        self.time_series_state.append(new_state)

        if self.vergangene_tage == self.kalender_tage -1:
            self.fertig = True

        return reward, self.fertig, np.array(self.time_series_state)

def test(simulation, lenght):
    for i in range(lenght):
        print("Run {}".format(i))
        state = simulation.reset()
        done = False
        k = 0
        while not done:
            k += 1
            reward, done, state = simulation.make_action(3)
        print(' %s Tage durchlaufen' % k)

""" 
For Testing and Debugging

"""
#LOAD = False
#SAVE = True
#TRAINER = True
#VALIDATOR = False
#LENGTH = 10

#simulation = StockSimulation('data', LENGTH, LOAD, SAVE, TRAINER)
#test_data, timeline = simulation.get_test_data()
#simulation.del_test_data()
#validator = StockSimulation('data', LENGTH, LOAD, SAVE, VALIDATOR, test_data, timeline)
#print("Test Simulation")
#test(simulation, 10)
#print("Test Validator")
#test(validator, 10)