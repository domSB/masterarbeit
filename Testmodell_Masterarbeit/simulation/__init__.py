# normaler Datafram
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
import multiprocessing
from collections import deque
import os


def load_weather(path, start, ende):
    df = pd.read_csv(
        path, 
        index_col="date", 
        memory_map=True

        )
    df = df.drop(columns=["Unnamed: 0", "HauptGruppe", "NebenGruppe"])
    # df = df.sort_index()
    # df["Datum"] = df.index.get_values()
    # df["Datum"] = pd.to_datetime(df["Datum"]*24*3600, unit='s')
    # df = df[df.Datum.dt.year.isin([2018,2019])]
    # df = df[df.Datum.dt.dayofweek != 6]
    df = df[df.index.isin(range(start, ende +3))]
    # Plus 2 Tage, da Wetter von morgen und übermorgen
    return df.to_numpy()
    
def load_prices(path):
    df = pd.read_csv(
        path, 
        names=["Zeile", "Preis","Artikelnummer","Datum"],
        header=0,
        index_col="Artikelnummer", 
        memory_map=True
        )
    df = df.sort_index()
    df = df.drop(columns=["Zeile"])
    return df

def load_sales(path):
    #TODO: Statische Artikelinfo aus der Absatztabelle rausnehmen. (Warengruppe, Abteilung)
    """
     for artikel in train_data["Artikel"].unique():
         warengruppen.append([artikel, train_data.loc[(slice(None), slice(5550,5550)),:].iloc[0].Warengruppe])
    """

    df = pd.read_csv(
        path, 
        names=["Zeile", "Datum", "Artikel", "Absatz", "Warengruppe", "Abteilung"], 
        header=0, 
        parse_dates=[1], 
        index_col=[1, 2],
        memory_map=True
        )
    df.dropna(how='any', inplace=True)
    df["Warengruppe"] = df["Warengruppe"].astype(np.uint8)
    df = df.drop(columns=['Abteilung', 'Zeile'])
    # Warengruppen auswählen
    # 13 Frischmilch
    # 14 Joghurt
    # 69 Tabak
    # 8 Obst Allgemen

    # warengruppen = [8, 13, 14, 69 ]
    warengruppen = [8]
    df = df[df['Warengruppe'].isin(warengruppen)]
    for i, wg in enumerate(warengruppen):
        df.loc[df.Warengruppe == wg, "Warengruppe"] = i
    df["Datum"] = df.index.get_level_values('Datum')
    df["Artikel"] = df.index.get_level_values('Artikel').astype(np.int32)
    # df["Wochentag"] = df["Datum"].apply(lambda x:x.dayofweek)
    # df["Jahrestag"] = df["Datum"].apply(lambda x:x.dayofyear)
    df["UNIXTag"] = df["Datum"].astype(np.int64)/(1000000000 * 24 * 3600)
    df["Jahr"] = df["Datum"].apply(lambda x:x.year)
    # df = df.drop(columns=['Datum'])
    df = df.sort_index()
    
    
    test_data = df[df["Jahr"]==2019]
    train_data = df[df["Jahr"]==2018]
    return test_data, train_data


def copy_data_to_numpy(big_df, artikel, start, end):
    """Returns a numpy array with lenght = self.kalendertage. Days without Sales are filled with zeros"""
    s = big_df[big_df.Artikel == artikel].copy()
    s.set_index(s.UNIXTag, inplace=True)
    wg = s.iloc[0][["Warengruppe"]][0]
    s = s.drop(columns=["Datum", "Artikel", "Warengruppe", "Jahr", "UNIXTag"])
    s = s.reindex(range(int(start), int(end+1)), fill_value=0)

    return s.to_numpy(), wg




class StockSimulation:
    def __init__(self, data_dir, time_series_lenght):
        """
        Lädt Daten selbstständig aus Data_dir und erstellt das Simulationsmodell. 
        1. Episode entspricht einem Durchlauf mit einem Artikel.
        
        
        
        """

        test_data, train_data = load_sales(os.path.join(data_dir, '3 absatz_altforweiler.csv'))

        self.df = train_data

        self.start_tag = int(min(train_data["UNIXTag"]))
        self.end_tag = int(max(train_data["UNIXTag"]))
        self.kalender_tage = self.end_tag - self.start_tag + 1

        preise = load_prices(os.path.join(data_dir, '3 preise_altforweiler.csv'))

        self.wetter = load_weather(os.path.join(data_dir, '2 wetter_saarlouis.csv'), self.start_tag, self.end_tag)
        
        self.warengruppen = self.df["Warengruppe"].unique()
        self.anz_wg = len(self.warengruppen)

        self.anfangsbestand = np.random.randint(0,10)

        self.time_series_lenght = time_series_lenght

        olt = 1

        self.absatz_data = {}
        self.static_state_data = {}
        for artikel in tqdm(self.df["Artikel"].unique()):
            art_df, wg = copy_data_to_numpy(self.df, artikel, self.start_tag, self.end_tag)
            self.absatz_data[artikel] = art_df
            wg = to_categorical(wg, num_classes=self.anz_wg)

            artikel_preis = preise.loc[artikel]

            if type(artikel_preis) == pd.core.frame.DataFrame:
                artikel_preis = np.array([artikel_preis[artikel_preis.Datum == max(artikel_preis.Datum)]["Preis"].iat[0]])
            elif type(artikel_preis) == pd.core.series.Series:
                artikel_preis = np.array([artikel_preis["Preis"]])
            elif type(artikel_preis) == int:
                artikel_preis = np.array([artikel_preis])
            else:
                raise AssertionError("Unknown Type for Price: {}".format(type(artikel_preis)))
            self.static_state_data[artikel] = {"Warengruppe":wg, "OrderLeadTime": olt, "Preis": artikel_preis}

        self.aktueller_tag = self.start_tag
        self.aktuelles_produkt = self.df["Artikel"].sample(1).to_numpy()[0]

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

    def reset(self):
        """ 
        Neuer State ist ein Numpy Array
        *Bestand
        *Wochentag
        *Warengruppe

        ** Absatz t+1 (echte Tage)
        ** Absatz t+2

        """
        self.fertig = False

        self.anfangsbestand = np.random.randint(0,10)

        self.aktueller_tag = self.start_tag
        self.vergangene_tage = 0
        
        self.aktuelles_produkt = self.df["Artikel"].sample(1).to_numpy()[0]

        self.akt_prod_bestand = self.anfangsbestand
        self.akt_prod_absatz = self.absatz_data[self.aktuelles_produkt]
        self.akt_prod_wg = self.static_state_data[self.aktuelles_produkt]["Warengruppe"]
        self.akt_prod_preis = self.static_state_data[self.aktuelles_produkt]["Preis"]
        self.akt_prod_olt = self.static_state_data[self.aktuelles_produkt]["OrderLeadTime"]

        absatz = self.akt_prod_absatz[self.vergangene_tage]

        wochentag = self.aktueller_tag % 7

        wochentag = to_categorical(wochentag, num_classes=7)

        new_state = self.create_new_state(wochentag)
        
        self.time_series_state = deque(maxlen=self.time_series_lenght)
        for _ in range(self.time_series_lenght):
            self.time_series_state.append(new_state)
        return np.array(self.time_series_state), {"Artikel": self.aktuelles_produkt}

    def make_action(self, action):
        if self.fertig == True:
            raise AssertionError("Simulation für diesen Artikel fertig. Simulation zurücksetzen")

        absatz = self.akt_prod_absatz[self.vergangene_tage][0]

        self.aktueller_tag += 1
        self.vergangene_tage += 1

        if self.aktueller_tag % 7 == 3: # Sonntag
            self.aktueller_tag += 1
            self.vergangene_tage += 1
        
        wochentag = self.aktueller_tag % 7

        # Action ist die Bestellte Menge an Artikeln
        # Tagsüber Absatz abziehen:
        self.akt_prod_bestand -= absatz

        # Nachmittag: Bestellung kommt an
        self.akt_prod_bestand += action

        # Abend: Bestand wird bewertet
        if self.akt_prod_bestand >= 0:
            reward = np.exp(-self.akt_prod_bestand/5)
        if self.akt_prod_bestand < 0:
            reward = np.expm1(self.akt_prod_bestand/2)
            # Nichtnegativität des Bestandes
            self.akt_prod_bestand = 0

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
        while not done:
            reward, done, state = simulation.make_action(3)

