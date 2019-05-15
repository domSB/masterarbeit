# normaler Datafram
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
import multiprocessing
from collections import deque

# für einen Artikel in einem Jahr
def create_df_with_calender(year):
    werktag = pd.tseries.offsets.CustomBusinessDay(weekmask='Mon Tue Wed Thu Fri Sat')
    sonntag = pd.tseries.offsets.CustomBusinessDay(weekmask='Sun')
    start = str(year) + '-01-01'
    stop = str(year) + '-12-31'
    kalender = pd.date_range(start, stop, freq="d")
    platzhalter = [[0, x.dayofweek] for x in kalender]
    kalender = [x.dayofyear for x in kalender]

    new_df = pd.DataFrame(data= platzhalter, index=kalender, columns=["Absatz","Wochentag"])

    sonntage = pd.date_range(start, stop, freq=sonntag)
    sontage = [x.dayofyear for x in sonntage]

    return new_df, sonntage

def copy_data_to_cal_df(big_df, cal_df, artikel):
    """Returns a copy of new_df"""
    
    s = big_df[big_df.Artikel == artikel].copy()
    s = s.reset_index(drop=True)
    wg, olt = s.iloc[0][["Warengruppe", "OrderLeadTime"]]
    s = s.drop(columns=["Zeile", "Abteilung", "Datum", "Artikel", "Warengruppe", "Jahr", "OrderLeadTime"])
    s.set_axis(s["Jahrestag"], inplace=True)
    cal_df.Absatz = s.Absatz
    cal_df = cal_df.fillna(0)
    # Versuch mit Numpy
    cal_df = cal_df.to_numpy(copy=True)

    #for i in s.index.get_values():
    #    cal_df.loc[i,"Absatz"] = s.loc[i, "Absatz"]
    return cal_df, wg, olt




class StockSimulation:
    def __init__(self, df, sample_produkte, preise, wetter, time_series_lenght):
        assert type(df) == pd.core.frame.DataFrame, "Wrong type for DataFrame"
        assert "Artikel" in df.columns, "Artikelnummer Spalte fehlt"
        assert "Warengruppe" in df.columns, "Warengruppe Spalte fehlt"
        assert "Wochentag" in df.columns, "Wochentag Spalte fehlt"
        assert "Jahrestag" in df.columns, "Jahrestag Spalte fehlt"
        assert "Jahr" in df.columns, "Jahr Spalte fehlt"
        assert "Absatz" in df.columns, "Absatz Spalte fehlt"
        assert "OrderLeadTime" in df.columns, "OrderLeadTime Spalte fehlt"
        assert np.array_equal(np.sort(df["Wochentag"].unique()), np.arange(0,6)), "Keine 6 Tage Woche"

        self.df = df.copy()
        self.sample_produkte = sample_produkte
        # self.produkte = self.df["Artikel"].unique()
        #
        #
        # Nur Beispielprodukte, damit es schneller geht
        alle_produkte = self.df["Artikel"].unique()
        np.random.shuffle(alle_produkte)
        self.produkte = alle_produkte[0:self.sample_produkte]
        self.warengruppen = self.df["Warengruppe"].unique()
        self.anz_wg = len(self.warengruppen)
        self.wochentage = np.arange(0,6)

        self.wetter = wetter
        # Anfangsbestand wird zufällig gewählt, für bessere Exploration und Verhindung von lokalen Maxima 
        self.anfangsbestand = pd.DataFrame(np.random.randint(0,10, len(self.produkte)), index=self.produkte)
        self.tage = 365
        self.jahre = self.df["Jahr"].unique()
        self.time_series_lenght = time_series_lenght

        #vorerst
        assert len(self.jahre) == 1
        cal_df, sonntage = create_df_with_calender(2018)
        self.cal_df = cal_df
        self.absatz_data = {}
        self.static_state_data = {}
        for artikel in tqdm(self.df["Artikel"].unique()):
            art_df, wg, olt = copy_data_to_cal_df(self.df, cal_df.copy(), artikel)
            self.absatz_data[artikel] = art_df
            wg = to_categorical(wg, num_classes=self.anz_wg)
            try:
                artikel_preis = preise.loc[artikel]
            except KeyError as error:
                artikel_preis = 0
                print(error)
            if type(artikel_preis) == pd.core.frame.DataFrame:
                artikel_preis = np.array([artikel_preis[artikel_preis.Datum == max(artikel_preis.Datum)]["Preis"].iat[0]])
            elif type(artikel_preis) == pd.core.series.Series:
                artikel_preis = np.array([artikel_preis["Preis"]])
            elif type(artikel_preis) == int:
                artikel_preis = np.array([artikel_preis])
            else:
                raise AssertionError("Unknown Type for Price: {}".format(type(artikel_preis)))
            self.static_state_data[artikel] = {"Warengruppe":wg, "OrderLeadTime": olt, "Preis": artikel_preis}



        self.aktueller_tag = None
        self.aktuelles_produkt = None

    def reset(self):
        """ 
        Neuer State ist ein Numpy Array
        *Bestand
        *Wochentag
        *Warengruppe

        ** Absatz t+1 (echte Tage)
        ** Absatz t+2

        """
        self.episode_fertig = False
        alle_produkte = self.df["Artikel"].unique()
        np.random.shuffle(alle_produkte)
        self.produkte = alle_produkte[0:self.sample_produkte]
        self.anfangsbestand = pd.DataFrame(np.random.randint(0,10, len(self.produkte)), index=self.produkte)
        self.bestand = self.anfangsbestand.copy()
        artikel = self.produkte[0]
        self.erster_tag = 0
        self.aktueller_tag = 0
        self.vergangene_tage = 0
        self.artikel_fertig = False
        self.aktuelles_produkt = artikel
        self.akt_prod_bestand = self.bestand.loc[self.aktuelles_produkt][0]
        self.akt_prod_absatz = self.absatz_data[self.aktuelles_produkt]
        self.akt_prod_wg = self.static_state_data[self.aktuelles_produkt]["Warengruppe"]
        self.akt_prod_preis = self.static_state_data[self.aktuelles_produkt]["Preis"]
        self.akt_prod_olt = self.static_state_data[self.aktuelles_produkt]["OrderLeadTime"]

        absatz, wochentag = self.akt_prod_absatz[self.aktueller_tag]

        if wochentag > 0:
            wochentag -= 1
        else:
            wochentag = 5

        wochentag = to_categorical(wochentag, num_classes=6)

        new_state = np.concatenate([[self.akt_prod_bestand], wochentag, self.akt_prod_wg, self.akt_prod_preis, self.wetter[self.aktueller_tag], self.wetter[self.aktueller_tag+1]])
        
        self.time_series_state = deque(maxlen=self.time_series_lenght)
        for _ in range(self.time_series_lenght):
            self.time_series_state.append(new_state)
        return np.array(self.time_series_state)

    def make_action(self, action):
        self.aktueller_tag += 1
        if self.aktueller_tag % 7 == 0: # Sonntag, anpassen, wenn mehrere Jahre, die nicht mit Montag anfangen
            self.aktueller_tag += 1
        
        self.vergangene_tage += 1

        action = np.array(action).astype(np.uint8)

        absatz, wochentag = self.akt_prod_absatz[self.aktueller_tag -1]
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


        # reward = reward.to_numpy().astype(np.float64)[0]

        wochentag = to_categorical(wochentag, num_classes=6)
        
        new_state = np.concatenate([[self.akt_prod_bestand], wochentag, self.akt_prod_wg, self.akt_prod_preis, self.wetter[self.aktueller_tag], self.wetter[self.aktueller_tag+1]])

        if self.artikel_fertig:
            self.time_series_state = self.time_series_state_neuer_artikel

        self.time_series_state.append(new_state)

        #Hier

        if self.aktueller_tag == self.tage:
            self.artikel_fertig = True
            self.aktueller_tag = self.erster_tag
            self.vergangene_tage = 0
            i_von_akt_produkt = np.where(self.produkte == self.aktuelles_produkt)
            verbleibende_produkte = np.delete(self.produkte, i_von_akt_produkt)
            if len(verbleibende_produkte) == 0:
                self.episode_fertig = True
                state_neuer_artikel = None
            else:
                self.produkte = verbleibende_produkte
                self.aktuelles_produkt = self.produkte[0].copy()
                print("Verbleibende Produkte: ", self.produkte.shape[0])
                self.akt_prod_bestand = self.bestand.loc[self.aktuelles_produkt].copy()[0]
                self.akt_prod_absatz = self.absatz_data[self.aktuelles_produkt]
                self.akt_prod_wg = self.static_state_data[self.aktuelles_produkt]["Warengruppe"]
                self.akt_prod_olt = self.static_state_data[self.aktuelles_produkt]["OrderLeadTime"]
                absatz, wochentag = self.akt_prod_absatz[self.aktueller_tag]
                wochentag = to_categorical(wochentag, num_classes=6)
                state_neuer_artikel = np.concatenate([[self.akt_prod_bestand], wochentag, self.akt_prod_wg, self.akt_prod_preis, self.wetter[self.aktueller_tag], self.wetter[self.aktueller_tag+1]])
                self.time_series_state_neuer_artikel = deque(maxlen=self.time_series_lenght)
                for _ in range(self.time_series_lenght):
                    self.time_series_state_neuer_artikel.append(new_state)
            
        else:
            self.artikel_fertig = False
            self.time_series_state_neuer_artikel = None

        
        return reward, self.artikel_fertig, np.array(self.time_series_state), np.array(self.time_series_state_neuer_artikel), self.episode_fertig
