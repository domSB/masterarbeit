import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
from collections import deque
import os
import pickle
from calender import get_german_holiday_calendar
from data.preparation.clean import Datapipeline


def incremental_mean(step, old_mean, new_x):
    new_mean = new_x/step + (step-1)/step*old_mean
    return new_mean


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
    df = df.drop(columns=["Unnamed: 0"])
    df = df[df.index.isin(range(start, ende +3))]
    # Plus 2 Tage, da Wetter von morgen und übermorgen verwendet wird
    return df.to_numpy()


def load_prices(path):
    df = pd.read_csv(
        path, 
        header=0,
        names=['Zeile', 'Preis', 'Artikel', 'Datum'],
        index_col='Artikel', 
        parse_dates=['Datum'],
        memory_map=True
        )
    df = df.sort_index()
    df = df.drop(columns=["Zeile"])
    return df


def load_promotions(path):
    df = pd.read_csv(
        path,
        header=0,
        index_col='Artikel',
        parse_dates=['DatumAb', 'DatumBis'],
        memory_map=True
    )
    return df


def load_sales(path, artikel_maske, is_trainer):
    """
    
    """
    df = pd.read_csv(
        path, 
        index_col=['Markt', 'Datum', 'Artikel'], 
        parse_dates=['Datum'], 
        memory_map=True
        )
    df.reset_index(inplace=True)
    # df["Datum"] = df.index.get_level_values('Datum')
    df["UNIXDatum"] = df["Datum"].astype(np.int64)/(1000000000 * 24 * 3600)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.set_index(["Markt", "UNIXDatum"], inplace=True)

    df = df[df.Artikel.isin(artikel_maske)]
    df = df.sort_index()

    maerkte = pd.unique(df.index.get_level_values('Markt'))
    artikel = pd.unique(df.Artikel)

    test_data = {}
    train_data = {}
    train = df[df.Datum.dt.year.isin([2016, 2017])].copy()
    test = df[df.Datum.dt.year.isin([2018])].copy()
    # Keine 2019 Absatzdaten verwenden, da historische Wetterdaten nur bis April und Absatzdaten bis Juni
    train.drop(columns=['Datum'], inplace=True)
    test.drop(columns=['Datum'], inplace=True)
    timeline = {
        "Train": {
            "Start": int(min(train.index.get_level_values('UNIXDatum'))),
            "Ende": int(max(train.index.get_level_values('UNIXDatum')))
            },
        "Test": {
            "Start": int(min(test.index.get_level_values('UNIXDatum'))),
            "Ende": int(max(test.index.get_level_values('UNIXDatum')))
            }
    }
    markt_artikel = {"Train": {}, "Test": {}}
    for markt in maerkte:
        temp_train = train.loc[markt]
        temp_test = test.loc[markt]

        print("ping .")
        test_data[markt] = get_grouped_dict(temp_test, timeline['Test']['Start'],  timeline['Test']['Ende'])
        print("ping ..")
        train_data[markt] = get_grouped_dict(temp_train, timeline['Train']['Start'],  timeline['Train']['Ende'])
        print("ping ...")
        markt_artikel['Train'][markt] = np.array(list(train_data[markt].keys()))
        markt_artikel['Test'][markt] = np.array(list(test_data[markt].keys()))
    if is_trainer:
        return train_data, timeline['Train'], artikel,  markt_artikel['Train']
    else:
        return test_data, timeline['Test'], artikel, markt_artikel['Test']


def get_grouped_dict(df, start, end):
    """

    """
    grouped = df.groupby('Artikel')
    grouped = grouped.apply(lambda x: x.reindex(range(int(start), int(end+1)), fill_value=0))
    grouped.drop(columns=["Artikel"], inplace=True)
    grouped.index = grouped.index.droplevel(1)
    grouped = dict(tuple(grouped.groupby('Artikel')))
    grouped = {id: frame.to_numpy() for id, frame in grouped.items()}

    return grouped


class StockSimulation:
    def __init__(self, data_dir, time_series_lenght, use_pickled, save_pickled, is_trainer, simulation_group, test_data=None, timeline=None):
        """
        Lädt Daten selbstständig aus Data_dir und erstellt das Simulationsmodell.
        Trainigsdurchlauf nur mit pickled-Data. Preprocessing führt langsames Reindexing durch und benötigt 20+ min.
        1. Episode entspricht einem Durchlauf mit einem Artikel.
        
        """

        warengruppen_maske = [1, 12, 55, 80, 17, 77, 71, 6, 28]
        # warengruppen_maske = [77]
        self.warengruppen = warengruppen_maske
        self.anz_wg = len(self.warengruppen)

        if is_trainer:
            filename = 'data/simulation.' + simulation_group + '.pickle'
        else:
            filename = 'data/validator.' + simulation_group + '.pickle'
            # Für schnelleres Ausführen, wenn sich die Daten nicht ändern.

        if use_pickled:
            with open(filename, "rb") as file:
                self.absatz_data = pickle.load(file)
                timeline = pickle.load(file)
                self.artikelstamm = pickle.load(file)
                self.artikel = pickle.load(file)
                self.wetter = pickle.load(file)
                self.kalender_tage = pickle.load(file)
                self.static_state_data = pickle.load(file)
                self.markt_artikel = pickle.load(file)
            self.start_tag = timeline['Start']
            self.end_tag = timeline['Ende']
            self.kalender_tage = self.end_tag - self.start_tag + 1
        else:
            self.artikelstamm = load_artikel(os.path.join(data_dir, '1 Artikelstamm.csv'))
            self.artikelstamm = self.artikelstamm[self.artikelstamm.Warengruppe.isin(warengruppen_maske)]
                
            self.artikel = np.unique(self.artikelstamm.index.values)

            self.absatz_data, timeline, self.artikel, self.markt_artikel = load_sales(
                os.path.join(data_dir, '1 Absatz.' + simulation_group + '.csv'),
                self.artikel,
                is_trainer
            )

            self.start_tag = timeline['Start']
            self.end_tag = timeline['Ende']

            preise = load_prices(os.path.join(data_dir, '1 Preise.csv'))

            aktionen = load_promotions(os.path.join(data_dir, '1 Preisaktionen.csv'))

            self.wetter = load_weather(os.path.join(data_dir, '1 Wetter.csv'), self.start_tag, self.end_tag)

            self.kalender_tage = self.end_tag - self.start_tag + 1

            olt = np.array([1])  # Fürs erste

            """ Statische Artikel Informationen """
            # region StatischeArtikelInformationen
            self.static_state_data = {}
            for artikel in tqdm(self.artikel):
                artikel_data = self.artikelstamm.loc[artikel].iloc[0]
                # Je Artikelindex 2 Einträge, da 2 BewegungsbaumIDs
                warengruppennummer = artikel_data.Warengruppe
                warengruppen_index = self.warengruppen.index(warengruppennummer)
                warengruppen_state = to_categorical(warengruppen_index, num_classes=self.anz_wg)

                eigenmarke = artikel_data.Eigenmarke
                if eigenmarke == 0:
                    eigenmarke = -1
                eigenmarke = np.array([eigenmarke])

                gattungsmarke = artikel_data.GuG
                if gattungsmarke == 0:
                    gattungsmarke = -1
                gattungsmarke = np.array([gattungsmarke])

                einheit = artikel_data.Einheit
                einheit_state = to_categorical(einheit, num_classes=9)

                mhd = artikel_data.MHD
                if mhd < 8:
                    mhd_state = -1
                elif mhd < 31:
                    mhd_state = 0
                else:
                    mhd_state = 1
                mhd_state = np.array([mhd_state])

                ose = artikel_data.OSE
                ose = np.array([ose])

                try:
                    artikel_preis = preise.loc[artikel].copy()
                except KeyError:
                    artikel_preis = pd.DataFrame(
                        data={'Preis': [0], 'Datum': ["2016-01-01"]}
                        )
                if type(artikel_preis) == pd.core.frame.DataFrame:
                    artikel_preis = artikel_preis.set_index("Datum")

                elif type(artikel_preis) == pd.core.series.Series:
                    artikel_preis = pd.DataFrame(
                        data={'Preis': [artikel_preis.Preis]}, 
                        index=[artikel_preis.Datum]
                        )
                else:
                    raise AssertionError("Unknown Type for Price: {}".format(type(artikel_preis)))

                try:
                    aktionspreise = aktionen.loc[artikel].copy()
                except KeyError:
                    aktionspreise = None

                self.static_state_data[artikel] = {
                    "Warengruppe": warengruppen_state, 
                    "OrderLeadTime": olt, 
                    "Preis": artikel_preis,
                    "Aktionspreise": aktionspreise,
                    "Eigenmarke": eigenmarke,
                    "Gattungsmarke": gattungsmarke,
                    "Einheit": einheit_state,
                    "MHD": mhd_state,
                    "OSE": ose
                    }
            # endregion
            del preise

        if save_pickled:
            with open(filename, "wb") as file:
                pickle.dump(self.absatz_data, file)
                pickle.dump(timeline, file)
                pickle.dump(self.artikelstamm, file)
                pickle.dump(self.artikel, file)
                pickle.dump(self.wetter, file)
                pickle.dump(self.kalender_tage, file)
                pickle.dump(self.static_state_data, file)
                pickle.dump(self.markt_artikel, file)
        
        cal_cls = get_german_holiday_calendar('SL')
        self.feiertage = cal_cls().holidays(
            pd.Timestamp.fromtimestamp(self.start_tag*24*3600),
            pd.Timestamp.fromtimestamp(self.end_tag*24*3600) + pd.DateOffset(3)
            )
        
        self.maerkte = list(self.absatz_data.keys())

        self.time_series_lenght = time_series_lenght

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
        self.akt_prod_promotionen = None
        self.akt_prod_eigenmarke = None
        self.akt_prod_gattungsmarke = None
        self.akt_prod_einheit = None
        self.akt_prod_mhd = None
        self.akt_prod_markt = None
        self.time_series_state = None
        self.stat_theo_bestand = None
        self.stat_fakt_bestand = None
        self.aktueller_tag = pd.Timestamp.fromtimestamp(self.start_tag*24*3600)

    def create_new_state(self, wochentag, kalenderwoche, feiertage):
        try:
            preis = self.akt_prod_preis[
                self.akt_prod_preis.index <= self.aktueller_tag.date().__str__()
                ].iloc[-1].Preis
        except IndexError:
            # Wenn Preis erst unterjährig eingetragen, aber Simulation vollen Zeitraum für das Produkt durchgeht.
            preis = self.akt_prod_preis.iloc[0].Preis

        promotions = np.zeros(3)
        if self.akt_prod_promotionen is not None:
            if type(self.akt_prod_promotionen) == pd.core.series.Series:
                if self.akt_prod_promotionen.DatumAb.date() >= self.aktueller_tag.date():
                    if self.akt_prod_promotionen.DatumBis.date() <= self.aktueller_tag.date():
                        promotions = np.array([
                            self.akt_prod_promotionen.relRabat, 
                            self.akt_prod_promotionen.absRabat,
                            self.akt_prod_promotionen.vDauer
                            ])
            elif type(self.akt_prod_promotionen) == pd.core.frame.DataFrame:
                promotions_df = self.akt_prod_promotionen[
                    (self.akt_prod_promotionen.DatumAb.dt.date>=self.aktueller_tag.date()) & 
                    (self.akt_prod_promotionen.DatumBis.dt.date<=self.aktueller_tag.date())
                    ]
                if not promotions_df.empty:
                    promotions_df = promotions_df.iloc[0]
                    promotions = np.array([
                        promotions_df.relRabat, 
                        promotions_df.absRabat,
                        promotions_df.vDauer
                        ])
            else:
                raise TypeError('Unbekannter Promotionstyp')

        new_state = np.concatenate(
            [
                self.akt_prod_markt,
                np.array([self.akt_prod_bestand]), 
                wochentag, 
                kalenderwoche,
                feiertage,
                self.akt_prod_wg,
                self.akt_prod_eigenmarke,
                self.akt_prod_gattungsmarke,
                self.akt_prod_einheit,
                self.akt_prod_mhd,
                [preis],
                promotions,
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
        self.aktueller_tag = pd.Timestamp.fromtimestamp(self.start_tag*24*3600)
        self.vergangene_tage = 0

        if markt:
            assert markt in self.maerkte, "Simulation kennt diesen Markt nicht."
            self.aktueller_markt = markt
        else:
            self.aktueller_markt = np.random.choice(self.maerkte, 1)[0]

        if artikel:
            assert artikel in self.markt_artikel[self.aktueller_markt], 'Simulation kennt diesen Markt-Artikel nicht.'
            self.aktuelles_produkt = artikel
        else:
            self.aktuelles_produkt = np.random.choice(self.markt_artikel[self.aktueller_markt], 1)[0]

        self.akt_prod_bestand = self.anfangsbestand
        self.akt_prod_absatz = self.absatz_data[self.aktueller_markt][self.aktuelles_produkt]
        self.akt_prod_wg = self.static_state_data[self.aktuelles_produkt]["Warengruppe"]
        self.akt_prod_preis = self.static_state_data[self.aktuelles_produkt]["Preis"]
        self.akt_prod_promotionen = self.static_state_data[self.aktuelles_produkt]["Aktionspreise"]
        self.akt_prod_olt = self.static_state_data[self.aktuelles_produkt]["OrderLeadTime"]
        self.akt_prod_eigenmarke = self.static_state_data[self.aktuelles_produkt]["Eigenmarke"]
        self.akt_prod_gattungsmarke = self.static_state_data[self.aktuelles_produkt]["Gattungsmarke"]
        self.akt_prod_einheit = self.static_state_data[self.aktuelles_produkt]["Einheit"]
        self.akt_prod_mhd = self.static_state_data[self.aktuelles_produkt]["MHD"]

        if self.aktueller_markt == 5:
            self.akt_prod_markt = np.array([-1])
        else:
            self.akt_prod_markt = np.array([1])

        wochentag = self.aktueller_tag.dayofweek
        wochentag = to_categorical(wochentag, num_classes=7)

        kalenderwoche = self.aktueller_tag.weekofyear
        kalenderwoche = to_categorical(kalenderwoche, num_classes=54)

        feiertage = np.zeros(4, dtype=np.int)
        for i in range(1, 5):
            if self.aktueller_tag.date() + pd.DateOffset(i) in self.feiertage:
                feiertage[i-1] = 1

        new_state = self.create_new_state(wochentag, kalenderwoche, feiertage)
        
        self.time_series_state = deque(maxlen=self.time_series_lenght)
        for _ in range(self.time_series_lenght):
            self.time_series_state.append(new_state)

        return np.array(self.time_series_state), {"Artikel": self.aktuelles_produkt}

    def make_action(self, action):
        if self.fertig:
            raise AssertionError("Simulation für diesen Artikel fertig. Simulation zurücksetzen")

        absatz = self.akt_prod_absatz[self.vergangene_tage][0]

        self.aktueller_tag += pd.DateOffset(1)
        self.vergangene_tage += 1

        while True:
            if self.aktueller_tag.dayofweek == 6 or self.aktueller_tag.date() in self.feiertage: # Sonntag oder Feiertage
                self.aktueller_tag += pd.DateOffset(1)
                self.vergangene_tage += 1
            else:
                break
        
        wochentag = self.aktueller_tag.dayofweek
        wochentag = to_categorical(wochentag, num_classes=7)

        kalenderwoche = self.aktueller_tag.weekofyear
        kalenderwoche = to_categorical(kalenderwoche, num_classes=54)

        feiertage = np.zeros(4, dtype=np.int)
        for i in range(1, 5):
            if self.aktueller_tag.date() + pd.DateOffset(i) in self.feiertage:
                feiertage[i-1] = 1

        # TODO: OrderLeadTime durch eine action-Deque realisieren

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
        
        new_state = self.create_new_state(wochentag, kalenderwoche, feiertage)

        self.time_series_state.append(new_state)

        if self.vergangene_tage >= self.kalender_tage -1:
            # auch größer, falls letzter Tag ein Sonn/Feiertag ist und übersprungen wird.
            self.fertig = True

        return reward, self.fertig, np.array(self.time_series_state)


class StockSimulationV2(object):
    def __init__(self, **kwargs):
        self.data = Datapipeline(**kwargs)
        self.data.read_files()
        self.data.prepare_for_simulation(**kwargs)
        self.markt_artikel = self.data.absatz.groupby('Markt')['Artikel'].unique()
        self.data.absatz = self.data.absatz.set_index(['Markt', 'Artikel'])
        self.step_size = kwargs['StepSize']
        self.aktueller_markt = None
        self.aktueller_artikel = None
        self.artikel_absatz = None
        self.vergangene_tage = None
        self.static_state = None
        self.dynamic_state = deque(maxlen=self.step_size)
        self.dynamic_state_data = None
        self.tage = len(self.data.tage)
        self.bestand = None
        self.stat_theo_bestand = None
        self.stat_fakt_bestand = None

    @property
    def state(self):
        state = {
            'RegressionState': {'dynamic_state': np.array(self.dynamic_state), 'static_state': self.static_state},
            'AgentState': self.bestand
        }
        return state

    @property
    def info(self):
        return {'Artikel': self.aktueller_artikel, 'Markt': self.aktueller_markt}

    def reset(self, artikel=None, markt=None):
        if markt:
            assert markt in self.markt_artikel.index.values, 'Markt nicht bekannt'
            self.aktueller_markt = markt
        else:
            self.aktueller_markt = np.random.choice(self.markt_artikel.index.values)
        if artikel:
            assert artikel in self.markt_artikel.loc[self.aktueller_markt], \
                'Dieser Artikel wird in diesem Markt nicht geführt'
            self.aktueller_artikel = artikel
        else:
            self.aktueller_artikel = np.random.choice(self.markt_artikel.loc[self.aktueller_markt])

        dyn_state = self.data.absatz.loc[:, self.data.dyn_state_scalar_cols].to_numpy(dtype=np.int8)
        for category, class_numbers in self.data.dyn_state_category_cols.items():
            category_state = to_categorical(
                self.data.absatz.loc[:, category],
                num_classes=class_numbers).astype(np.int8)
            dyn_state = np.concatenate((dyn_state, category_state), axis=1)
        self.artikel_absatz = dyn_state[:, 0]
        self.dynamic_state_data = dyn_state[:, 1:]
        self.vergangene_tage = 0
        self.bestand = np.random.randint(10)
        self.stat_theo_bestand = []
        self.stat_fakt_bestand = []
        stat_state = self.data.artikelstamm.loc[
            self.aktueller_artikel,
            self.data.stat_state_scalar_cols].to_numpy(dtype=np.int8)
        for category, class_numbers in self.data.stat_state_category_cols.items():
            category_state = to_categorical(
                self.data.artikelstamm.loc[self.aktueller_artikel, category],
                num_classes=class_numbers).astype(np.int8)
            stat_state = np.concatenate((stat_state, category_state), axis=0)
        self.static_state = stat_state
        for i in range(self.step_size):
            self.dynamic_state.append(self.dynamic_state_data[self.vergangene_tage][1:])
        return self.state, self.info

    def make_action(self, action):
        absatz = self.artikel_absatz[self.vergangene_tage]
        self.vergangene_tage += 1
        self.dynamic_state.append(self.dynamic_state_data[self.vergangene_tage][1:])
        done = self.tage <= self.vergangene_tage + 1

        # Tagsüber Absatz abziehen und bewerten:
        self.bestand -= absatz
        self.stat_theo_bestand.append(self.bestand)

        if self.bestand >= 27.5:
            reward = 0.004992 - (self.bestand - 27.5) / 1000
        elif self.bestand >= 1:
            reward = np.exp((1 - self.bestand) / 5)
        else:
            reward = np.exp((self.bestand - 1) * 1.5) - 1
            # Nichtnegativität des Bestandes
            self.bestand = 0
        self.stat_fakt_bestand.append(self.bestand)

        # Nachmittag: Bestellung kommt an und wird verräumt
        self.bestand += action

        return reward, self.state, done


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
