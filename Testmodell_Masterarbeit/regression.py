"""
Hier sollen die Inputdaten auf den Absatz regressiert werden, um die Vorhersagekraft zu bestimmen.

ACHTUNG!!
Script läd komplette Absatzdaten inkl. Zusatzinfos zu Artikel in einen Numpy-Array.
Arbeitsspeicherverbrauch von 10 GB pro Markt!
"""
# TODO: Regression von State auf Absatz
# import sys
# sys.path.extend(['/home/dominic/PycharmProjects/masterarbeit',
#                  '/home/dominic/PycharmProjects/masterarbeit/Testmodell_Masterarbeit'])
import os

import numpy as np
import pandas as pd
from keras.utils import to_categorical
import tensorflow as tf

from calender.german_holidays import get_german_holiday_calendar

DATA_PATH = os.path.join('data')


class DataPipeline(object):
    def __init__(
            self,
            series_type='time',
            data_path=os.path.join('data')
    ):
        self.series_type = series_type
        self.data_path = data_path
        self.dynamic_state = None
        self.static_state = None
        self.index_list = None
        self.detail_warengruppen_index = None
        self.warengruppen_index = None
        self.einheit_index = None
        self.tage = None
        self.time_series_index = None
        self.dyn_state_scalar_cols = ['Menge', 'MaxTemp_1D', 'MinTemp_1D', 'Wolken_1D',
                                      'Regen_1D', 'MaxTemp_2D', 'MinTemp_2D', 'Wolken_2D', 'Regen_2D',
                                      'Preis', 'relRabat', 'absRabat', 'vDauer']
        self.dyn_state_label_cols = ['in1', 'in2', 'in3', 'in4', 'in5']
        self.dyn_state_category_cols = {'Wochentag': 7, 'Kalenderwoche': 54}
        self.stat_state_scalar_cols = ['Eigenmarke', 'GuG', 'OSE', 'Saisonal', 'Kern', 'Bio', 'Glutenfrei',
                                       'Laktosefrei']
        self.stat_state_category_cols = {'MHDgroup': 7, 'Warengruppe': 9, 'Detailwarengruppe': None, 'Einheit': None}

    def prepare_data(self, start_date, end_date):
        # region Artikelstammm
        print('INFO - Lese Artikelstamm')
        artikelstamm = pd.read_csv(
            os.path.join(self.data_path, '0 ArtikelstammV4.csv'),
            header=0,
            names=['Artikel', 'Warengruppe', 'Detailwarengruppe', 'Bezeichnung',
                   'Eigenmarke', 'Einheit', 'Verkaufseinheit', 'MHD',
                   'GuG', 'OSE', 'OSEText', 'Saisonal',
                   'Kern', 'Bio', 'Glutenfrei',
                   'Laktosefrei', 'MarkeFK', 'Region']
            )
        warengruppen_maske = [1, 12, 55, 80, 17, 77, 71, 6, 28]
        artikelstamm = artikelstamm[artikelstamm.Warengruppe.isin(warengruppen_maske)]
        artikel_maske = pd.unique(artikelstamm.Artikel)

        # Fehlende Detailwarengruppen mit wahrscheinlich richtiger DetailWarengruppe füllen
        wg_group = artikelstamm.loc[:, ['Warengruppe', 'Detailwarengruppe']].groupby('Warengruppe').median()
        detail_warengruppen_nan_index = wg_group.to_dict()['Detailwarengruppe']
        artikelstamm['DetailwarengruppeBackup'] = artikelstamm['Warengruppe'].map(detail_warengruppen_nan_index)
        artikelstamm['Detailwarengruppe'].fillna(value=artikelstamm['DetailwarengruppeBackup'], inplace=True)
        artikelstamm.drop(columns=['DetailwarengruppeBackup'], inplace=True)

        # Restlaufzeit von Anzahl-Tage in kategoriale Gruppen sortieren
        mhd_labels = [0, 1, 2, 3, 4, 5, 6]
        mhd_bins = [0, 1, 7, 14, 28, 100, 1000, 100000]
        artikelstamm['MHDgroup'] = pd.cut(artikelstamm.MHD, mhd_bins, right=False, labels=mhd_labels)
        self.detail_warengruppen_index = {
            value: index for index, value in enumerate(np.sort(pd.unique(artikelstamm.Detailwarengruppe)))
        }
        self.stat_state_category_cols['Detailwarengruppe'] = len(self.detail_warengruppen_index)
        self.warengruppen_index = {
            value: index for index, value in enumerate(np.sort(pd.unique(artikelstamm.Warengruppe)))
        }
        self.einheit_index = {
            value: index for index, value in enumerate(np.sort(pd.unique(artikelstamm.Einheit)))
        }
        self.stat_state_category_cols['Einheit'] = len(self.einheit_index)
        artikelstamm['Detailwarengruppe'] = artikelstamm['Detailwarengruppe'].map(self.detail_warengruppen_index)
        artikelstamm['Warengruppe'] = artikelstamm['Warengruppe'].map(self.warengruppen_index)
        artikelstamm['Einheit'] = artikelstamm['Einheit'].map(self.einheit_index)
        artikelstamm.drop(columns=['MHD', 'Region', 'MarkeFK', 'Verkaufseinheit', 'OSEText'], inplace=True)
        artikelstamm.set_index('Artikel', inplace=True)
        # endregion

        # region Bewegungsdaten
        print('INFO - Lese Bewegungsdaten')
        warenausgang = pd.read_csv(
            os.path.join(self.data_path, '0 Warenausgang.time.csv'),
            header=0,
            names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
            )

        warenausgang = warenausgang[warenausgang.Artikel.isin(artikel_maske)]

        warenausgang['Datum'] = pd.to_datetime(warenausgang['Datum'], format='%d.%m.%y')
        warenausgang['Belegtyp'] = warenausgang['Belegtyp'].astype('category')
        warenausgang = warenausgang.loc[
            (warenausgang.Belegtyp == 'UMSATZ_SCANNING') | (warenausgang.Belegtyp == 'UMSATZ_AKTION')
            ].copy()
        warenausgang = warenausgang.groupby(['Markt', 'Datum', 'Artikel'],  as_index=False).sum()

        cal_cls = get_german_holiday_calendar('SL')
        cal = cal_cls()
        sl_bd = pd.tseries.offsets.CustomBusinessDay(calendar=cal, weekmask='Mon Tue Wed Tue Fri Sat')
        # Testzeitraum ist nur von 1.Jan 2018 bis 31.12.2019, der Zeitraum muss aber um 5 Tage verlängert werden, damit
        zeitraum = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.DateOffset(7), freq=sl_bd)
        self.tage = zeitraum
        warenausgang.set_index('Datum', inplace=True)
        warenausgang = warenausgang.groupby(['Markt', 'Artikel']).apply(lambda x: x.reindex(zeitraum, fill_value=0))
        warenausgang.drop(columns=['Markt', 'Artikel'], inplace=True)
        warenausgang.reset_index(inplace=True)
        warenausgang.rename(columns={'level_2': 'Datum'}, inplace=True)

        warenausgang['Wochentag'] = warenausgang.Datum.dt.dayofweek
        warenausgang['Kalenderwoche'] = warenausgang.Datum.dt.weekofyear
        warenausgang["UNIXDatum"] = warenausgang["Datum"].astype(np.int64)/(1000000000 * 24 * 3600)
        # endregion

        # region Wetter
        print('INFO - Lese Wetter')
        wetter = pd.read_csv(
            os.path.join(self.data_path, '1 Wetter.csv')
            )
        wetter["date_shifted_oneday"] = wetter["date"] - 1
        wetter["date_shifted_twodays"] = wetter["date"] - 2
        wetter.drop(columns=['Unnamed: 0'], inplace=True)
        warenausgang = pd.merge(
            warenausgang,
            wetter,
            left_on='UNIXDatum',
            right_on='date_shifted_oneday'
        )
        warenausgang = pd.merge(
            warenausgang,
            wetter,
            left_on='UNIXDatum',
            right_on='date_shifted_twodays',
            suffixes=('_1D', '_2D')
        )
        warenausgang.drop(
            columns=["date_shifted_oneday_1D",
                     "date_shifted_twodays_1D",
                     "date_shifted_oneday_2D",
                     "date_shifted_twodays_2D",
                     "date_1D",
                     "date_2D"
                     ],
            inplace=True
        )
        # endregion

        # region Preise
        print('INFO - Lese Preise')
        preise = pd.read_csv(
            os.path.join(self.data_path, '1 Preise.csv')
            )
        preise.drop(columns=['Unnamed: 0'], inplace=True)
        preise['Datum'] = pd.to_datetime(preise['Datum'], format='%Y-%m-%d')
        preise.drop_duplicates(inplace=True)
        preise = preise.sort_values(by=['Datum', 'Artikel'])
        preise['Next'] = preise.groupby(['Artikel'], as_index=True)['Datum'].shift(-1)
        preise = preise.where(~preise.isna(), pd.to_datetime(end_date) + pd.DateOffset(7))
        # preise.set_index('Artikel', inplace=True)
        warenausgang.reset_index(inplace=True)
        warenausgang = pd.merge_asof(
            warenausgang,
            preise.loc[:, ["Preis", "Artikel", "Datum"]],
            left_on='Datum',
            right_on='Datum',
            by='Artikel'
        )
        # endregion

        # region Aktionspreise hinzufügen
        print('INFO - Lese Aktionspreise')
        aktionspreise = pd.read_csv(
            os.path.join(self.data_path, '1 Preisaktionen.csv'),
            decimal=','
            )
        aktionspreise.drop(columns=['Unnamed: 0'], inplace=True)
        aktionspreise['DatumAb'] = pd.to_datetime(aktionspreise['DatumAb'], format='%Y-%m-%d')
        aktionspreise['DatumBis'] = pd.to_datetime(aktionspreise['DatumBis'], format='%Y-%m-%d')
        aktionspreise.drop_duplicates(inplace=True)
        aktionspreise = aktionspreise.sort_values(by=['DatumAb', 'DatumBis', 'Artikel'])

        warenausgang = pd.merge_asof(warenausgang, aktionspreise, left_on='Datum', right_on='DatumAb', by='Artikel')
        warenausgang.relRabat.fillna(0., inplace=True)
        warenausgang.absRabat.fillna(0., inplace=True)
        warenausgang.vDauer.fillna(0, inplace=True)
        warenausgang.relRabat = warenausgang.relRabat.astype(np.float64)
        warenausgang.absRabat = warenausgang.absRabat.astype(np.float64)
        warenausgang.vDauer = warenausgang.vDauer.astype(np.int)
        warenausgang.drop(columns=['DatumAb', 'DatumBis'], inplace=True)
        # endregion

        # region Shift Targets
        print('INFO - Erzeuge Targets')
        warenausgang['in1'] = warenausgang.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-1)
        warenausgang['in2'] = warenausgang.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-2)
        warenausgang['in3'] = warenausgang.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-3)
        warenausgang['in4'] = warenausgang.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-4)
        warenausgang['in5'] = warenausgang.groupby(['Markt', 'Artikel'], as_index=False)['Menge'].shift(-5)
        # endregion

        warenausgang.dropna(axis=0, inplace=True)
        warenausgang.set_index(['Markt', 'Artikel', 'UNIXDatum'], inplace=True, drop=False)
        warenausgang.sort_index(inplace=True)
        self.index_list = list(zip(warenausgang.Markt, warenausgang.Artikel, warenausgang.UNIXDatum))
        self.dynamic_state = warenausgang
        self.static_state = artikelstamm

    def store_data(self):
        # region Store Data
        print('INFO - Speichere Daten')
        store = pd.HDFStore(os.path.join(self.data_path, self.series_type + '.h5'))
        store.put('dynamic_state', self.dynamic_state)
        store.put('static_state', self.static_state, format='table')
        store.close()
        print('INFO - Fertig')
        # endregion

    def load_data(self):
        print('INFO - Reading Data')
        store = pd.HDFStore(os.path.join(self.data_path, self.series_type + '.h5'))
        self.dynamic_state = store.get('dynamic_state')
        self.static_state = store.get('static_state')
        store.close()

    def create_dataset(self, batch_size, step_length):
        zeiten = np.array([x.timestamp() / (24 * 3600)for x in self.tage])
        zeiten = np.append(np.repeat(zeiten[0], step_length), zeiten)
        self.time_series_index = {zeiten[k]: zeiten[k - step_length:k] for k in range(len(zeiten))}

        def gen():
            for index in self.index_list:
                idx = [(index[0], index[1], day) for day in self.time_series_index[index[2]]]
                art_idx = index[1]
                dyn_state = self.dynamic_state.loc[idx, self.dyn_state_scalar_cols].to_numpy()
                for category, class_numbers in self.dyn_state_category_cols.items():
                    category_state = to_categorical(self.dynamic_state.loc[idx, category], num_classes=class_numbers)
                    dyn_state = np.concatenate((dyn_state, category_state), axis=1)

                stat_state = self.static_state.loc[art_idx, self.stat_state_scalar_cols].to_numpy()
                for category, class_numbers in self.stat_state_category_cols.items():
                    category_state = to_categorical(self.static_state.loc[art_idx, category], num_classes=class_numbers)
                    stat_state = np.append(stat_state, category_state)

                labels = self.dynamic_state.loc[index, self.dyn_state_label_cols].to_numpy()

                yield {'dynamic_input': dyn_state, 'static_input': stat_state}, labels

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_types=({'dynamic_input': tf.float64, 'static_input': tf.float64}, tf.float64),
            output_shapes=(
            {'dynamic_input': tf.TensorShape([step_length, 74]), 'static_input': tf.TensorShape([490])}, tf.TensorShape([5]))
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset


class Predictor(object):
    def __init__(self):
        self.state_shape = 91
        self.target_shape = 5
        self.learning_rate = 0.01
        self.epochs = 30
        self.lr_decay = 0.01 / self.epochs
        inputs = tf.keras.Input(shape=(self.state_shape,))
        x = tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="Dense_1"
        )(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="Dense_2"
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="Dense_3"
        )(x)
        predictions = tf.keras.layers.Dense(self.target_shape, activation='relu', name="Predictions")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=predictions)
        adam = tf.keras.optimizers.RMSprop(lr=self.learning_rate)
        self.model.compile(optimizer=adam, loss='mean_squared_error', metrics=["mean_absolute_error", "mean_squared_error"])

    def train(self, x, y):
        history = self.model.fit(x, y, batch_size=512, epochs=self.epochs, validation_split=0.2)
        return history

    def predict(self, x):
        y = self.model.predict(x)
        return y


pipeline = DataPipeline()
pipeline.prepare_data('2018-01-01', '2018-12-31')
# pipeline.store_data()
dataset = pipeline.create_dataset(32, 5)
iterator = dataset.make_one_shot_iterator()
el = iterator.get_next()
with tf.Session() as sess:
    for i in range(10):
        state, lab = sess.run(el)
        print(state['dynamic_input'].shape, state['static_input'].shape, lab.shape)
# TODO: create new Price-Tables from preise.markt and preise.time
