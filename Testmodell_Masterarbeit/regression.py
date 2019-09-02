"""
Hier sollen die Inputdaten auf den Absatz regressiert werden, um die Vorhersagekraft zu bestimmen.

ACHTUNG!!
Script läd komplette Absatzdaten inkl. Zusatzinfos zu Artikel in einen Numpy-Array.
Arbeitsspeicherverbrauch von 10 GB pro Markt!
"""
# TODO: Regression von State auf Absatz
import os
import time

import numpy as np
import pandas as pd
from keras.utils import to_categorical
import tensorflow as tf

from calender import get_german_holiday_calendar

DATA_PATH = os.path.join('data')


def prepare_data():
    # region Artikelstammm
    print('INFO - Lese Artikelstamm')
    artikelstamm = pd.read_csv(
        os.path.join(DATA_PATH, '0 ArtikelstammV4.csv'),
        header=0,
        names=['Artikel', 'Warengruppe', 'Detailwarengruppe', 'Bezeichnung',
           'Eigenmarke', 'Einheit', 'Verkaufseinheit', 'MHD',
           'GuG', 'OSE', 'OSEText', 'Saisonal',
           'Kern', 'Bio', 'Glutenfrei',
           'Laktosefrei', 'MarkeFK', 'Region'],
        memory_map=True
        )
    warengruppen_maske = [1, 12, 55, 80, 17, 77, 71, 6, 28]
    artikelstamm = artikelstamm[artikelstamm.Warengruppe.isin(warengruppen_maske)]
    artikel_maske = pd.unique(artikelstamm.Artikel)
    # endregion

    # region Bewegungsdaten
    print('INFO - Lese Bewegungsdaten')
    warenausgang = pd.read_csv(
        os.path.join(DATA_PATH, '0 Warenausgang.time.csv'),
        header=0,
        names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
        )

    warenausgang = warenausgang[warenausgang.Artikel.isin(artikel_maske)]

    warenausgang['Datum'] = pd.to_datetime(warenausgang['Datum'], format='%d.%m.%y')
    warenausgang['Belegtyp'] = warenausgang['Belegtyp'].astype('category')
    warenausgang.drop(columns=['Markt'], inplace=True)
    warenausgang = warenausgang.loc[
        (warenausgang.Belegtyp == 'UMSATZ_SCANNING') | (warenausgang.Belegtyp == 'UMSATZ_AKTION')
        ].copy()
    warenausgang = warenausgang.groupby(["Datum", "Artikel"],  as_index=False).sum()

    cal_cls = get_german_holiday_calendar('SL')
    cal = cal_cls()
    sl_bd = pd.tseries.offsets.CustomBusinessDay(calendar=cal, weekmask='Mon Tue Wed Tue Fri Sat')
    zeitraum = pd.date_range('2018-01-01', '2018-12-31', freq=sl_bd)

    warenausgang.set_index('Datum', inplace=True)
    warenausgang = warenausgang.groupby('Artikel').apply(lambda x: x.reindex(zeitraum, fill_value=0))
    warenausgang.drop(columns=['Artikel'], inplace=True)
    warenausgang.reset_index(inplace=True)
    warenausgang.rename(columns={'level_1': 'Datum'}, inplace=True)

    warenausgang['Wochentag'] = warenausgang.Datum.dt.dayofweek
    warenausgang['Kalenderwoche'] = warenausgang.Datum.dt.weekofyear
    warenausgang["UNIXDatum"] = warenausgang["Datum"].astype(np.int64)/(1000000000 * 24 * 3600)

    mhd_labels = [0, 1, 2, 3, 4, 5, 6]
    mhd_bins = [0, 1, 7, 14, 28, 100, 1000, 100000]
    artikelstamm['MHDgroup'] = pd.cut(artikelstamm.MHD, mhd_bins, right=False, labels=mhd_labels)
    warenausgang = pd.merge(warenausgang, artikelstamm.loc[:,["Artikel", "Warengruppe", "Einheit", "Eigenmarke", "GuG", "MHDgroup"]], how='left', on='Artikel', validate='many_to_one')
    # endregion

    # region Wetter
    print('INFO - Lese Wetter')
    wetter = pd.read_csv(
        os.path.join(DATA_PATH, '1 Wetter.csv'),
        memory_map=True
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
                 "date_shifted_twodays_2D"
                 ],
        inplace=True
    )
    # endregion

    # region Preise
    print('INFO - Lese Preise')
    preise = pd.read_csv(
        os.path.join(DATA_PATH, '1 Preise.csv'),
        memory_map=True
        )
    preise.drop(columns=['Unnamed: 0'], inplace=True)
    preise['Datum'] = pd.to_datetime(preise['Datum'], format='%Y-%m-%d')
    preise.drop_duplicates(inplace=True)
    preise = preise.sort_values(by=['Datum', 'Artikel'])
    preise['Next'] = preise.groupby(['Artikel'], as_index=True)['Datum'].shift(-1)
    preise = preise.where(~preise.isna(), pd.to_datetime('2019-07-01'))
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
        os.path.join(DATA_PATH, '1 Preisaktionen.csv'),
        decimal =',',
        memory_map=True
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
    warenausgang.head(1).T.squeeze()
    # endregion

    # region Shift Targets
    print('INFO - Erzeuge Targets')
    warenausgang['in1'] = warenausgang.groupby(['Artikel'], as_index=False)['Menge'].shift(-1)
    warenausgang.in1.fillna(0., inplace=True)
    warenausgang['in2'] = warenausgang.groupby(['Artikel'], as_index=False)['Menge'].shift(-2)
    warenausgang.in2.fillna(0., inplace=True)
    warenausgang['in3'] = warenausgang.groupby(['Artikel'], as_index=False)['Menge'].shift(-3)
    warenausgang.in3.fillna(0., inplace=True)
    warenausgang['in4'] = warenausgang.groupby(['Artikel'], as_index=False)['Menge'].shift(-4)
    warenausgang.in4.fillna(0., inplace=True)
    warenausgang['in5'] = warenausgang.groupby(['Artikel'], as_index=False)['Menge'].shift(-5)
    warenausgang.in5.fillna(0., inplace=True)
    # endregion

    # region Store Data
    print('INFO - Speichere Daten')
    store = pd.HDFStore('./data/time.h5')
    store.put('warenausgang', warenausgang, format='table')
    store.close()
    print('INFO - Fertig')
    # endregion


""" 
Hier mit fertigen Daten weiterarbeiten. 
"""


def get_numpy_from_file():
    print('INFO - Reading Data')
    store = pd.HDFStore('./data/time.h5')
    dataframe = store.get('warenausgang')
    store.close()
    print('INFO - Getting Calendar')
    cal_cls = get_german_holiday_calendar('SL')
    feiertage = cal_cls().holidays(
        min(dataframe.Datum),
        max(dataframe.Datum) + pd.DateOffset(4)
    )
    warengruppen_index = {
                            1: 0,
                            12: 1,
                            55: 2,
                            80: 3,
                            17: 4,
                            77: 5,
                            71: 6,
                            6: 7,
                            28: 8}
    einheiten_index = {
                          0: 0,
                          2: 1,
                          4: 2,
                          1: 3,
                          3: 4,
                          7: 5
    }

    start = time.time()
    print('INFO - Mapping')
    dataframe['Warengruppe'] = dataframe['Warengruppe'].map(warengruppen_index)
    dataframe['Einheit'] = dataframe['Einheit'].map(einheiten_index)
    # Damit der Dataframe in den Arbeitsspeicher passt
    dataframe['MaxTemp_1D'] = dataframe['MaxTemp_1D'].astype(np.float16)
    dataframe['MinTemp_1D'] = dataframe['MinTemp_1D'].astype(np.float16)
    dataframe['Wolken_1D'] = dataframe['Wolken_1D'].astype(np.float16)
    dataframe['MaxTemp_2D'] = dataframe['MaxTemp_2D'].astype(np.float16)
    dataframe['MinTemp_2D'] = dataframe['MinTemp_2D'].astype(np.float16)
    dataframe['Wolken_2D'] = dataframe['Wolken_2D'].astype(np.float16)
    dataframe['Regen_2D'] = dataframe['Regen_2D'].astype(np.float16)
    dataframe['Preis'] = dataframe['Preis'].astype(np.float16)
    dataframe['relRabat'] = dataframe['relRabat'].astype(np.float16)
    dataframe['absRabat'] = dataframe['absRabat'].astype(np.float16)

    print('INFO - Getting Column Subset')
    ints = dataframe.loc[:, [
                                    'Artikel',
                                    'Eigenmarke',
                                    'GuG',
                                    'MHDgroup',
                                    'MaxTemp_1D',
                                    'MinTemp_1D',
                                    'Wolken_1D',
                                    'Regen_1D',
                                    'MaxTemp_2D',
                                    'MinTemp_2D',
                                    'Wolken_2D',
                                    'Regen_2D',
                                    'Preis',
                                    'relRabat',
                                    'absRabat',
                                    'vDauer',
                                    'in1',
                                    'in2',
                                    'in3',
                                    'in4',
                                    'in5'
                                 ]
           ].to_numpy()
    print('INFO - Categorical Data')
    weekday = to_categorical(dataframe.Wochentag, num_classes=7).astype(np.int8)
    kalenderwoche = to_categorical(dataframe.Kalenderwoche, num_classes=54).astype(np.int8)
    warengruppe = to_categorical(
        dataframe.Warengruppe,
        num_classes=9
    ).astype(np.int8)
    einheit = to_categorical(dataframe.Einheit, num_classes=6).astype(np.int8)
    dataframe = np.concatenate((ints, weekday, kalenderwoche, warengruppe, einheit), axis=1)

    end = time.time()
    duration = end - start
    print('Time :', duration)
    return dataframe


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


# prepare_data()
# frame = get_numpy_from_file()
# np.save(os.path.join(DATA_PATH, 'regression.npy'), frame)

print('INFO - Loading Numpy Array')
frame = np.load(os.path.join(DATA_PATH, 'regression.npy'), allow_pickle=True)
frame = frame.astype(np.float32)
# drop NaNs, due to missing prices in the Price Table.
# TODO: create new Price-Tables from preise.markt and preise.time
mask = np.any(np.isnan(frame), axis=1)
frame = frame[~mask]
target_index = np.array([16, 17, 18, 19, 20])
input_index = np.delete(np.arange(1, frame.shape[1]), target_index)  # Artikelnummer mit Index 1 wird fallen gelassen
y_train = frame[:, target_index]
x_train = frame[:, input_index]
predictor = Predictor()
history = predictor.train(x_train, y_train)

hist = pd.DataFrame(history.history)
