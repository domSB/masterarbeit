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


def extend_list(list_of_str, length):
    list_copy = list_of_str.copy()
    for i in range(1, length):
        list_of_str.extend([name + '_' + str(i) for name in list_copy])
    return list_of_str


def concat(df, length):
    cols = list(df.columns)
    cols = extend_list(cols, length)
    df_s = df.copy()
    for i in range(1, length):
        df = pd.concat((df, df_s.shift(-i)), axis=1)
    df.columns = cols
    print(df.shape)
    df.dropna(axis=0, inplace=True)
    print(df.shape)
    for i in range(1, length):
        df = df[df['Artikel'] == df['Artikel_' + str(i)]]
    x_cols = ['Menge', 'MaxTemp_1D', 'MinTemp_1D', 'Wolken_1D', 'Regen_1D',
              'MaxTemp_2D', 'MinTemp_2D', 'Wolken_2D', 'Regen_2D', 'Preis', 'relRabat', 'absRabat', 'vDauer']
    x_cols = extend_list(x_cols, length)
    weekday_col = ['Wochentag']
    weekday_col = extend_list(weekday_col, length)
    yearweek_col = ['Kalenderwoche']
    yearweek_col = extend_list(yearweek_col, length)
    y_cols = ['in1', 'in2', 'in3', 'in4', 'in5']
    x_arr = df[x_cols].to_numpy(dtype=np.float32).reshape(-1, length, int(len(x_cols) / length))
    weekday_arr = df[weekday_col].to_numpy(dtype=np.float32).reshape(-1, length, 1)
    weekday_arr = to_categorical(weekday_arr, num_classes=7)
    yearweek_arr = df[yearweek_col].to_numpy(dtype=np.float32).reshape(-1, length, 1)
    yearweek_arr = to_categorical(yearweek_arr, num_classes=54)
    big_x_arr = np.concatenate((x_arr, weekday_arr, yearweek_arr), axis=2)
    y_arr = df[y_cols].to_numpy(dtype=np.float32)
    stat_df = df['Artikel']
    return y_arr, big_x_arr, stat_df


class DataPipeline(object):
    def __init__(
            self,
            series_type='time',
            data_path=os.path.join('data')
    ):
        self.series_type = series_type
        self.data_path = data_path
        self.start_date = None
        self.end_date = None
        self.dynamic_state = None
        self.static_state = None
        self.index_list = None
        self.detail_warengruppen_index = None
        self.warengruppen_index = None
        self.einheit_index = None
        self.tage = None
        self.time_series_index = None
        self.mae = None
        self.mse = None
        self.dyn_state_scalar_cols = ['Menge', 'MaxTemp_1D', 'MinTemp_1D', 'Wolken_1D',
                                      'Regen_1D', 'MaxTemp_2D', 'MinTemp_2D', 'Wolken_2D', 'Regen_2D',
                                      'Preis', 'relRabat', 'absRabat', 'vDauer']
        self.dyn_state_label_cols = ['in1', 'in2', 'in3', 'in4', 'in5']
        self.dyn_state_category_cols = {'Wochentag': 7, 'Kalenderwoche': 54}
        self.stat_state_scalar_cols = ['Eigenmarke', 'GuG', 'OSE', 'Saisonal', 'Kern', 'Bio', 'Glutenfrei',
                                       'Laktosefrei']
        self.stat_state_category_cols = {'MHDgroup': 7, 'Warengruppe': 9, 'Detailwarengruppe': None, 'Einheit': None}

    def prepare_data(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
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
        artikelstamm['OSE'].fillna(0, inplace=True)
        artikelstamm['Saisonal'].fillna(0, inplace=True)
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
        warenausgang.Preis.fillna(method='bfill', inplace=True)
        warenausgang.Preis.fillna(method='ffill', inplace=True)
        warenausgang.dropna(axis=0, inplace=True)
        warenausgang.set_index(['Markt', 'Artikel', 'UNIXDatum'], inplace=True, drop=False)
        warenausgang.sort_index(inplace=True)
        self.index_list = list(zip(warenausgang.Markt, warenausgang.Artikel, warenausgang.UNIXDatum))
        self.dynamic_state = warenausgang
        self.static_state = artikelstamm

    def get_baseline(self, prediction_days):
        """
        Prints the Mean Squared Error and the Mean Absolute Error of the dynamic_state
        :param prediction_days:
        :return:
        """
        bewegung = self.dynamic_state.loc[:, ['Markt', 'Artikel', 'Datum', 'Menge', 'UNIXDatum']].copy()
        bewegung.reset_index(inplace=True, drop=True)
        bewegung['Prediction'] = bewegung.groupby(['Markt', 'Artikel'])['Menge'].shift(1)
        bewegung['AError'] = np.abs(bewegung['Menge'] - bewegung['Prediction'])
        bewegung['SError'] = np.square(bewegung['AError'])
        bewegung.dropna(inplace=True)
        bewegung['MAE'] = bewegung['AError'].rolling(prediction_days).mean()
        bewegung['MSE'] = bewegung['SError'].rolling(prediction_days).mean()
        self.mae = np.mean(bewegung['MAE'])
        self.mse = np.mean(bewegung['MSE'])
        print('BASELINE\n---\nMean Average Error: {mae} \nMean Squared Error: {mse}'.format(
            mae=self.mae,
            mse=self.mse
        ))

    def create_numpy(self, step_length):
        self.dynamic_state.reset_index(inplace=True, drop=True)
        self.dynamic_state.drop(columns=['index', 'Datum'], inplace=True)
        print('INFO - Concatenating dynamic states')
        y, x, stat_df = concat(self.dynamic_state, step_length)
        print('INFO - Reindexing static state')
        stat_df = self.static_state.reindex(stat_df)
        print('INFO - Creating categorical states')
        stat_state = stat_df.loc[:, self.stat_state_scalar_cols].to_numpy(dtype=np.int8)
        for category, class_numbers in self.stat_state_category_cols.items():
            category_state = to_categorical(stat_df.loc[:, category], num_classes=class_numbers).astype(np.int8)
            stat_state = np.concatenate((stat_state, category_state), axis=1)
        return y, x, stat_state

    def save_numpy(self, step_length):
        print('INFO - Speichere NPZ-Dateien')
        lab, dyn, stat = self.create_numpy(step_length)
        filename = '-'.join([self.series_type, self.start_date, self.end_date, str(step_length)])
        path = os.path.join(self.data_path, filename)
        np.savez(path, lab=lab, dyn=dyn, stat=stat)


def load_numpy(path):
    print('INFO - Lese NPZ-Dateien')
    files = np.load(path)
    lab = files['lab']
    dyn = files['dyn']
    stat = files['stat']
    return lab, dyn, stat


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif 3 <= epoch < 10:
        return 1e-4
    else:
        return 1e-5


def create_dataset(lab, dyn, stat, batch_size):
    step_length = dyn.shape[1]
    _steps_per_epoch = int(dyn.shape[0]/batch_size)

    def gen():
        while True:
            rand_idx = np.random.randint(0, lab.shape[0], 1)
            yield {'dynamic_input': dyn[rand_idx][0], 'static_input': stat[rand_idx][0]}, lab[rand_idx][0]

    _dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=({'dynamic_input': tf.float32, 'static_input': tf.int8}, tf.float32),
        output_shapes=(
            {'dynamic_input': tf.TensorShape([step_length, 74]), 'static_input': tf.TensorShape([490])},
            tf.TensorShape([5]))
    )
    _dataset = _dataset.batch(batch_size)
    _dataset = _dataset.repeat()
    _dataset = _dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return _dataset, _steps_per_epoch


class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(
               epoch + 1, tf.keras.backend.get_value(predictor.model.optimizer.lr)))


class Predictor(object):
    def __init__(self):
        self.model = None

    def build_model(self, _params):
        dynamic_inputs = tf.keras.Input(shape=(_params['time_steps'], _params['dynamic_state_shape']),
                                        name='dynamic_input')
        static_inputs = tf.keras.Input(shape=(_params['static_state_shape'],), name='static_input')
        dynamic_x = tf.keras.layers.LSTM(
            64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="LSTM_1",
            return_sequences=True
        )(dynamic_inputs)
        dynamic_x = tf.keras.layers.LSTM(
            64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="LSTM_2"
        )(dynamic_x)
        x = tf.keras.layers.concatenate([dynamic_x, static_inputs])
        x = tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="Dense_1"
        )(x)
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
        predictions = tf.keras.layers.Dense(_params['forecast_state'], activation='relu', name="predictions")(x)
        self.model = tf.keras.Model(inputs=[dynamic_inputs, static_inputs], outputs=predictions)
        rms = tf.keras.optimizers.RMSprop(lr=_params['learning_rate'])
        self.model.compile(
            optimizer=rms,
            loss='mean_squared_error',
            metrics=['mean_squared_error', 'mean_absolute_error']
        )

    def train(self, _dataset, _val_dataset, _params):
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='./logs/Reg2baselineVal',
            histogram_freq=0,
            batch_size=32,
            write_graph=True,
            write_grads=True,
            update_freq='batch')
        nan_callback = tf.keras.callbacks.TerminateOnNaN()
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            './model/Reg2baselineVal/weights.{epoch:02d}-{loss:.2f}.hdf5',
            monitor='loss',
            verbose=0,
            period=1)
        lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(decay)
        lr_print_callback = PrintLR()
        stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0,
            patience=3,
            verbose=0,
            restore_best_weights=True
        )
        history = self.model.fit(
            _dataset,
            batch_size=512,
            callbacks=[
                tb_callback,
                nan_callback,
                save_callback,
                lr_schedule_callback,
                lr_print_callback,
                stop_callback
            ],
            steps_per_epoch=_params['steps_per_epoch'],
            epochs=_params['epochs'],
            validation_data=_val_dataset
        )
        return history

    def predict(self, x):
        y = self.model.predict(x)
        return y


params = {
    'forecast_state': 5,
    'learning_rate': 0.001,
    'time_steps': 6,
    'dynamic_state_shape': 74,
    'static_state_shape': 490,
    'epochs': 20,
    'batch_size': 512
}
# TODO: create new Price-Tables from preise.markt and preise.time
# TODO: Vorhersage-Vortag als Baseline für Vergleichs-MSE nehmen
# pipeline = DataPipeline()
# pipeline.prepare_data('2017-01-01', '2017-12-31')
# pipeline.get_baseline(params['forecast_state'])
# pipeline.save_numpy(6)
val_l, val_d, val_s = load_numpy(os.path.join(DATA_PATH, 'time-2018-01-01-2018-12-31-6.npz'))
l, d, s = load_numpy(os.path.join(DATA_PATH, 'time-2017-01-01-2017-12-31-6.npz'))
dataset, steps_per_epoch = create_dataset(l, d, s, params['batch_size'])
val_dataset, _ = create_dataset(val_l, val_d, val_s, params['batch_size'])
params.update({'steps_per_epoch': steps_per_epoch})
predictor = Predictor()
predictor.build_model(params)
hist = predictor.train(dataset, val_dataset, params)
