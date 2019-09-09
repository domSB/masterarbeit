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
import tensorflow as tf
import datetime


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


def create_dataset(lab, dyn, stat, _params):
    def gen():
        while True:
            rand_idx = np.random.randint(0, lab.shape[0], 1)
            yield {'dynamic_input': dyn[rand_idx][0], 'static_input': stat[rand_idx][0]}, lab[rand_idx][0]

    _dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=({'dynamic_input': tf.float32, 'static_input': tf.int8}, tf.float32),
        output_shapes=(
            {'dynamic_input': tf.TensorShape([_params['time_steps'], _params['dynamic_state_shape']]),
             'static_input': tf.TensorShape([_params['static_state_shape']])},
            tf.TensorShape([5]))
    )
    _dataset = _dataset.batch(_params['batch_size'])
    _dataset = _dataset.repeat()
    _dataset = _dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return _dataset


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
        if os.path.exists(os.path.join('files', 'logging', 'Predictor', _params['Name'])):
            name = datetime.datetime.now().__str__()
        else:
            name = _params['Name']
        os.mkdir(os.path.join('files', 'logging', 'Predictor', name))
        os.mkdir(os.path.join('files', 'models', 'Predictor', name))
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('files', 'logging', 'Predictor', name),
            histogram_freq=1,
            batch_size=32,
            write_graph=True,
            write_grads=True,
            update_freq='batch')
        nan_callback = tf.keras.callbacks.TerminateOnNaN()
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join('files', 'models', 'Predictor', name, 'weights.{epoch:02d}-{loss:.2f}.hdf5'),
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
            # batch_size=512,
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
            validation_data=_val_dataset,
            validation_steps=_params['val_steps_per_epoch']
        )
        return history

    def predict(self, x):
        y = self.model.predict(x)
        return y


params = {
    'forecast_state': 5,
    'learning_rate': 0.001,
    'time_steps': None,
    'dynamic_state_shape': None,
    'static_state_shape': None,
    'epochs': 20,
    'batch_size': 512
}
DATA_PATH = os.path.join('./files/prepared')
val_l, val_d, val_s = load_numpy(os.path.join(DATA_PATH, 'Time-2018-01-01-2018-12-31-6.npz'))
l, d, s = load_numpy(os.path.join(DATA_PATH, 'Time-2016-01-01-2017-12-31-6.npz'))
params.update({
    'time_steps': d.shape[1],
    'steps_per_epoch': int(d.shape[0] / params['batch_size']),
    'val_steps_per_epoch': int(val_d.shape[0] / params['batch_size']),
    'dynamic_state_shape': d.shape[2],
    'static_state_shape': s.shape[1],
    'Name': 'FullRegTime'
})
print(params)
dataset = create_dataset(l, d, s, params)
val_dataset = create_dataset(val_l, val_d, val_s, params)
predictor = Predictor()
predictor.build_model(params)
hist = predictor.train(dataset, val_dataset, params)


