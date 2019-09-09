"""
Hier sollen die Inputdaten auf den Absatz regressiert werden, um die Vorhersagekraft zu bestimmen.

ACHTUNG!!
Script läd komplette Absatzdaten inkl. Zusatzinfos zu Artikel in einen Numpy-Array.
Arbeitsspeicherverbrauch von 20+ GB !
"""
# TODO: Regression von State auf Absatz
import os
import numpy as np
import tensorflow as tf
from agents import Predictor


def load_numpy(path):
    print('INFO - Lese NPZ-Dateien')
    files = np.load(path)
    lab = files['lab']
    dyn = files['dyn']
    stat = files['stat']
    return lab, dyn, stat


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
val_l, val_d, val_s = load_numpy(os.path.join(DATA_PATH, 'Markt-2018-01-01-2018-12-31-6.npz'))
l, d, s = load_numpy(os.path.join(DATA_PATH, 'Markt-2017-01-01-2017-12-31-6.npz'))
params.update({
    'time_steps': d.shape[1],
    'steps_per_epoch': int(d.shape[0] / params['batch_size']),
    'val_steps_per_epoch': int(val_d.shape[0] / params['batch_size']),
    'dynamic_state_shape': d.shape[2],
    'static_state_shape': s.shape[1],
    'Name': 'FullRegMarkt'
})
print(params)
dataset = create_dataset(l, d, s, params)
val_dataset = create_dataset(val_l, val_d, val_s, params)
predictor = Predictor()
predictor.build_model(params)
hist = predictor.train(dataset, val_dataset, params)


