"""
Hier sollen die Inputdaten auf den Absatz regressiert werden, um die Vorhersagekraft zu bestimmen.

"""
import os
import numpy as np
import tensorflow as tf
from agents import Predictor
from data.access import DataPipeLine
from data.preparation import split_np_arrays
# [1, 12, 55, 80, 17, 77, 71, 6, 28]


def create_dataset(_lab, _dyn, _stat, _params):
    def gen():
        while True:
            rand_idx = np.random.randint(0, _lab.shape[0], 1)
            yield {'dynamic_input': _dyn[rand_idx][0], 'static_input': _stat[rand_idx][0]}, _lab[rand_idx][0]

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
regression_params = {
    'InputDirectory': os.path.join('files', 'raw'),
    'OutputDirectory': os.path.join('files', 'prepared'),
    'ZielWarengruppen': [17],
    'StatStateCategoricals': {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6},
}
pipeline = DataPipeLine(**regression_params)
lab, dyn, stat, split_helper = pipeline.get_regression_data()
train_data, test_data = split_np_arrays(lab, dyn, stat, split_helper)
params.update({
    'time_steps': dyn.shape[1],
    'steps_per_epoch': int(train_data[1].shape[0] / params['batch_size']),
    'val_steps_per_epoch': int(test_data[1].shape[0] / params['batch_size']),
    'dynamic_state_shape': dyn.shape[2],
    'static_state_shape': stat.shape[1],
    'Name': 'FullRegMarkt'
})
print(params)
dataset = create_dataset(*train_data, params)
val_dataset = create_dataset(*test_data, params)
predictor = Predictor()
predictor.build_model(params)
hist = predictor.train(dataset, val_dataset, params)


