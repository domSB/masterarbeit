import os

import matplotlib.pyplot as plt
import tensorflow as tf

from agents import Predictor, A3CNetwork
from agents.evaluation import Evaluator
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from simulation import StockSimulation
from utils import Hyperparameter

plt.style.use('ggplot')
# region Hyperparameter

hps = Hyperparameter()
hps.load(os.path.join('files', 'logging', 'A3C', '7eval17', 'Hyperparameter.yaml'))

predictor_dir = os.path.join('files', 'models', 'PredictorV2', '02RegWG' + str(hps.warengruppe[0]))
available_weights = os.listdir(predictor_dir)
available_weights.sort()
predictor_path = os.path.join(predictor_dir, available_weights[-1])
# endregion

pipeline = DataPipeLine(ZielWarengruppen=hps.warengruppe, DetailWarengruppe=hps.detail_warengruppe)
simulation_data = pipeline.get_regression_data()

train_data, test_data = split_np_arrays(*simulation_data)

predictor = Predictor()
predictor.build_model(
    dynamic_state_shape=simulation_data[1].shape[2],
    static_state_shape=simulation_data[2].shape[1]
)
predictor.load_from_weights(predictor_path)

print('Predicting ', end='')
train_pred = predictor.predict(
    {
        'dynamic_input': train_data[1],
        'static_input': train_data[2]
    }
)
print('and done ;)')
print('Predicting ', end='')
test_pred = predictor.predict(
    {
        'dynamic_input': test_data[1],
        'static_input': test_data[2]
    }
)
print('and done ;)')
# endregion

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    master_network = A3CNetwork('global', None, hps)
    saver = tf.train.Saver(max_to_keep=1, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global'))

with tf.Session() as sess:
    print('Loading Model...')

    ckpt = tf.train.get_checkpoint_state(hps.model_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    simulation = StockSimulation(train_data, train_pred, hps)
    validator = StockSimulation(test_data, test_pred, hps)

    evaluation = Evaluator(master_network, simulation, validator, hps, session=sess)
    evaluation.show()
