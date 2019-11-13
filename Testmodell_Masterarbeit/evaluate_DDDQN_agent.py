import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import DDDQAgent, Predictor
from agents.evaluation import Evaluator
from utils import Hyperparameter

plt.style.use('ggplot')
tf.get_logger().setLevel('ERROR')

# region Hyperparams
hps = Hyperparameter()
hps.load(os.path.join('files', 'logging', 'DDDQN', '38eval17', 'Hyperparameter.yaml'))

predictor_dir = os.path.join('files',  'models', 'PredictorV2', '02RegWG' + str(hps.warengruppe[0]))
available_weights = os.listdir(predictor_dir)
available_weights.sort()
predictor_path = os.path.join(predictor_dir, available_weights[-1])
agent_path = os.path.join('files', 'models', 'DDDQN', '38eval17')
# endregion

pipeline = DataPipeLine(ZielWarengruppen=hps.warengruppe, DetailWarengruppe=hps.detail_warengruppe)
simulation_data = pipeline.get_regression_data()
train_data, test_data = split_np_arrays(*simulation_data, percentage=0.3)

predictor = Predictor()
predictor.build_model(
    dynamic_state_shape=simulation_data[1].shape[2],
    static_state_shape=simulation_data[2].shape[1]
)
predictor.load_from_weights(predictor_path)
print('Predicting', end='')
train_pred = predictor.predict(
    {
        'dynamic_input': train_data[1],
        'static_input': train_data[2]
    }
)
print('and done ;)')
print('Predicting', end='')
test_pred = predictor.predict(
    {
        'dynamic_input': test_data[1],
        'static_input': test_data[2]
    }
)
print('and done ;)')

simulation = StockSimulation(train_data, train_pred, hps)
validator = StockSimulation(test_data, test_pred, hps)
# endregion

session = tf.Session()
agent = DDDQAgent(session, hps)

saver = tf.train.Saver()
saver.restore(agent.sess, tf.train.latest_checkpoint(agent_path))

evaluation = Evaluator(agent, simulation, validator, hps)
evaluation.show()


