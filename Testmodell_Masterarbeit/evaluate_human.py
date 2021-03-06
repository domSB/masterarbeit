import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import Predictor, Mensch
from agents.evaluation import Evaluator
from utils import Hyperparameter

plt.style.use('ggplot')
tf.get_logger().setLevel('ERROR')

# region Hyperparams
hps = Hyperparameter(
    run_id=5,
    warengruppe=[17],
    detail_warengruppe=None,
    sicherheitsaufschlag=0,
    rundung=None,
    bestell_zyklus=3,
    state_size=[4],
    reward_func='TDGewinn V2',
    state_FullPredict=False,
    state_Predict=True,
    state_Time=False,
    state_Weather=False,
    state_Sales=False,
    state_ArtikelInfo=False,
    rest_laufzeit=28,
    ordersatz_einheit=None,
    use_one_article=False,
    use_lstm=False,
    time_steps=1,
)

predictor_dir = os.path.join('files',  'models', 'PredictorV2', '02RegWG' + str(hps.warengruppe[0]))
available_weights = os.listdir(predictor_dir)
available_weights.sort()
predictor_path = os.path.join(predictor_dir, available_weights[-1])
# endregion

pipeline = DataPipeLine(ZielWarengruppen=hps.warengruppe, DetailWarengruppe=hps.detail_warengruppe)
simulation_data = pipeline.get_regression_data()
train_data, test_data = split_np_arrays(*simulation_data, percentage=0)

predictor = Predictor()
predictor.build_model(
    dynamic_state_shape=simulation_data[1].shape[2],
    static_state_shape=simulation_data[2].shape[1]
)
predictor.load_from_weights(predictor_path)
print('Predicting', end='')
pred = predictor.predict(
    {
        'dynamic_input': train_data[1],
        'static_input': train_data[2]
    }
)
print('and done ;)')

simulation = StockSimulation(train_data, pred, hps)
hps.set_hparam('state_size', list(simulation.state_size))
# endregion

agent = Mensch(hps)

evaluation = Evaluator(agent, None, simulation, hps, validation=False)
evaluation.show()


