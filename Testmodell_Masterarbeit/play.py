import os
import random
import tensorflow as tf
import numpy as np
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import Predictor

tf.get_logger().setLevel('ERROR')


# region Hyperparams
warengruppe = 1

simulation_params = {
    'InputDirectory': os.path.join('files', 'raw'),
    'OutputDirectory': os.path.join('files', 'prepared'),
    'ZielWarengruppen': [warengruppe],
    'StatStateCategoricals': {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6},
}

predictor_dir = os.path.join('files',  'models', 'PredictorV2', '02RegWG' + str(warengruppe))
available_weights = os.listdir(predictor_dir)
available_weights.sort()
predictor_path = os.path.join(predictor_dir, available_weights[-1])
# endregion

pipeline = DataPipeLine(**simulation_params)
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
print(' and done ;)')

simulation = StockSimulation(train_data, pred, 0, 'Bestandsreichweite')

spiele = True

while spiele:
    step = 0
    state, info = simulation.reset()
    done = False
    while not done:
        step += 1
        print(state)
        action = int(input('Bestellmenge: '))
        if action == 99:
            break
        reward, done, state = simulation.make_action(action)
        print('Belohnung: ', reward)

    spiele = input('Weiterspielen?(j/n)') == 'j'




