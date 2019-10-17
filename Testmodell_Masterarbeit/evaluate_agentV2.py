
from simulation import StockSimulation
from agents import Agent, Predictor
from data.access import DataPipeLine
from data.preparation import split_np_arrays

import os
import numpy as np
import tensorflow as tf

# import cProfile


def create_dataset(_lab, _dyn, _stat, _params):
    def gen():
        while True:
            rand_idx = np.random.randint(0, _lab.shape[0])
            labels = _lab[rand_idx]
            yield {'dynamic_input': _dyn[rand_idx], 'static_input': _stat[rand_idx]}, \
                  {
                      '1day': labels[0],
                      '2day': labels[1],
                      '3day': labels[2],
                      '4day': labels[3],
                      '5day': labels[4],
                      '6day': labels[5],
                  }

    _dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=(
            {'dynamic_input': tf.float32, 'static_input': tf.int8},
            {
                '1day': tf.int8,
                '2day': tf.int8,
                '3day': tf.int8,
                '4day': tf.int8,
                '5day': tf.int8,
                '6day': tf.int8,
            }
        ),
        output_shapes=(
            {'dynamic_input': tf.TensorShape([_params['time_steps'], _params['dynamic_state_shape']]),
             'static_input': tf.TensorShape([_params['static_state_shape']])},
            {
                '1day': tf.TensorShape([16]),
                '2day': tf.TensorShape([16]),
                '3day': tf.TensorShape([16]),
                '4day': tf.TensorShape([16]),
                '5day': tf.TensorShape([16]),
                '6day': tf.TensorShape([16]),
            }
        )
    )
    _dataset = _dataset.batch(_params['batch_size'])
    _dataset = _dataset.repeat()
    _dataset = _dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return _dataset

""" Hyperparameters """
# region Simulation Parameters
simulation_params = {
    'InputDirectory': os.path.join('files', 'raw'),
    'OutputDirectory': os.path.join('files', 'prepared'),
    'ZielWarengruppen': [71],
    'StatStateCategoricals': {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6},
}
# endregion

# region  Hyperparameter
epochs = 1000
do_train = True
order_none = 0
order_one = 1
order_two = 2
order_tree = 3
order_four = 4
order_five = 5

possible_actions = [
    order_none,
    order_one,
    order_two,
    order_tree,
    order_four,
    order_five
    ]
n_step = 16
update_target_network = n_step * 16
use_model_path = os.path.join('files', 'models', 'AgentV2', '2019-08-27-23.54.59', 'model.h5')
use_saved_model = False

agent_params = {
    'MemorySize': 300*200,
    'AktionSpace': 6,
    'Gamma': 1,
    'LearningRate': 0.001,
    'LearningRateDecay': 0.001/epochs,
    'BatchSize': 32,
    'Epsilon': 1,
    'EpsilonDecay': 0.999,
    'EpsilonMin': 0.03,
    'PossibleActions': possible_actions,
    'RunDescription': '13GrosserExpSpeicher'
}
if not do_train:
    agent_params.update(
        {
            'Epsilon': 0,
            'EpsilonDecay': 0
        }
    )

predictor_path = os.path.join('files', 'models', 'PredictorV2', '01RegWG71', 'weights.30-0.21.hdf5')

# endregion

# region Initilize
pipeline = DataPipeLine(**simulation_params)
simulation_data = pipeline.get_regression_data()
train_data, test_data = split_np_arrays(*simulation_data)
simulation = StockSimulation(train_data)
validator = StockSimulation(test_data)
agent = Agent(**agent_params, ArticleStateShape=train_data[2].shape[1])

params = {
    'forecast_state': 6,
    'learning_rate': 0.0001,
    'time_steps': None,
    'dynamic_state_shape': None,
    'static_state_shape': None,
    'epochs': 30,
    'batch_size': 32
}
params.update({
    'steps_per_epoch': int(train_data[1].shape[0] / params['batch_size']),
    'val_steps_per_epoch': int(test_data[1].shape[0] / params['batch_size']),
    'dynamic_state_shape': train_data[1].shape[2],
    'static_state_shape': train_data[2].shape[1],
    'Name': '01RegWG71'
})

dataset = create_dataset(*train_data[:3], params)
val_dataset = create_dataset(*test_data[:3], params)

predictor = Predictor()
predictor.build_model(dynamic_state_shape=train_data[1].shape[2], static_state_shape=train_data[2].shape[1])
predictor.load_from_weights(predictor_path)
# if use_saved_model:
#     agent.load(use_model_path)
# endregion

# TODO: let predictor predict a hole set for one article
# TODO: compare predictions to actual labels
# TODO: evaluate predictor with same dataset

history = predictor.model.evaluate(
    val_dataset,
    verbose=1,
    steps=params['steps_per_epoch']
)
lab, dyn, stat, _ = test_data
max_idx = lab.shape[0]
for i in range(10):
    print('Neue Runde\n-----')
    idx = np.random.randint(0, max_idx)
    inputs = {'dynamic_input': np.expand_dims(dyn[idx], axis=0), 'static_input': np.expand_dims(stat[idx], axis=0)}
    print(np.argmax(lab[idx], axis=1))
    print(np.argmax(predictor.predict(inputs), axis=1))

for value, name in zip(history, predictor.model.metrics_names):
    print(name, value)

agent_states = []
regression_states = []
full_state, info = simulation.reset()
predict_state = predictor.predict(full_state['RegressionState'])
regression_states.append(full_state['RegressionState'].copy())
agent_state = {
    'predicted_sales': predict_state,
    'current_stock': full_state['AgentState'],
    'article_info': full_state['RegressionState']['static_input'].reshape(-1)
}
agent_states.append(agent_state)
days = 0
while True:
    # Train
    print('Day:', days)
    action = agent.act(agent_state)
    reward, fertig, new_full_state = simulation.make_action(action)
    new_predict_state = predictor.predict(new_full_state['RegressionState'])
    regression_states.append(new_full_state['RegressionState'].copy())
    new_agent_state = {
        'predicted_sales': new_predict_state,
        'current_stock': new_full_state['AgentState'],
        'article_info': new_full_state['RegressionState']['static_input'].reshape(-1)
    }
    agent.remember(agent_state, action, reward, new_agent_state, fertig)
    agent_states.append(agent_state)
    agent_state = new_agent_state
    days += 1
    if fertig:
        break

for pred, true in zip(agent_states[:20], simulation.kristall_glas[:20]):
    print('True', np.argmax(true, axis=1))
    print('Pred', np.argmax(pred['predicted_sales'], axis=1), '\n')

