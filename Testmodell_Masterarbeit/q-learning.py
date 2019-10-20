
from simulation import StockSimulation
from agents import Agent, Predictor
from data.access import DataPipeLine
from data.preparation import split_np_arrays

import os
import numpy as np

# import cProfile

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
epochs = 20
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
n_step = 4
update_target_network = 1000
use_model_path = os.path.join('files', 'models', 'AgentV2', '2019-08-27-23.54.59', 'model.h5')
use_saved_model = False

agent_params = {
    'MemorySize': 300*20,
    'AktionSpace': 6,
    'Gamma': 1,
    'LearningRate': 0.00001,
    'BatchSize': 32,
    'Epsilon': 1,
    'EpsilonDecay': 0.99995,
    'EpsilonMin': 0.01,
    'PossibleActions': possible_actions,
    'RunDescription': 'Blub'
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
simulation = StockSimulation(train_data, predictor_path)
validator = StockSimulation(test_data, predictor_path)

agent = Agent(**agent_params, ArticleStateShape=train_data[2].shape[1])

if use_saved_model:
    agent.load(use_model_path)
# endregion

# region Training Loop
#
agent_states = []
#
global_steps = 0
for epoch in range(epochs):
    epoch_step = 0
    state, info = simulation.reset()
    val_state, _ = validator.reset()
    val_fertig = False
    while True:
        # Train
        action = agent.act(state)
        global_steps += 1
        reward, fertig, new_state = simulation.make_action(action)
        epoch_step += 1
        agent.remember(state, action, reward, new_state, fertig)
        agent_states.append(state)
        state = new_state

        # Validate
        if not val_fertig:  # Validation Zeitraum ggf. kürzer oder gleichlang
            val_action = agent.act(val_state)
            val_reward, val_fertig, new_val_state = validator.make_action(val_action)
            val_agent_state = new_val_state

        if global_steps % n_step == 0:
            if do_train:
                history = agent.replay()
                if history:
                    curr_loss = history["loss"][0]
                    curr_acc = history["acc"][0]
            else:
                curr_loss = 0
                curr_acc = 0

        if global_steps % update_target_network == 0:
            agent.target_train()

        if fertig:
            if do_train:
                history = agent.replay()
                if history:
                    curr_loss = history["loss"][0]
                    curr_acc = history["acc"][0]
                else:
                    curr_loss = 0
                    curr_acc = 0
            else:
                curr_loss = 0
                curr_acc = 0

            tf_summary = agent.sess.run(
                agent.merged,
                feed_dict={
                    agent.loss: curr_loss,
                    agent.accuracy: curr_acc,
                    agent.rewards: simulation.statistics.rewards(),
                    agent.val_rewards: validator.statistics.rewards(),
                    agent.bestand: simulation.statistics.bestand(),
                    agent.fehlmenge: simulation.statistics.fehlmenge(),
                    agent.abschrift: simulation.statistics.abschrift(),
                    agent.actions: simulation.statistics.actions(),
                    agent.tf_epsilon: agent.epsilon
                    }
                )
            agent.writer.add_summary(tf_summary, epoch)
            agent.writer.flush()
            if epoch % 10 == 0:
                print('\nEpoche {}'.format(epoch))
                agent.save()
            else:
                print('.', end='')
            break
agent.writer.close()
agent.sess.close()
# endregion
