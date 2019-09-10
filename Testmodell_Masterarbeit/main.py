
from simulation import StockSimulationV2
from agents import AgentTwo, Predictor

import os
import numpy as np

# import cProfile

""" Hyperparameters """
# region Simulation Parameters
data_dir = os.path.join('files', 'raw')
output_dir = os.path.join('files', 'prepared')
warengruppen_maske = [1, 12, 55, 80, 17, 77, 71, 6, 28]
dyn_state_scalar_cols = ['Menge', 'MaxTemp_1D', 'MinTemp_1D', 'Wolken_1D',
                         'Regen_1D', 'MaxTemp_2D', 'MinTemp_2D', 'Wolken_2D', 'Regen_2D',
                         'Preis', 'relRabatt', 'absRabatt', 'vDauer']
dyn_state_label_cols = ['in1', 'in2', 'in3', 'in4', 'in5']
dyn_state_category_cols = {'Wochentag': 7, 'Kalenderwoche': 54}
stat_state_scalar_cols = ['Eigenmarke', 'GuG', 'OSE', 'Saisonal', 'Kern', 'Bio', 'Glutenfrei',
                          'Laktosefrei']
stat_state_category_cols = {'MHDgroup': 7, 'Warengruppe': 9, 'Detailwarengruppe': None, 'Einheit': None}
simulation_params = {
    'InputPath': data_dir,
    'OutputPath': output_dir,
    'ZielWarengruppen': warengruppen_maske,
    'Type': 'Markt',
    'DynStateScalarCols': dyn_state_scalar_cols,
    'DynStateLabelCols': dyn_state_label_cols,
    'DynStateCategoryCols': dyn_state_category_cols,
    'StatStateScalarCols': stat_state_scalar_cols,
    'StatStateCategoryCols': stat_state_category_cols,
    'StartDatum': '2017-01-01',
    'EndDatum': '2017-12-31',
    'StepSize': 6
}
validator_params = simulation_params
validator_params.update({
    'StartDatum': '2018-01-01',
    'EndDatum': '2018-12-31'
})

# endregion

# region  Hyperparameter
epochs = 300
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
n_step = 64
update_target_network = n_step * 10
use_model_path = os.path.join('files', 'models', 'AgentV2', '2019-08-27-23.54.59', 'model.h5')
use_saved_model = False

agent_params = {
    'MemorySize': 300*40,
    'StateShape': 6,
    'AktionSpace': 6,
    'Gamma': 0.9,
    'LearningRate': 0.001,
    'LearningRateDecay': 0.001/epochs,
    'BatchSize': 256,
    'Epsilon': 1,
    'EpsilonDecay': 0.99,
    'EpsilonMin': 0.01,
    'PossibleActions': possible_actions,
    'RunDescription': '5AndereDynamicState'
}
if not do_train:
    agent_params.update(
        {
            'Epsilon': 0,
            'EpsilonDecay': 0
        }
    )

predictor_params = {
    'forecast_state': 5,
    'learning_rate': 0.001,
    'time_steps': 6,
    'dynamic_state_shape': 73,
    'static_state_shape': 490
}
predictor_path = os.path.join('files', 'models', 'Predictor', '1RegBaselineTime', 'weights.15-0.02.hdf5')

# endregion

# region Initilize
simulation = StockSimulationV2(**simulation_params)
validator = StockSimulationV2(**validator_params)

agent = AgentTwo(**agent_params)
predictor = Predictor()
predictor.build_model(**predictor_params)
predictor.load_from_weights(predictor_path)
if use_saved_model:
    agent.load(use_model_path)
# endregion

# region Training Loop



global_steps = 0
stats = {"loss": [], "acc": [], "rew": []}
for epoch in range(epochs):
    full_state, info = simulation.reset()
    predict_state = predictor.predict(full_state['RegressionState'])
    agent_state = np.concatenate((predict_state, np.array([full_state['AgentState']])), axis=1)
    val_full_state, _ = validator.reset()
    val_predict_state = predictor.predict(val_full_state['RegressionState'])
    val_agent_state = np.concatenate((val_predict_state, np.array([val_full_state['AgentState']])), axis=1)
    val_fertig = False
    current_rewards = []
    current_val_rewards = []
    current_actions = []
    while True:
        # Train
        action = agent.act(agent_state)
        global_steps += 1
        reward, fertig, new_full_state = simulation.make_action(action)
        new_predict_state = predictor.predict(new_full_state['RegressionState'])
        new_agent_state = np.concatenate((new_predict_state, np.array([new_full_state['AgentState']])), axis=1)
        current_rewards.append(reward)
        current_actions.append(action)
        agent.remember(agent_state[0], action, reward, new_agent_state[0], fertig)
        agent_state = new_agent_state

        # Validate
        if not val_fertig:  # Validation Zeitraum ggf. kürzer oder gleichlang
            val_action = agent.act(val_agent_state)
            val_reward, val_fertig, new_val_full_state = validator.make_action(val_action)
            new_val_predict_state = predictor.predict(new_val_full_state['RegressionState'])
            new_val_agent_state = np.concatenate(
                (new_val_predict_state, np.array([new_val_full_state['AgentState']])),
                axis=1
            )
            current_val_rewards.append(val_reward)
            val_agent_state = new_val_agent_state

        if global_steps % n_step == 0:
            if do_train:
                history = agent.replay()
                if history:
                    curr_loss = history["loss"][0]
                    curr_acc = history["acc"][0]
                    stats["loss"].append(curr_loss)
                    stats["acc"].append(curr_acc)
            else:
                curr_loss = 0
                curr_acc = 0

        if global_steps % update_target_network == 0:
            agent.target_train()

        if fertig:
            if do_train:
                history = agent.replay()
                curr_loss = history["loss"][0]
                curr_acc = history["acc"][0]
            else:
                curr_loss = 0
                curr_acc = 0
            tf_summary = agent.sess.run(
                agent.merged,
                feed_dict={
                    agent.loss: curr_loss,
                    agent.accuracy: curr_acc,
                    agent.rewards: current_rewards,
                    agent.val_rewards: current_val_rewards,
                    agent.theo_bestand: simulation.stat_theo_bestand,
                    agent.fakt_bestand: simulation.stat_fakt_bestand,
                    agent.actions: current_actions,
                    agent.tf_epsilon: agent.epsilon
                    }
                )
            agent.writer.add_summary(tf_summary, epoch)
            if epoch % 10 == 0:
                print("Epoche {}".format(epoch))
                agent.save()
                # TODO: Validate Model with a trial Period in a seperate Simulation
            else:
                print('.', end='')
            break
agent.writer.close()
agent.sess.close()


#big_loop()
# endregion
