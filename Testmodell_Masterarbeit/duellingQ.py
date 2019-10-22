import os
import random
import tensorflow as tf
from collections import deque
import numpy as np
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import DDDQAgent, Experience

tf.get_logger().setLevel('ERROR')


def name_run(number):
    return 'Run' + str(number)


# region Hyperparams
state_size = np.array([9+6+3])  # Zeitdimension, 6 Vorhersagen, Bestand, Abschriften, Fehlbestand
time_steps = 6
action_size = 6
learning_rate = 0.0001

memory_size = 10000

episodes = 1000
pretrain_episodes = int(memory_size / 300)  # etwas mehr als 300 Experiences per Episode. An Anfang kürzere möglich.
batch_size = 32

learn_step = 2
max_tau = learn_step * 200

epsilon_start = 1
epsilon_stop = 0.05
epsilon_decay = 0.9999

gamma = 0.999

test_artikel = 18506
test_markt = 5

training = True

run_id = 30

while os.path.exists(os.path.join('files', 'models', 'DDDQN', name_run(run_id))):
    run_id += 1
model_path = os.path.join('files', 'models', 'DDDQN', name_run(run_id))
log_dir = os.path.join('files', 'logging', 'DDDQN', name_run(run_id))

simulation_params = {
    'InputDirectory': os.path.join('files', 'raw'),
    'OutputDirectory': os.path.join('files', 'prepared'),
    'ZielWarengruppen': [71],
    'StatStateCategoricals': {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6},
}

predictor_path = os.path.join('files', 'models', 'PredictorV2', '01RegWG71', 'weights.30-0.21.hdf5')
# endregion

pipeline = DataPipeLine(**simulation_params)
simulation_data = pipeline.get_regression_data()
train_data, test_data = split_np_arrays(*simulation_data)
simulation = StockSimulation(train_data, predictor_path)

if training:
    # region Initialisieren
    session = tf.Session()
    agent = DDDQAgent(
        epsilon_start,
        epsilon_stop,
        epsilon_decay,
        learning_rate,
        batch_size,
        action_size,
        state_size,
        time_steps,
        gamma,
        memory_size,
        session,
        log_dir
    )
    # endregion
    # region ReplayBuffer befüllen
    saver = tf.train.Saver()
    for episode in range(pretrain_episodes):
        state, info = simulation.reset((test_artikel, test_markt))
        recurrent_state = deque(maxlen=time_steps)
        for i in range(time_steps):
            recurrent_state.append(state)
        done = False
        while not done:
            action = random.choice(agent.possible_actions)
            reward, done, next_state = simulation.make_action(np.argmax(action))
            next_recurrent_state = recurrent_state
            next_recurrent_state.append(next_state)
            experience = Experience(recurrent_state, reward, done, next_recurrent_state, action)
            agent.remember(experience)
            recurrent_state = next_recurrent_state
    # endregion
    tau = 0
    for episode in range(episodes):
        # region Training
        step = 0
        state, info = simulation.reset((test_artikel, test_markt))
        recurrent_state = deque(maxlen=time_steps)
        for i in range(time_steps):
            recurrent_state.append(state)
        done = False
        while not done:
            step += 1
            tau += 1
            action = agent.act(np.array(recurrent_state))
            reward, done, next_state = simulation.make_action(np.argmax(action))
            next_recurrent_state = recurrent_state
            next_recurrent_state.append(next_state)
            experience = Experience(recurrent_state, reward, done, next_recurrent_state, action)
            agent.remember(experience)
            recurrent_state = next_recurrent_state

            if step % learn_step == 0:
                agent.train()

            if tau > max_tau:
                agent.update_target()
                tau = 0
        # endregion
        # region Lerninformationen
        summary = agent.sess.run(
            agent.merged,
            feed_dict={
                agent.v_rewards: simulation.statistics.rewards(),
                agent.v_actions: simulation.statistics.actions(),
                agent.v_bestand: simulation.statistics.bestand(),
                agent.v_abschriften: simulation.statistics.abschrift(),
                agent.v_fehlmenge: simulation.statistics.fehlmenge(),
                agent.v_absatz: simulation.statistics.absaetze(),
                agent.v_epsilon: agent.epsilon,
                agent.v_loss: agent.curr_loss,
                agent.v_beta: agent.memory.per_beta
            }
        )
        agent.writer.add_summary(summary, episode)
        agent.writer.flush()
        if episode % 25 == 0:
            print('Finished Episode: {epi} @ Epsilon {epsi}'.format(epi=episode, epsi=agent.epsilon))
            save_path = saver.save(agent.sess, os.path.join(model_path, 'model_{episode}.ckpt').format(episode=episode))
        # endregion



