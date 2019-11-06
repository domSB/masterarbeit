import os
import random
import tensorflow as tf
from collections import deque
import numpy as np
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import DDDQAgent, Experience, Predictor

tf.get_logger().setLevel('ERROR')


# region Hyperparams
warengruppe = 6

state_size = np.array([18])  # Zeitdimension, 6 Vorhersagen, Bestand, Abschriften, Fehlbestand
time_steps = 3
action_size = 6
learning_rate = 0.0001

memory_size = 100000

episodes = 3000
pretrain_episodes = int(memory_size / 300)  # etwas mehr als 300 Experiences per Episode. An Anfang kürzere möglich.
batch_size = 32

learn_step = 1
max_tau = learn_step * 10000

epsilon_start = 1
epsilon_stop = 0.01
epsilon_decay = 0.9999

gamma = 0.99

training = True

model_path = os.path.join('files', 'models', 'DDDQN', '02eval' + str(warengruppe))
log_dir = os.path.join('files', 'logging', 'DDDQN', '02eval' + str(warengruppe))

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
train_data, test_data = split_np_arrays(*simulation_data)

print([tr.shape for tr in train_data])
state_size[0] += simulation_data[2].shape[1]

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

simulation = StockSimulation(train_data, pred, 2, 'Bestandsreichweite V2', 3)

# endregion

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
        gamma,
        memory_size,
        session,
        log_dir
    )

    # endregion
    # region ReplayBuffer befüllen
    saver = tf.train.Saver(max_to_keep=1)
    for episode in range(pretrain_episodes):
        state, info = simulation.reset()
        # recurrent_state = deque(maxlen=time_steps)
        # for i in range(time_steps):
        #     recurrent_state.append(state)
        done = False
        while not done:
            action = random.choice(agent.possible_actions)
            reward, done, next_state = simulation.make_action(np.argmax(action))
            # next_recurrent_state = recurrent_state
            # next_recurrent_state.append(next_state)
            experience = Experience(state, reward, done, next_state, action)
            agent.remember(experience)
            state = next_state
    # endregion
    tau = 0
    for episode in range(episodes):
        # region Training
        step = 0
        state, info = simulation.reset()
        # recurrent_state = deque(maxlen=time_steps)
        # for i in range(time_steps):
        #     recurrent_state.append(state)
        done = False
        while not done:
            step += 1
            tau += 1
            action = agent.act(np.array(state))
            reward, done, next_state = simulation.make_action(np.argmax(action))
            # next_recurrent_state = recurrent_state
            # next_recurrent_state.append(next_state)
            experience = Experience(state, reward, done, next_state, action)
            agent.remember(experience)
            state = next_state

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
            save_path = saver.save(agent.sess, os.path.join(model_path, 'model.ckpt').format(episode=episode))
        # endregion



