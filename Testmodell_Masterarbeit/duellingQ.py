import os
import random
import tensorflow as tf
import numpy as np
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import DDDQAgent, Experience, Predictor

tf.get_logger().setLevel('ERROR')


# region Hyperparameter
warengruppe = 55
bestell_zyklus = 3

state_size = np.array([18])  # Zeitdimension, 6 Vorhersagen, Bestand, Abschriften, Fehlbestand
action_size = 6
learning_rate = 0.0001

memory_size = 10000

episodes = 3000
pretrain_episodes = int(memory_size / (388 / bestell_zyklus))
# etwas mehr als 300 Experiences per Episode. An Anfang kürzere möglich.
batch_size = 32

learn_step = 1
max_tau = learn_step * 10000

epsilon_start = 1
epsilon_stop = 0.05
epsilon_decay = 0.9999

gamma = 0.99

training = True

model_path = os.path.join('files', 'models', 'DDDQN', '07eval' + str(warengruppe))
log_dir = os.path.join('files', 'logging', 'DDDQN', '07eval' + str(warengruppe))

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

simulation = StockSimulation(train_data, pred, 2, 'Bestandsreichweite V2', bestell_zyklus)

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
        done = False
        while not done:
            action = random.choice(agent.possible_actions)
            reward, done, next_state = simulation.make_action(np.argmax(action))
            experience = Experience(state, reward, done, next_state, action)
            agent.remember(experience)
            state = next_state
    # endregion
    tau = 0
    target_update_counter = 0
    for episode in range(episodes):
        # region Training
        step = 0
        state, info = simulation.reset()
        done = False
        while not done:
            step += 1
            tau += 1
            action = agent.act(np.array(state))
            reward, done, next_state = simulation.make_action(np.argmax(action))
            experience = Experience(state, reward, done, next_state, action)
            agent.remember(experience)
            state = next_state

            if step % learn_step == 0:
                agent.train()

            if tau > max_tau:
                agent.update_target()
                tau = 0
                target_update_counter += 1
        # endregion
        # region Lerninformationen
        summary = agent.sess.run(
            agent.merged,
            feed_dict={
                agent.v_rewards: simulation.statistics.rewards(),
                agent.v_reward_optimal: info['Optimal'],
                agent.v_actions: simulation.statistics.actions(),
                agent.v_bestand: simulation.statistics.bestand(),
                agent.v_abschriften: simulation.statistics.abschrift(),
                agent.v_fehlmenge: simulation.statistics.fehlmenge(),
                agent.v_absatz: simulation.statistics.absaetze(),
                agent.v_epsilon: agent.epsilon,
                agent.v_loss: agent.curr_loss,
                agent.v_beta: agent.memory.per_beta,
                agent.v_target_updates: target_update_counter
            }
        )
        agent.writer.add_summary(summary, episode)
        agent.writer.flush()
        if episode % 25 == 0:
            print('Finished Episode: {epi} @ Epsilon {epsi}'.format(epi=episode, epsi=agent.epsilon))
            save_path = saver.save(agent.sess, os.path.join(model_path, 'model.ckpt').format(episode=episode))
        # endregion



