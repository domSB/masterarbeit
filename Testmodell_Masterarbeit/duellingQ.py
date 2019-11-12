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

evaluation_run = 31
warengruppe = [55]
detail_warengruppe = [2363]
bestell_zyklus = 3

state_size = np.array([18])  # Zeitdimension, 6 Vorhersagen, Bestand, Abschriften, Fehlbestand
action_size = 6
learning_rate = 0.0001

memory_size = 30000

episodes = 10000
pretrain_episodes = 5
batch_size = 64

learn_step = 1
max_tau = learn_step * 10000

epsilon_start = 1
epsilon_stop = 0.03
epsilon_decay = 0.9999

gamma = 0.99

training = True
dir_name = str(evaluation_run) + 'eval' + str(warengruppe[0])
if detail_warengruppe:
    dir_name = dir_name + '-' + str(detail_warengruppe[0])
model_path = os.path.join('files', 'models', 'DDDQN', dir_name)
log_dir = os.path.join('files', 'logging', 'DDDQN', dir_name)

simulation_params = {
    'ZielWarengruppen': warengruppe,
    'DetailWarengruppe': detail_warengruppe
}

predictor_dir = os.path.join('files',  'models', 'PredictorV2', '02RegWG' + str(warengruppe[0]))
available_weights = os.listdir(predictor_dir)
available_weights.sort()
predictor_path = os.path.join(predictor_dir, available_weights[-1])
# endregion

pipeline = DataPipeLine(**simulation_params)
simulation_data = pipeline.get_regression_data()
train_data, test_data = split_np_arrays(*simulation_data)

print([tr.shape for tr in train_data])
# state_size[0] += simulation_data[2].shape[1]

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

simulation = StockSimulation(train_data, pred, 1, 'TDGewinn V2', bestell_zyklus)

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
    artikel_markt = simulation.possibles[np.random.choice(len(simulation.possibles))]
    saver = tf.train.Saver(max_to_keep=1)
    for episode in range(pretrain_episodes):
        state, info = simulation.reset(artikel_markt)
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
        state, info = simulation.reset(artikel_markt)
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



