import os
import random
import tensorflow as tf
import numpy as np
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import DDDQAgent, Experience, Predictor
from utils import Hyperparameter

tf.get_logger().setLevel('ERROR')


# region Hyperparameter
hps = Hyperparameter(
    run_id=33,
    warengruppe=[55],
    detail_warengruppe=[2363],
    use_one_article=False,
    bestell_zyklus=3,
    state_size=[18],  # Zeitdimension, 6 Vorhersagen, Bestand, Abschriften, Fehlbestand
    action_size=6,
    learning_rate=0.001,
    memory_size=30000,
    episodes=10000,
    pretrain_episodes=5,
    batch_size=32,
    learn_step=1,
    max_tau=10000,
    epsilon_start=1,
    epsilon_stop=0.03,
    epsilon_decay=0.9999,
    gamma=0.99,
    do_train=True,
    reward_func='TDGewinn V2',
    sim_state_group=1,
    main_size=256,
    main_activation='relu',
    main_regularizer=None,
    value_size=32,
    value_activation='relu',
    value_regularizer=None,
    avantage_size=32,
    advantage_activation='relu',
    advantage_regularizer=None,
    per_epsilon=0.01,
    per_alpha=0.6,
    per_beta=0.4,
    per_beta_increment=0.00025,
    per_error_clip=1.0
)

training = True
dir_name = str(hps.run_id) + 'eval' + str(hps.warengruppe[0])
if hps.detail_warengruppe:
    dir_name = dir_name + '-' + str(hps.detail_warengruppe[0])
hps.add_hparam('model_dir', os.path.join('files', 'models', 'DDDQN', dir_name))
hps.add_hparam('log_dir', os.path.join('files', 'logging', 'DDDQN', dir_name))
if not os.path.exists(hps.log_dir):
    os.mkdir(hps.log_dir)
    os.mkdir(hps.model_dir)

simulation_params = {
    'ZielWarengruppen': hps.warengruppe,
    'DetailWarengruppe': hps.detail_warengruppe
}

predictor_dir = os.path.join('files',  'models', 'PredictorV2', '02RegWG' + str(hps.warengruppe[0]))
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

simulation = StockSimulation(train_data, pred, hps)
hps.save(os.path.join(hps.log_dir, 'Hyperparameter.yaml'))
# endregion

if training:
    # region Initialisieren
    session = tf.Session()
    agent = DDDQAgent(session, hps)
    # endregion
    # region ReplayBuffer befüllen
    if hps.use_one_article:
        artikel_markt = simulation.possibles[np.random.choice(len(simulation.possibles))]
    else:
        artikel_markt = None
    saver = tf.train.Saver(max_to_keep=None)
    for episode in range(hps.pretrain_episodes):
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
    for episode in range(hps.episodes):
        # region Training
        step = 0
        state, info = simulation.reset(artikel_markt)
        done = False
        while not done:
            step += 1

            action = agent.act(np.array(state))
            reward, done, next_state = simulation.make_action(np.argmax(action))
            experience = Experience(state, reward, done, next_state, action)
            agent.remember(experience)
            state = next_state

            if step % hps.learn_step == 0:
                tau += 1
                agent.train()

            if tau > hps.max_tau:
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



