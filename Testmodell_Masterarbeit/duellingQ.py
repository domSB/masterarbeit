import os
import random
import time
import tensorflow as tf
import numpy as np
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import DDDQAgent, Experience, Predictor
from agents.evaluation import Evaluator
from utils import Hyperparameter, StateOperator

tf.get_logger().setLevel('ERROR')

# region Hyperparameter
hps = Hyperparameter(
    run_id=56,
    warengruppe=[28],
    detail_warengruppe=None,
    use_one_article=False,
    bestell_zyklus=3,
    state_size=[18],  # Zeitdimension, 6 Vorhersagen, Bestand, Abschriften, Fehlbestand
    action_size=6,
    learning_rate=0.00025,
    memory_size=100000,
    episodes=50000,
    pretrain_episodes=5,
    batch_size=32,
    learn_step=4,
    max_tau=5000,
    epsilon_start=1,
    epsilon_stop=0.05,
    epsilon_decay=0.99995,
    gamma=0.95,
    do_train=True,
    reward_func='TDGewinn V2',
    sim_state_group=2,
    use_lstm=True,
    lstm_units=32,
    time_steps=9,
    main_size=256,
    main_activation='elu',
    main_regularizer=None,
    value_size=32,
    value_activation='tanh',
    value_regularizer=None,
    avantage_size=32,
    advantage_activation='tanh',
    advantage_regularizer=None,
    drop_out_rate=0.3,
    per_epsilon=0.01,
    per_alpha=0.6,
    per_beta=0.4,
    per_beta_increment=0.00002,
    per_error_clip=1.0,
    use_importance_sampling=False,
    rest_laufzeit=6
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

predictor_dir = os.path.join('files',  'models', 'PredictorV2', '02RegWG' + str(hps.warengruppe[0]))
available_weights = os.listdir(predictor_dir)
available_weights.sort()
predictor_path = os.path.join(predictor_dir, available_weights[-1])
# endregion

pipeline = DataPipeLine(ZielWarengruppen=hps.warengruppe, DetailWarengruppe=hps.detail_warengruppe)
simulation_data = pipeline.get_regression_data()
train_data, test_data = split_np_arrays(*simulation_data)

print([tr.shape for tr in train_data])
if hps.sim_state_group > 1:
    state_size = hps.state_size
    state_size[0] += simulation_data[2].shape[1]
    hps.set_hparam('state_size', state_size)

predictor = Predictor()
predictor.build_model(
    dynamic_state_shape=simulation_data[1].shape[2],
    static_state_shape=simulation_data[2].shape[1]
)
predictor.load_from_weights(predictor_path)
print('Predicting ', end='')
train_pred = predictor.predict(
    {
        'dynamic_input': train_data[1],
        'static_input': train_data[2]
    }
)
print('and done ;)')
print('Predicting ', end='')
test_pred = predictor.predict(
    {
        'dynamic_input': test_data[1],
        'static_input': test_data[2]
    }
)
print('and done ;)')

simulation = StockSimulation(train_data, train_pred, hps)
train_op = StateOperator(hps)
val_op = StateOperator(hps)

validator = StockSimulation(test_data, test_pred, hps)


hps.save(os.path.join(hps.log_dir, 'Hyperparameter.yaml'))
# endregion

session = tf.Session()
agent = DDDQAgent(session, hps)
if training:

    # region ReplayBuffer befüllen
    if hps.use_one_article:
        artikel_markt = simulation.possibles[np.random.choice(len(simulation.possibles))]
    else:
        artikel_markt = None
    saver = tf.train.Saver(max_to_keep=None)
    for episode in range(hps.pretrain_episodes):
        state, info = simulation.reset(artikel_markt)
        train_op.start(state)
        done = False
        while not done:
            action = random.choice(agent.possible_actions)
            reward, done, next_state = simulation.make_action(np.argmax(action))
            train_op.add(next_state)
            experience = Experience(train_op.pre_state, reward, done, train_op.state, action)
            agent.remember(experience)
    # endregion
    tau = 0
    target_update_counter = 0
    for episode in range(hps.episodes):
        # region Training
        step = 0
        state, info = simulation.reset(artikel_markt)
        train_op.start(state)
        val_state, _ = validator.reset(artikel_markt)
        val_op.start(val_state)
        done = False
        while not done:
            step += 1

            action = agent.act(np.array(train_op.state))
            reward, done, next_state = simulation.make_action(np.argmax(action))
            train_op.add(next_state)
            experience = Experience(train_op.pre_state, reward, done, train_op.state, action)
            agent.remember(experience)

            val_action = agent.act(np.array(val_op.state))
            val_reward, _, val_next_state = validator.make_action(np.argmax(val_action))
            val_op.add(val_next_state)

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
                agent.v_val_rewards: validator.statistics.rewards(),
                agent.v_as: agent.curr_as,
                agent.v_vs: agent.curr_vs,
                agent.v_actions: simulation.statistics.actions(),
                agent.v_val_actions: validator.statistics.actions(),
                agent.v_bestand: simulation.statistics.bestand(),
                agent.v_abschriften: simulation.statistics.abschrift(),
                agent.v_fehlmenge: simulation.statistics.fehlmenge(),
                agent.v_absatz: simulation.statistics.absaetze(),
                agent.v_val_abschriften: validator.statistics.abschrift(),
                agent.v_val_fehlmenge: validator.statistics.fehlmenge(),
                agent.v_val_absatz: validator.statistics.absaetze(),
                agent.v_epsilon: agent.epsilon,
                agent.v_loss: agent.curr_loss,
                agent.v_beta: agent.memory.per_beta,
                agent.v_target_updates: target_update_counter
            }
        )
        agent.writer.add_summary(summary, episode)
        agent.writer.flush()
        if episode % 25 == 0:
            print(time.strftime('%T') + ' Episode: {epi} @ Epsilon {epsi}'.format(epi=episode, epsi=agent.epsilon))
            if episode % 1000 == 0:
                save_path = saver.save(agent.sess, os.path.join(hps.model_dir, 'model.ckpt'), global_step=episode)
        # endregion

evaluation = Evaluator(agent, simulation, validator, hps)
evaluation.show()


