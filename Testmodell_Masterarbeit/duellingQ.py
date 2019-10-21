import os
import tensorflow as tf
from collections import deque
import random
import numpy as np
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays

from collections import defaultdict
tf.get_logger().setLevel('ERROR')


class DDDQNetwork:
    def __init__(self, _state_size, _time_steps, _action_size, _learning_rate, name):
        self.state_size = _state_size
        self.action_size = _action_size
        self.learning_rate = _learning_rate
        self.name = name

        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(tf.float32, [None, _time_steps, *_state_size], name='Inputs')

            self.actions_ = tf.placeholder(tf.float32, [None, _action_size], name='Actions')

            self.target_q = tf.placeholder(tf.float32, [None], name='Target')

            self.lstm = tf.keras.layers.LSTM(
                units=64
            )(self.inputs_)
            self.dense = tf.keras.layers.Dense(
                units=64,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='EingangsDense'
            )(self.lstm)
            self.value_fc = tf.keras.layers.Dense(
                units=64,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='ValueFC'
            )(self.dense)
            self.value = tf.keras.layers.Dense(
                units=1,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='Value'
            )(self.value_fc)
            self.advantage_fc = tf.keras.layers.Dense(
                units=64,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='AdvantageFC'
            )(self.dense)
            self.advantage = tf.keras.layers.Dense(
                units=self.action_size,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='Advantage'
            )(self.advantage_fc)

            # Zusammenführen von V(s) und A(s, a)
            self.output = self.value + tf.subtract(
                self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True)
            )

            self.q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class Memory:
    def __init__(self, _memory_size):
        self.memory = deque(maxlen=_memory_size)

    def store(self, _experience):
        self.memory.append(_experience)

    def sample(self, _batch_size):
        return random.sample(self.memory, _batch_size)


class Experience:
    def __init__(self, _state, _reward, _done, _next_state, _action):
        self.state = np.array(_state)
        self.reward = _reward
        self.done = _done
        self.next_state = np.array(_next_state)
        self.action = _action


class DDDQAgent:
    def __init__(self, _eps_start, _eps_stop, _eps_decay, _batch_size, _action_size, _time_steps, _gamma, _session, _log_dir):
        self.sess = _session
        self.epsilon = _eps_start
        self.eps_stop = _eps_stop
        self.eps_decay = _eps_decay
        self.batch_size = _batch_size
        self.step_size = _time_steps
        self.gamma = _gamma
        self.curr_loss = -1
        self.possible_actions = np.identity(_action_size, dtype=int).tolist()
        self.dq_network = DDDQNetwork(state_size, _time_steps, _action_size, learning_rate, name='DQNetwork')
        self.target_network = DDDQNetwork(state_size, _time_steps, _action_size, learning_rate, name='TargetNetwork')
        self.game_buffer = Memory(memory_size)
        self.sess.run(tf.global_variables_initializer())
        self.update_target()
        self.writer = tf.summary.FileWriter(_log_dir, self.sess.graph)
        with tf.name_scope('Belohnungen'):
            self.v_rewards = tf.placeholder(tf.float32, shape=None, name='Belohnungen')
            self.v_rewards_sum = tf.math.reduce_sum(self.v_rewards)
            self.summary_reward_sum = tf.summary.scalar('Summe', self.v_rewards_sum)
            self.v_rewards_min = tf.math.reduce_min(self.v_rewards)
            self.summary_reward_min = tf.summary.scalar('Minimum', self.v_rewards_min)
        with tf.name_scope('Modell'):
            self.v_epsilon = tf.placeholder(tf.float32, shape=None, name='Epsilon')
            self.summary_epsilon = tf.summary.scalar('Epsilon', self.v_epsilon)
            self.v_loss = tf.placeholder(tf.float32, shape=None, name='Loss')
            self.summary_loss = tf.summary.scalar('Loss', self.v_loss)
        with tf.name_scope('Aktionen'):
            self.v_actions = tf.placeholder(tf.float32, shape=None, name='Aktionen')
            self.action_histo = tf.summary.histogram('Aktionen', self.v_actions)
            self.v_actions_sum = tf.math.reduce_sum(self.v_actions)
            self.summary_actions_sum = tf.summary.scalar('Bestellmenge', self.v_actions_sum)
        with tf.name_scope('Bestand'):
            self.v_bestand = tf.placeholder(tf.float32, shape=None, name='Bestand')
            self.v_bestand_max = tf.math.reduce_max(self.v_bestand)
            self.summary_bestand_max = tf.summary.scalar('Maximum', self.v_bestand_max)
            self.v_bestand_mean = tf.math.reduce_mean(self.v_bestand)
            self.summary_bestand_mean = tf.summary.scalar('Durchschnitt', self.v_bestand_mean)
        with tf.name_scope('Bewegungen'):
            self.v_abschriften = tf.placeholder(tf.float32, shape=None)
            self.v_abschriften_sum = tf.math.reduce_sum(self.v_abschriften, name='PLAbschriften')
            self.summary_abschriften_sum = tf.summary.scalar('Abschriften', self.v_abschriften_sum)
            self.v_fehlmenge = tf.placeholder(tf.float32, shape=None, name='PLFehlmenge')
            self.v_fehlmenge_sum = tf.math.reduce_sum(self.v_fehlmenge)
            self.summary_fehlmenge_sum = tf.summary.scalar('Fehlmenge', self.v_fehlmenge_sum)
            self.v_absatz = tf.placeholder(tf.float32, shape=None, name='PLAbsatz')
            self.v_absatz_sum = tf.math.reduce_sum(self.v_absatz)
            self.summary_absatz_sum = tf.summary.scalar('Absatz', self.v_absatz_sum)
            self.v_fehlmenge_proz = tf.math.divide(self.v_fehlmenge_sum, self.v_absatz_sum)
            self.summary_fehlmenge_proz = tf.summary.scalar('FehlmengeProzent', self.v_fehlmenge_proz)
            self.v_abschrift_proz = tf.math.divide(self.v_abschriften_sum, self.v_actions_sum)
            self.summary_abschrift_proz = tf.summary.scalar('AbschriftProzent', self.v_abschrift_proz)

        self.merged = tf.summary.merge_all()

    def act(self, _state):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_stop)
        if self.epsilon > np.random.rand():
            _action = random.choice(self.possible_actions)
        else:
            qs_of_state = self.sess.run(
                self.dq_network.output,
                feed_dict={
                    self.dq_network.inputs_: _state.reshape((1, *_state.shape))
                })
            choice = np.argmax(qs_of_state)
            _action = self.possible_actions[int(choice)]
        return _action

    def update_target(self):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DQNetwork')
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'TargetNetwork')
        update_ops = []
        for from_var, to_var in zip(from_vars, to_vars):
            update_ops.append(to_var.assign(from_var))
        self.sess.run(update_ops)

    def remember(self, _experience):
        self.game_buffer.store(_experience)

    def train(self):
        batch = self.game_buffer.sample(self.batch_size)
        state_batch = np.array([exp.state for exp in batch])
        reward_batch = np.array([exp.reward for exp in batch])
        done_batch = np.array([exp.done for exp in batch])
        next_state_batch = np.array([exp.next_state for exp in batch])
        action_batch = np.array([exp.action for exp in batch])

        target_qs_batch = []
        q_next_state = self.sess.run(
            self.dq_network.output,
            feed_dict={
                self.dq_network.inputs_: next_state_batch
            }
        )
        q_target_next_state = self.sess.run(
            self.target_network.output,
            feed_dict={
                self.target_network.inputs_: next_state_batch
            }
        )
        for i in range(len(batch)):
            terminal = done_batch[i]

            action = np.argmax(q_next_state[i])

            if terminal:
                target_qs_batch.append(reward_batch[i])
            else:
                target = reward_batch[i] + self.gamma * q_target_next_state[i][action]
                target_qs_batch.append(target)

        target_qs_batch = np.array(target_qs_batch)
        _, loss, = self.sess.run(
            [self.dq_network.optimizer, self.dq_network.loss],
            feed_dict={
                self.dq_network.inputs_: state_batch,
                self.dq_network.target_q: target_qs_batch,
                self.dq_network.actions_: action_batch
            }
        )
        self.curr_loss = loss


class ProbeSimulation:
    def __init__(self):
        self.step = 0

    @property
    def state(self):
        return {'PredictedState': np.random.random((6, 16)), 'Agentstate': np.random.random((3,))}

    def reset(self):
        self.step = 0
        return self.state, 'Wir sind im Probelauf'

    def make_action(self, _action):
        rew = np.argmax(_action) * 0.3
        self.step += 1
        return rew, self.step >= 10, self.state


# region Hyperparams
state_size = np.array([6+3])
time_steps = 6
action_size = 6
learning_rate = 0.0001

episodes = 4000
pretrain_episodes = 4
batch_size = 32

learn_step = 8
max_tau = learn_step * 1000

epsilon_start = 1
epsilon_stop = 0.01
epsilon_decay = 0.9999

gamma = 0.999

memory_size = 70000

training = True

model_path = os.path.join('files', 'models', 'DDDQN', 'Run30')
log_dir = os.path.join('files', 'logging', 'DDDQN', 'Run30')

simulation_params = {
    'InputDirectory': os.path.join('files', 'raw'),
    'OutputDirectory': os.path.join('files', 'prepared'),
    'ZielWarengruppen': [71],
    'StatStateCategoricals': {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6},
}

predictor_path = os.path.join('files', 'models', 'PredictorV2', '01RegWG71', 'weights.30-0.21.hdf5')
# endregion

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

pipeline = DataPipeLine(**simulation_params)
simulation_data = pipeline.get_regression_data()
train_data, test_data = split_np_arrays(*simulation_data)
simulation = StockSimulation(train_data, predictor_path)

if training:
    session = tf.Session()
    # simulation = ProbeSimulation(state_size)
    agent = DDDQAgent(epsilon_start, epsilon_stop, epsilon_decay, batch_size, action_size, time_steps, gamma, session, log_dir)
    saver = tf.train.Saver()
    for episode in range(pretrain_episodes):
        state, info = simulation.reset()
        recurrent_state = deque(maxlen=time_steps)
        for i in range(time_steps):
            recurrent_state.append(state)
        done = False
        while not done:
            action = agent.act(np.array(recurrent_state))
            reward, done, next_state = simulation.make_action(np.argmax(action))
            next_recurrent_state = recurrent_state
            next_recurrent_state.append(next_state)
            experience = Experience(recurrent_state, reward, done, next_recurrent_state, action)
            agent.remember(experience)
            recurrent_state = next_recurrent_state
    tau = 0
    for episode in range(episodes):
        step = 0
        state, info = simulation.reset()
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
                agent.v_loss: agent.curr_loss
            }
        )
        agent.writer.add_summary(summary, episode)
        agent.writer.flush()
        if episode % 25 == 0:
            print('Finished Episode: {epi} @ Epsilon {epsi}'.format(epi=episode, epsi=agent.epsilon))
            save_path = saver.save(agent.sess, os.path.join(model_path, 'model_{episode}.ckpt').format(episode=episode))
    session.close()



