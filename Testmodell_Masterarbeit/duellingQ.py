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

            self.is_weights = tf.placeholder(tf.float32, [None, 1], name='IS_Weights')

            self.actions_ = tf.placeholder(tf.float32, [None, _action_size], name='Actions')

            self.target_q = tf.placeholder(tf.float32, [None], name='Target')

            self.lstm = tf.keras.layers.LSTM(
                units=64
            )(self.inputs_)
            self.dense = tf.keras.layers.Dense(
                units=128,
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

            self.absolute_errors = tf.abs(self.target_q - self.q)

            self.loss = tf.reduce_mean(self.is_weights * tf.squared_difference(self.target_q, self.q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class Memory:
    per_epsilon = 0.01
    per_alpha = 0.6
    per_beta = 0.4
    per_beta_increment = 0.001
    abs_error_clip = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, _experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_error_clip

        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []
        b_idx, b_is_weights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / n
        self.per_beta = np.min([1., self.per_beta + self.per_beta_increment])

        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.per_beta)
        for k in range(n):
            a, b = priority_segment * k, priority_segment * (k + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)

            sampling_probabilities = priority / self.tree.total_priority
            b_is_weights[k, 0] = np.power(n * sampling_probabilities, -self.per_beta) / max_weight
            b_idx[k] = index
            _experience = data
            memory_b.append(_experience)
        return b_idx, memory_b, b_is_weights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.per_epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_error_clip)
        ps = np.power(clipped_errors, self.per_alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class SumTree:
    """
    SumTree als Laufzeit optimierte Alternative zu einem sortierten Replaybuffer.
    Implementierung nach Thomas Simonini & Morvan Zhou.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # mit neuen Einträgen überschreiben
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


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
        self.memory = Memory(memory_size)
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
            self.v_beta = tf.placeholder(tf.float32, shape=None, name='Beta')
            self.summary_beta = tf.summary.scalar('Beta', self.v_beta)
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
        self.memory.store(_experience)

    def train(self):
        tree_idx, batch, is_weights_batch = self.memory.sample(self.batch_size)
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
        _, loss, absolute_errors = self.sess.run(
            [self.dq_network.optimizer, self.dq_network.loss, self.dq_network.absolute_errors],
            feed_dict={
                self.dq_network.inputs_: state_batch,
                self.dq_network.target_q: target_qs_batch,
                self.dq_network.actions_: action_batch,
                self.dq_network.is_weights: is_weights_batch
            }
        )
        self.memory.batch_update(tree_idx, absolute_errors)
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
state_size = np.array([6+6+3])  # Zeitdimension, 6 Vorhersagen, Bestand, Abschriften, Fehlbestand
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

memory_size = 100000

training = True

model_path = os.path.join('files', 'models', 'DDDQN', 'Run33')
log_dir = os.path.join('files', 'logging', 'DDDQN', 'Run33')

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
                agent.v_loss: agent.curr_loss,
                agent.v_beta: agent.memory.per_beta
            }
        )
        agent.writer.add_summary(summary, episode)
        agent.writer.flush()
        if episode % 25 == 0:
            print('Finished Episode: {epi} @ Epsilon {epsi}'.format(epi=episode, epsi=agent.epsilon))
            save_path = saver.save(agent.sess, os.path.join(model_path, 'model_{episode}.ckpt').format(episode=episode))
    session.close()



