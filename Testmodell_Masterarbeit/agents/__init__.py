import datetime
import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
from scipy.signal import lfilter

from utils import StateOperator

tf.get_logger().setLevel('ERROR')


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif 3 <= epoch < 10:
        return 1e-4
    else:
        return 1e-5


def normalized_columns_initializer(std=1.0):
    """ Vorgeschlagen von Arthur Juliani"""

    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def update_target_graph(from_scope, to_scope):
    """
    Helper function to copy Parameters from one Network to another.
    :param from_scope:
    :param to_scope:
    :return: list of ops to execute
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, _gamma):
    """
    Helper function to discount rewards.
    :param x:
    :param _gamma:
    :return:
    """
    return lfilter([1], [1, -_gamma], x[::-1], axis=0)[::-1]


# region Prädiktor
class Predictor(object):
    def __init__(self):
        self.model = None

    def build_model(self, **kwargs):
        dynamic_inputs = tf.keras.Input(shape=(6, kwargs['dynamic_state_shape']),
                                        name='dynamic_input')
        static_inputs = tf.keras.Input(shape=(kwargs['static_state_shape'],), name='static_input')
        dynamic_x = tf.keras.layers.LSTM(
            32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="LSTM_1",
            return_sequences=True
        )(dynamic_inputs)
        dynamic_x = tf.keras.layers.LSTM(
            32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="LSTM_2"
        )(dynamic_x)
        x = tf.keras.layers.concatenate([dynamic_x, static_inputs])
        x = tf.keras.layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="Dense_1"
        )(x)
        predictions_1d = tf.keras.layers.Dense(16, activation='softmax', name="1day")(x)
        predictions_2d = tf.keras.layers.Dense(16, activation='softmax', name="2day")(x)
        predictions_3d = tf.keras.layers.Dense(16, activation='softmax', name="3day")(x)
        predictions_4d = tf.keras.layers.Dense(16, activation='softmax', name="4day")(x)
        predictions_5d = tf.keras.layers.Dense(16, activation='softmax', name="5day")(x)
        predictions_6d = tf.keras.layers.Dense(16, activation='softmax', name="6day")(x)
        self.model = tf.keras.Model(
            inputs=[dynamic_inputs, static_inputs],
            outputs=[predictions_1d, predictions_2d, predictions_3d, predictions_4d, predictions_5d, predictions_6d])
        rms = tf.keras.optimizers.Adam(lr=kwargs.get('learning_rate', 0.001))
        self.model.compile(
            optimizer=rms,
            loss='categorical_crossentropy',
            loss_weights={'1day': 0.6, '2day': 0.5, '3day': 0.4, '4day': 0.3,
                          '5day': 0.3, '6day': 0.3},
            metrics=[tf.keras.metrics.categorical_accuracy]
        )

    def train(self, _dataset, _val_dataset, _params):
        if os.path.exists(os.path.join('files', 'logging', 'PredictorV2', _params['Name'])):
            name = datetime.datetime.now().__str__()
        else:
            name = _params['Name']
        os.mkdir(os.path.join('files', 'logging', 'PredictorV2', name))
        os.mkdir(os.path.join('files', 'models', 'PredictorV2', name))
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('files', 'logging', 'PredictorV2', name),
            histogram_freq=1,
            batch_size=32,
            write_graph=True,
            write_grads=True,
            update_freq='batch')
        nan_callback = tf.keras.callbacks.TerminateOnNaN()
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join('files', 'models', 'PredictorV2', name, 'weights.{epoch:02d}-{loss:.2f}.hdf5'),
            monitor='loss',
            verbose=0,
            period=1)
        lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(decay)
        stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0,
            patience=3,
            verbose=0,
            restore_best_weights=True
        )
        history = self.model.fit(
            _dataset,
            callbacks=[
                tb_callback,
                nan_callback,
                save_callback,
                lr_schedule_callback,
                stop_callback
            ],
            steps_per_epoch=_params['steps_per_epoch'],
            epochs=_params['epochs'],
            validation_data=_val_dataset,
            validation_steps=_params['val_steps_per_epoch']
        )
        return history

    def load_from_weights(self, path):
        self.model.load_weights(path)

    def predict(self, x):
        y = self.model.predict(x)
        y = np.swapaxes(np.array(y), 0, 1)
        return y


# endregion


# region Duelling Double Deep Agent
class DDDQNetwork:
    """
    Duelling Double Deep Agent Network
    """

    def __init__(self, hparams, name):
        self.state_size = hparams.state_size
        self.action_size = hparams.action_size
        self.learning_rate = hparams.learning_rate
        self.name = name

        with tf.variable_scope(self.name):
            if hparams.use_double_lstm:
                self.inputs = tf.placeholder(
                    shape=[None, hparams.time_steps, *hparams.state_size],
                    dtype=tf.float32,
                    name='Inputs')
                self.intermediate_lstm = tf.keras.layers.LSTM(
                    hparams.lstm_units,
                    return_sequences=True,
                    name='LSTM_1')(self.inputs)
                self.firsts = tf.keras.layers.LSTM(hparams.lstm_units, name='LSTM_2')(self.intermediate_lstm)
            elif hparams.use_lstm:
                self.inputs = tf.placeholder(
                    shape=[None, hparams.time_steps, *hparams.state_size],
                    dtype=tf.float32,
                    name='Inputs')
                self.firsts = tf.keras.layers.LSTM(hparams.lstm_units, name='LSTM')(self.inputs)
            else:
                self.inputs = tf.placeholder(shape=[None, *hparams.state_size], dtype=tf.float32, name='Inputs')
                self.firsts = tf.keras.layers.Flatten(name='FlatInputs')(self.inputs)
            self.is_weights = tf.placeholder(tf.float32, [None, 1], name='IS_Weights')

            self.actions_ = tf.placeholder(tf.float32, [None, hparams.action_size], name='Actions')

            self.target_q = tf.placeholder(tf.float32, [None], name='Target')

            self.dense_one = tf.keras.layers.Dense(
                units=hparams.main_size,
                activation=hparams.main_activation,
                kernel_regularizer=hparams.main_regularizer,
                # kernel_initializer=tf.random_normal_initializer(0., 0.3),
                # bias_initializer=tf.constant_initializer(0.1),
                name='EingangsDense'
            )(self.firsts)
            self.value_fc = tf.keras.layers.Dense(
                units=hparams.value_size,
                activation=hparams.value_activation,
                kernel_regularizer=hparams.value_regularizer,
                # kernel_initializer=tf.random_normal_initializer(0., 0.3),
                # bias_initializer=tf.constant_initializer(0.1),
                name='ValueFC'
            )(self.dense_one)
            self.value = tf.keras.layers.Dense(
                units=1,
                activation=None,
                # kernel_initializer=tf.random_normal_initializer(0., 0.3),
                # bias_initializer=tf.constant_initializer(0.1),
                name='Value'
            )(self.value_fc)
            self.advantage_fc = tf.keras.layers.Dense(
                units=hparams.avantage_size,
                activation=hparams.advantage_activation,
                kernel_regularizer=hparams.advantage_regularizer,
                # kernel_initializer=tf.random_normal_initializer(0., 0.3),
                # bias_initializer=tf.constant_initializer(0.1),
                name='AdvantageFC'
            )(self.dense_one)
            self.advantage = tf.keras.layers.Dense(
                units=self.action_size,
                activation=None,
                # kernel_initializer=tf.random_normal_initializer(0., 0.3),
                # bias_initializer=tf.constant_initializer(0.1),
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


class UniformSamplingMemory:
    """
    Replay Buffer with uniform sampling
    """

    def __init__(self, hparams):
        self.storage = deque(maxlen=hparams.memory_size)
        self.per_beta = 0

    def store(self, _experience):
        """
        Store an experience in the deque
        :param _experience:
        :return:
        """
        self.storage.append(_experience)

    def sample(self, n):
        """
        Return a batch of size n
        :param n:
        :return: None, Experiences, uniform_weights
        """
        storage_size = len(self.storage)
        index = np.random.choice(np.arange(storage_size),
                                 size=n,
                                 replace=False)
        experiences = [self.storage[i] for i in index]
        uniform_weights = np.array([[1 / n] for i in index])
        return None, experiences, uniform_weights

    def batch_update(self, tree_idx, abs_errors):
        """
        Methode für einheitlichen Aufbau.
        Macht nix.
        :param tree_idx:
        :param abs_errors:
        :return:
        """
        pass


class ImportanceSamplingMemory:
    """
    Replay-Buffer mit Importance Sampling
    """

    def __init__(self, hparams):
        self.tree = SumTree(hparams.memory_size)
        self.per_epsilon = hparams.per_epsilon
        self.per_alpha = hparams.per_alpha
        self.per_beta = hparams.per_beta
        self.per_beta_increment = hparams.per_beta_increment
        self.abs_error_clip = hparams.per_error_clip

    def store(self, _experience):
        """
        Store an experience in the binary tree
        :param _experience:
        :return:
        """
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_error_clip

        self.tree.add(max_priority, _experience)

    def sample(self, n):
        """
        Return a batch of size n
        :param n:
        :return: Experience-Indexes, Experiences, importance_sampling_weights
        """
        memory_b = []
        b_idx, b_is_weights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / n
        self.per_beta = np.min([1., self.per_beta + self.per_beta_increment])

        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        if p_min == 0:
            # Initialisierungsproblem, p_min auf Epsilon setzen, da Epsilon die minimale Wahrscheinlichkeit ist
            p_min = self.per_epsilon
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
        """
        Update Importance Weights
        :param tree_idx:
        :param abs_errors:
        :return:
        """
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
        """
        Add an experience.
        :param priority:
        :param data:
        :return:
        """
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # mit neuen Einträgen überschreiben
            self.data_pointer = 0

    def update(self, tree_index, priority):
        """
        Update the importance of a batch of experiences.
        :param tree_index:
        :param priority:
        :return:
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        """
        Get Leaf-Node for weight v
        :param v:
        :return:
        """
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
    """
    Simple helper to store Experiences.
    """

    def __init__(self, _state, _reward, _done, _next_state, _action):
        self.state = np.array(_state)
        self.reward = _reward
        self.done = _done
        self.next_state = np.array(_next_state)
        self.action = _action


class DDDQAgent:
    """
    Deep Duelling Double Q-Learning Agent
    """

    def __init__(self, _session, hparams):
        self.sess = _session
        self.epsilon = hparams.epsilon_start
        self.eps_stop = hparams.epsilon_stop
        self.eps_decay = hparams.epsilon_decay
        self.batch_size = hparams.batch_size
        self.gamma = hparams.gamma
        self.curr_loss = -1
        self.curr_vs = []
        self.curr_as = []
        self.possible_actions = np.identity(hparams.action_size, dtype=int).tolist()
        self.dq_network = DDDQNetwork(hparams, name='DQNetwork')
        self.target_network = DDDQNetwork(hparams, name='TargetNetwork')
        if hparams.use_importance_sampling:
            self.memory = ImportanceSamplingMemory(hparams)
        else:
            self.memory = UniformSamplingMemory(hparams)
        self.sess.run(tf.global_variables_initializer())
        # self.update_target()
        self.writer = tf.summary.FileWriter(hparams.log_dir, self.sess.graph)
        with tf.name_scope('Belohnungen'):
            # Belohnung
            self.v_rewards = tf.placeholder(tf.float32, shape=None, name='v_Belohnungen')
            self.reward_histo = tf.summary.histogram('Verteilung', self.v_rewards)
            self.v_rewards_sum = tf.math.reduce_sum(self.v_rewards)
            self.summary_reward_sum = tf.summary.scalar('Summe', self.v_rewards_sum)
            self.v_rewards_min = tf.math.reduce_min(self.v_rewards)
            self.summary_reward_min = tf.summary.scalar('Minimum', self.v_rewards_min)

            # Validierungs Belohnung
            self.v_val_rewards = tf.placeholder(tf.float32, shape=None, name='v_val_Belohnungen')
            self.val_reward_histo = tf.summary.histogram('Val_Verteilung', self.v_val_rewards)
            self.v_val_rewards_sum = tf.math.reduce_sum(self.v_val_rewards)
            self.summary_val_reward_sum = tf.summary.scalar('Val_Summe', self.v_val_rewards_sum)

        with tf.name_scope('Values'):
            # StateValues
            self.v_vs = tf.placeholder(tf.float32, shape=None, name='v_StateValues')
            self.summary_v_values = tf.summary.histogram('StateValues', self.v_vs)
            self.v_mean_v = tf.math.reduce_mean(self.v_vs)
            self.summary_mean_v_value = tf.summary.scalar('StateValue', self.v_mean_v)

            # AdvantageValues
            self.v_as = tf.placeholder(tf.float32, shape=None, name='v_AdvantageValues')
            self.v_best_as = tf.math.reduce_max(self.v_as, axis=0)
            self.summary_a_values = tf.summary.histogram('AdvantageValues', self.v_as)
            self.summary_best_a_values = tf.summary.histogram('BestAdvantageValues', self.v_best_as)
            self.v_mean_a = tf.math.reduce_mean(self.v_best_as)
            self.v_std_a = tf.math.reduce_std(self.v_best_as)
            self.summary_mean_a_value = tf.summary.scalar('BestAdvantageValueMean', self.v_mean_a)
            self.summary_std_a_value = tf.summary.scalar('BestAdvantageValueVariance', self.v_std_a)

            # Loss
            self.v_loss = tf.placeholder(tf.float32, shape=None, name='Loss')
            self.summary_loss = tf.summary.scalar('Loss', self.v_loss)

        with tf.name_scope('Hyperparameter'):
            # Epsilon
            self.v_epsilon = tf.placeholder(tf.float32, shape=None, name='v_Epsilon')
            self.summary_epsilon = tf.summary.scalar('Epsilon', self.v_epsilon)
            # Beta
            self.v_beta = tf.placeholder(tf.float32, shape=None, name='v_Beta')
            self.summary_beta = tf.summary.scalar('Beta', self.v_beta)
            # TargetUpdates
            self.v_target_updates = tf.placeholder(tf.float32, shape=None, name='v_TargetUpdates')
            self.summary_target_update = tf.summary.scalar('TargetUpdates', self.v_target_updates)

        with tf.name_scope('Bestand'):
            # Bestand
            self.v_bestand = tf.placeholder(tf.float32, shape=None, name='v_Bestand')
            self.v_bestand_max = tf.math.reduce_max(self.v_bestand)
            self.summary_bestand_max = tf.summary.scalar('Maximum', self.v_bestand_max)

            self.v_bestand_mean = tf.math.reduce_mean(self.v_bestand)
            self.summary_bestand_mean = tf.summary.scalar('Durchschnitt', self.v_bestand_mean)

            # Actions
            self.v_actions = tf.placeholder(tf.float32, shape=None, name='v_Aktionen')
            self.action_histo = tf.summary.histogram('Aktionen', self.v_actions)
            self.v_actions_sum = tf.math.reduce_sum(self.v_actions)
            self.summary_actions_sum = tf.summary.scalar('Bestellmenge', self.v_actions_sum)

            # Validation Actions
            self.v_val_actions = tf.placeholder(tf.float32, shape=None, name='v_val_Aktionen')
            self.v_val_actions_sum = tf.math.reduce_sum(self.v_val_actions)

        with tf.name_scope('Bewegungen'):
            # Absatz
            self.v_absatz = tf.placeholder(tf.float32, shape=None, name='v_Absatz')
            self.v_absatz_sum = tf.math.reduce_sum(self.v_absatz)
            self.summary_absatz_sum = tf.summary.scalar('Absatz', self.v_absatz_sum)

            # Validation Absatz
            self.v_val_absatz = tf.placeholder(tf.float32, shape=None, name='v_val_Absatz')
            self.v_val_absatz_sum = tf.math.reduce_sum(self.v_val_absatz)

            # Abschriften
            self.v_abschriften = tf.placeholder(tf.float32, shape=None, name='v_Abschriften')
            self.v_abschriften_sum = tf.math.reduce_sum(self.v_abschriften)
            self.summary_abschriften_sum = tf.summary.scalar('Abschriften', self.v_abschriften_sum)

            self.v_abschrift_proz = tf.math.divide(self.v_abschriften_sum, self.v_actions_sum)
            self.summary_abschrift_proz = tf.summary.scalar('AbschriftProzent', self.v_abschrift_proz)

            # Validation Abschriften
            self.v_val_abschriften = tf.placeholder(tf.float32, shape=None, name='v_val_Abschriften')
            self.v_val_abschriften_sum = tf.math.reduce_sum(self.v_val_abschriften)

            self.v_val_abschrift_proz = tf.math.divide(self.v_val_abschriften_sum, self.v_val_actions_sum)
            self.summary_val_abschrift_proz = tf.summary.scalar('Val_AbschriftProzent', self.v_val_abschrift_proz)

            # Fehlmenge
            self.v_fehlmenge = tf.placeholder(tf.float32, shape=None, name='v_Fehlmenge')
            self.v_fehlmenge_sum = tf.math.reduce_sum(self.v_fehlmenge)
            self.summary_fehlmenge_sum = tf.summary.scalar('Fehlmenge', self.v_fehlmenge_sum)

            self.v_fehlmenge_proz = tf.math.divide(self.v_fehlmenge_sum, self.v_absatz_sum)
            self.summary_fehlmenge_proz = tf.summary.scalar('FehlmengeProzent', self.v_fehlmenge_proz)

            # Validation Fehlmenge
            self.v_val_fehlmenge = tf.placeholder(tf.float32, shape=None, name='v_val_Fehlmenge')
            self.v_val_fehlmenge_sum = tf.math.reduce_sum(self.v_val_fehlmenge)

            self.v_val_fehlmenge_proz = tf.math.divide(self.v_val_fehlmenge_sum, self.v_val_absatz_sum)
            self.summary_val_fehlmenge_proz = tf.summary.scalar('Val_FehlmengeProzent', self.v_val_fehlmenge_proz)

        self.merged = tf.summary.merge_all()

    def act(self, _state):
        """
        Method to choose an action.
        Reduces Epsilon by the factor eps_decay every time.
        :param _state:
        :return:
        """
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_stop)
        if self.epsilon > np.random.rand():
            _action = random.choice(self.possible_actions)
        else:
            qs_of_state = self.sess.run(
                self.dq_network.output,
                feed_dict={
                    self.dq_network.inputs: np.expand_dims(_state, axis=0)
                })
            choice = np.argmax(qs_of_state)
            _action = self.possible_actions[int(choice)]
        return _action

    def update_target(self):
        """
        Copies the Parameters of the Action-Network to the Target-Network
        :return:
        """
        update_ops = update_target_graph('DQNetwork', 'TargetNetwork')
        self.sess.run(update_ops)

    def remember(self, _experience):
        """
        Stores an experience in an internal Replay-Buffer
        :param _experience:
        :return:
        """
        self.memory.store(_experience)

    def train(self):
        """
        Takes a batch from the replay-buffer and performs a network update
        :return:
        """
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
                self.dq_network.inputs: next_state_batch
            }
        )
        q_target_next_state = self.sess.run(
            self.target_network.output,
            feed_dict={
                self.target_network.inputs: next_state_batch
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
        _, loss, absolute_errors, state_vals, advantages = self.sess.run(
            [
                self.dq_network.optimizer,
                self.dq_network.loss,
                self.dq_network.absolute_errors,
                self.dq_network.value,
                self.dq_network.advantage,
            ],
            feed_dict={
                self.dq_network.inputs: state_batch,
                self.dq_network.target_q: target_qs_batch,
                self.dq_network.actions_: action_batch,
                self.dq_network.is_weights: is_weights_batch,
            }
        )
        self.memory.batch_update(tree_idx, absolute_errors)
        self.curr_loss = loss
        self.curr_vs = state_vals
        self.curr_as = advantages


# endregion


# region Asynchronous Advantage Actor Critic
class A3CNetwork:
    """
    Neuronales Netz des Actor-Critic.
    """

    def __init__(self, scope, _trainer, hparams):
        with tf.variable_scope(scope):
            if hparams.use_double_lstm:
                self.inputs = tf.placeholder(
                    shape=[None, hparams.time_steps, *hparams.state_size],
                    dtype=tf.float32,
                    name='Inputs')
                self.intermediate_lstm = tf.keras.layers.LSTM(
                    hparams.lstm_units,
                    return_sequences=True,
                    name='LSTM_1')(self.inputs)
                self.firsts = tf.keras.layers.LSTM(hparams.lstm_units, name='LSTM_2')(self.intermediate_lstm)
            elif hparams.use_lstm:
                self.inputs = tf.placeholder(
                    shape=[None, hparams.time_steps, *hparams.state_size],
                    dtype=tf.float32,
                    name='Inputs')
                self.firsts = tf.keras.layers.LSTM(hparams.lstm_units, name='LSTM')(self.inputs)
            else:
                self.inputs = tf.placeholder(shape=[None, *hparams.state_size], dtype=tf.float32, name='Inputs')
                self.firsts = tf.keras.layers.Flatten(name='FlatInputs')(self.inputs)
            self.fc_main = tf.keras.layers.Dense(
                hparams.main_size,
                activation=hparams.main_activation,
                kernel_regularizer=hparams.main_regularizer
            )(self.firsts)
            self.dropout_main = tf.keras.layers.Dropout(hparams.drop_out_rate)(self.fc_main)
            self.fc_policy = tf.keras.layers.Dense(
                hparams.avantage_size,
                activation=hparams.advantage_activation,
                kernel_regularizer=hparams.advantage_regularizer
            )(self.dropout_main)
            self.dropout_policy = tf.keras.layers.Dropout(hparams.drop_out_rate)(self.fc_policy)
            self.fc_value = tf.keras.layers.Dense(
                hparams.value_size,
                activation=hparams.value_activation,
                kernel_regularizer=hparams.value_regularizer
            )(self.dropout_main)
            self.dropout_value = tf.keras.layers.Dropout(hparams.drop_out_rate)(self.fc_value)
            self.policy = tf.keras.layers.Dense(
                hparams.action_size,
                activation=tf.keras.activations.softmax,
                kernel_initializer=normalized_columns_initializer(0.01),
                bias_initializer=None
            )(self.dropout_policy)
            self.policy += 1e-7
            # verhindert NANs wenn log(policy) berechnet wird. Führt sonst zum Abbruch des Worker-Threads
            self.value = tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=normalized_columns_initializer(1.0),
                bias_initializer=None
            )(self.dropout_value)

        if scope != 'global':
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, hparams.action_size, dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

            # Loss functions
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
            self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
            self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = _trainer.apply_gradients(zip(grads, global_vars))


class Worker:
    """
    Workerclass for Actor Critic
    """

    def __init__(self, env, name, _trainer, _global_episodes, hparams, use_as_validator=False):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = hparams.model_dir
        self.gamma = hparams.gamma
        self.trainer = _trainer
        self.global_episodes = _global_episodes
        self.max_episodes = hparams.episodes
        self.use_as_validator = use_as_validator
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(os.path.join(hparams.log_dir, "train_" + str(self.number)))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = A3CNetwork(self.name, _trainer, hparams)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = np.identity(hparams.action_size, dtype=bool).tolist()
        self.env = env
        self.rewards_plus = None
        self.value_plus = None
        self.hparams = hparams

    def train(self, rollout, _sess, bootstrap_value):
        if self.use_as_validator:
            return None, None, None, None, None
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        # next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, self.gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + self.gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, self.gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.array(list(observations)),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages}
        v_l, p_l, e_l, g_n, v_n, _ = _sess.run(
            [
                self.local_AC.value_loss,
                self.local_AC.policy_loss,
                self.local_AC.entropy,
                self.local_AC.grad_norms,
                self.local_AC.var_norms,
                self.local_AC.apply_grads
            ],
            feed_dict=feed_dict
        )
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, _sess, _coord, _saver):
        episode_count = _sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with _sess.as_default(), _sess.graph.as_default():
            while not _coord.should_stop():
                _sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                # episode_states = []
                episode_reward = 0
                episode_step_count = 0
                done = False
                state_op = StateOperator(self.hparams)
                state, info = self.env.reset()
                state_op.start(state)
                # episode_states.append(state_op.state)
                while not done:
                    # Take an action using probabilities from policy network output.
                    a_dist, v = _sess.run(
                        [self.local_AC.policy, self.local_AC.value],
                        feed_dict={self.local_AC.inputs: [state_op.state]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    reward, done, next_state = self.env.make_action(a)
                    state_op.add(next_state)
                    # episode_states.append(state_op.state)

                    episode_buffer.append([state_op.pre_state, a, reward, state_op.state, done, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += reward
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and not done:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        target_v = _sess.run(
                            self.local_AC.value,
                            feed_dict={self.local_AC.inputs: [state_op.state]}
                        )[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, _sess, target_v)
                        episode_buffer = []
                        _sess.run(self.update_local_ops)
                    if done:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, _sess, 0.0)

                # Periodically save model parameters and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        _saver.save(_sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    bestell_menge = self.env.statistics.actions().sum()
                    absatz_menge = self.env.statistics.absaetze().sum()
                    abschrift_menge = self.env.statistics.abschrift().sum()
                    fehlmenge = self.env.statistics.fehlmenge().sum()
                    max_bestand = self.env.statistics.bestand().max()
                    abschrift_quote = abschrift_menge / bestell_menge
                    fehl_quote = fehlmenge / absatz_menge
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    if not self.use_as_validator:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    summary.value.add(tag='Model/Bestellmenge', simple_value=float(bestell_menge))
                    summary.value.add(tag='Model/Absatz', simple_value=float(absatz_menge))
                    summary.value.add(tag='Model/Abschriften', simple_value=float(abschrift_menge))
                    summary.value.add(tag='Model/AbschriftQuote', simple_value=float(abschrift_quote))
                    summary.value.add(tag='Model/Fehlmenge', simple_value=float(fehlmenge))
                    summary.value.add(tag='Model/FehlmengeQuote', simple_value=float(fehl_quote))
                    summary.value.add(tag='Model/Höchstbestand', simple_value=float(max_bestand))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    _sess.run(self.increment)
                episode_count += 1
                if episode_count > self.max_episodes and self.name == 'worker_0':
                    _coord.request_stop()


# endregion


class Mensch:
    """
    Objekt, dass wie ein Mensch bestellen würde.
    """

    def __init__(self, hparams):
        self.sicherheitsaufschlag = hparams.sicherheitsaufschlag
        self.rundung = hparams.rundung
        self.zyklus = hparams.bestell_zyklus

    def act(self, state):
        """
        Methode zur Auswahl der optimalen Bestellmenge
        :param state:
        :return: action
        """
        state = state[0]
        bestand = state[0] * 8
        ose = state[4] * 10
        prediction = state[5:]
        prognose = prediction[0:self.zyklus].sum()
        optimale_menge = max(prognose - bestand, 0)
        #         if optimale_menge > 0:
        optimale_menge += self.sicherheitsaufschlag
        restmenge = optimale_menge % ose
        bruchmenge = optimale_menge / ose
        if 0 < bruchmenge < 1:
            bestellmenge = 1
        elif bruchmenge >= 1:
            if restmenge == 0:
                bestellmenge = bruchmenge
            else:
                if self.rundung == 'kleiner':
                    bestellmenge = np.floor(bruchmenge)
                elif self.rundung == 'groesser':
                    bestellmenge = np.ceil(bruchmenge)
                else:
                    if restmenge > ose / 2:
                        bestellmenge = np.ceil(bruchmenge)
                    else:
                        bestellmenge = np.floor(bruchmenge)
        else:
            bestellmenge = 0

        return bestellmenge
