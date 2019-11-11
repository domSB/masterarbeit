import random
import os
import datetime
from collections import deque

import numpy as np
import tensorflow as tf

from scipy.signal import lfilter

tf.get_logger().setLevel('ERROR')


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif 3 <= epoch < 10:
        return 1e-4
    else:
        return 1e-5


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


# region Double Deep Agent
class Agent(object):
    def __init__(self, **kwargs):
        self.memory_size = kwargs['MemorySize']
        self.article_state_shape = kwargs['ArticleStateShape']
        self.action_space = kwargs['AktionSpace']
        self.gamma = kwargs['Gamma']
        self.learning_rate = kwargs['LearningRate']
        # self.lr_decay = kwargs['LearningRateDecay']
        self.batch_size = kwargs['BatchSize']
        self.epsilon = kwargs['Epsilon']
        self.epsilon_decay = kwargs['EpsilonDecay']
        self.epsilon_min = kwargs['EpsilonMin']
        self.possible_actions = kwargs['PossibleActions']
        self.memory = deque(maxlen=self.memory_size)
        self.model = self.create_model("Train")
        self.target_model = self.create_model("Target")
        log_dir = './files/logging/AgentV2/' + kwargs['RunDescription']
        model_dir = './files/models/AgentV2/' + kwargs['RunDescription']
        if os.path.exists(log_dir):
            log_dir = './files/logging/AgentV2/' + datetime.datetime.today().date().__str__() + "-" \
                      + datetime.datetime.today().time().__str__()[:8].replace(":", ".")
            model_dir = './files/models/AgentV2/' + datetime.datetime.today().date().__str__() + "-" \
                        + datetime.datetime.today().time().__str__()[:8].replace(":", ".")
        os.mkdir(log_dir)
        os.mkdir(model_dir)
        self.logdir = log_dir
        self.modeldir = model_dir

        with tf.name_scope("Eigene_Variablen"):
            # Training
            self.rewards = tf.placeholder(tf.float32, shape=None, name="Rewards")

            self.reward_max = tf.get_variable("Max", dtype=tf.float32, initializer=tf.constant(0.0))
            self.reward_max_op = self.reward_max.assign(tf.math.reduce_max(self.rewards))

            self.reward_min = tf.get_variable("Min", dtype=tf.float32, initializer=tf.constant(0.0))
            self.reward_min_op = self.reward_min.assign(tf.math.reduce_min(self.rewards))

            self.reward_mean = tf.get_variable("Mean", dtype=tf.float32, initializer=tf.constant(0.0))
            self.reward_mean_op = self.reward_mean.assign(tf.math.reduce_mean(self.rewards))

            self.reward_sum = tf.get_variable("Sum", dtype=tf.float32, initializer=tf.constant(0.0))
            self.reward_sum_op = self.reward_sum.assign(tf.math.reduce_sum(self.rewards))

            # Validation
            self.val_rewards = tf.placeholder(tf.float32, shape=None, name="Val_Rewards")
            self.val_reward_max = tf.get_variable("Val_Max", dtype=tf.float32, initializer=tf.constant(0.0))
            self.val_reward_max_op = self.val_reward_max.assign(tf.math.reduce_max(self.val_rewards))

            self.val_reward_min = tf.get_variable("Val_Min", dtype=tf.float32, initializer=tf.constant(0.0))
            self.val_reward_min_op = self.val_reward_min.assign(tf.math.reduce_min(self.val_rewards))

            self.val_reward_mean = tf.get_variable("Val_Mean", dtype=tf.float32, initializer=tf.constant(0.0))
            self.val_reward_mean_op = self.val_reward_mean.assign(tf.math.reduce_mean(self.val_rewards))

            self.val_reward_sum = tf.get_variable("Val_Sum", dtype=tf.float32, initializer=tf.constant(0.0))
            self.val_reward_sum_op = self.val_reward_sum.assign(tf.math.reduce_sum(self.val_rewards))

            self.loss = tf.placeholder(tf.float32, name="Loss")
            self.accuracy = tf.placeholder(tf.float32, name="Accuracy")

            self.bestand = tf.placeholder(tf.float32, shape=None, name="Bestand")
            self.fehlmenge = tf.placeholder(tf.float32, shape=None, name="Fehlmenge")
            self.abschrift = tf.placeholder(tf.float32, shape=None, name="Abschrift")
            self.actions = tf.placeholder(tf.float32, shape=None, name="Actions")
            self.tf_epsilon = tf.placeholder(tf.float32, shape=None, name="Epsilon")
        with tf.name_scope("Reward_Stats"):
            self.summary_rewards = tf.summary.histogram("Rewards", self.rewards)
            self.summary_reward = tf.summary.scalar("Sum", self.reward_sum_op)
            self.summary_reward_mean = tf.summary.scalar("Mean", self.reward_mean_op)
            self.summary_reward_max = tf.summary.scalar("Max", self.reward_max_op)
            self.summary_reward_min = tf.summary.scalar("Min", self.reward_min_op)
        with tf.name_scope("Val_Reward_Stats"):
            self.summary_val_rewards = tf.summary.histogram("Rewards", self.val_rewards)
            self.summary_val_reward = tf.summary.scalar("Sum", self.val_reward_sum_op)
            self.summary_val_reward_mean = tf.summary.scalar("Mean", self.val_reward_mean_op)
            self.summary_val_reward_max = tf.summary.scalar("Max", self.val_reward_max_op)
            self.summary_val_reward_min = tf.summary.scalar("Min", self.val_reward_min_op)

        with tf.name_scope("Bestand_Stats"):
            self.summary_actions = tf.summary.histogram("Actions", self.actions)

            self.bestand_max = tf.get_variable("Bestand_Max", dtype=tf.float32, initializer=tf.constant(0.0))
            self.bestand_max_op = self.bestand_max.assign(tf.math.reduce_max(self.bestand))

            self.fehlmenge_sum = tf.get_variable("Fehlmenge", dtype=tf.float32, initializer=tf.constant(0.0))
            self.fehlmenge_sum_op = self.fehlmenge_sum.assign(tf.math.reduce_sum(self.fehlmenge))

            self.abschrift_sum = tf.get_variable("Abgeschriebene_Menge", dtype=tf.float32, initializer=tf.constant(0.0))
            self.abschrift_sum_op = self.abschrift_sum.assign(tf.math.reduce_sum(self.abschrift))

            self.bestand_min = tf.get_variable("Bestand_Min", dtype=tf.float32, initializer=tf.constant(0.0))
            self.bestand_min_op = self.bestand_min.assign(tf.math.reduce_min(self.bestand))

            self.bestand_mean = tf.get_variable("Bestand_Mean", dtype=tf.float32, initializer=tf.constant(0.0))
            self.bestand_mean_op = self.bestand_mean.assign(tf.math.reduce_mean(self.bestand))

            self.summary_bestand_max = tf.summary.scalar("Max", self.bestand_max_op)
            self.summary_bestand_min = tf.summary.scalar("Min", self.bestand_min_op)
            self.summary_bestand_mean = tf.summary.scalar("Mean", self.bestand_mean_op)
            self.summary_fehlmenge_sum = tf.summary.scalar("Fehlmenge", self.fehlmenge_sum_op)
            self.summary_abschrift_sum = tf.summary.scalar("Abschrift", self.abschrift_sum_op)

            self.summary_bestand = tf.summary.histogram("Bestand", self.bestand)
        with tf.name_scope("Model_Stats"):
            self.summary_loss = tf.summary.scalar("Loss", self.loss)
            self.summary_mse = tf.summary.scalar("Accuracy", self.accuracy)
            self.summary_epsilon = tf.summary.scalar("Epsilon", self.tf_epsilon)

        self.merged = tf.summary.merge(
            [
                self.summary_reward,
                self.summary_reward_mean,
                self.summary_reward_max,
                self.summary_reward_min,
                self.summary_val_reward,
                self.summary_val_reward_mean,
                self.summary_val_reward_max,
                self.summary_val_reward_min,
                self.summary_loss,
                self.summary_mse,
                self.summary_rewards,
                self.summary_val_rewards,
                self.summary_bestand,
                self.summary_fehlmenge_sum,
                self.summary_abschrift_sum,
                self.summary_bestand_max,
                self.summary_bestand_min,
                self.summary_bestand_mean,
                self.summary_actions,
                self.summary_epsilon
            ])
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

    def create_model(self, name):
        with tf.name_scope(name):
            inputs = tf.keras.Input(shape=(99, ), name='input')
            x = tf.keras.layers.Dense(
                32,
                activation='elu',
                kernel_regularizer=tf.keras.regularizers.l1(0.1),
                name="Dense_Sales"
            )(inputs)
            x = tf.keras.layers.Dense(
                64,
                activation='elu',
                kernel_regularizer=tf.keras.regularizers.l1(0.1),
                name="Dense_Concat"
            )(x)
            x = tf.keras.layers.Dense(
                128,
                activation='elu',
                kernel_regularizer=tf.keras.regularizers.l1(0.1),
                name="Dense_top"
            )(x)
            predictions = tf.keras.layers.Dense(self.action_space, activation=None, name="Predictions")(x)
            model = tf.keras.Model(inputs=inputs, outputs=predictions)
            adam = tf.keras.optimizers.Adam(self.learning_rate, epsilon=1e-8, clipvalue=0.5)
            model.compile(optimizer=adam, loss='mse', metrics=["accuracy"])

        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        new_states = np.array([sample[3] for sample in samples])
        dones = np.array([sample[4] for sample in samples])
        targets = self.target_model.predict(np.array(states))
        qs_new_states = self.target_model.predict(np.array(new_states))

        target_qs_batch = []
        for i in range(self.batch_size):
            terminal = dones[i]

            if terminal:
                updated_target = targets[i]
                updated_target[actions[i]] = rewards[i]
                target_qs_batch.append(updated_target)
            else:
                updated_target = targets[i]
                updated_target[actions[i]] = rewards[i] + self.gamma * np.max(qs_new_states[i])
                target_qs_batch.append(updated_target)

        targets = np.array([each for each in target_qs_batch])
        history = self.model.fit(states, targets, epochs=1, verbose=0, callbacks=[])
        return history.history

    def target_train(self):
        print('TU', end='')
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def act(self, state):

        self.epsilon *= self.epsilon_decay
        self.epsilon = np.max([self.epsilon, self.epsilon_min])

        if random.random() < self.epsilon:
            return random.sample(self.possible_actions, 1)[0]
        predictions = self.model.predict(np.expand_dims(state, axis=0))[0]
        return np.argmax(predictions)

    def save(self):
        self.target_model.save(os.path.join(self.modeldir, "model.h5"))

    def load(self, path):
        model = tf.keras.models.load_model(path, compile=False)
        adam = tf.keras.optimizers.Adam()
        model.compile(optimizer=adam, loss='mse', metrics=["accuracy"])
        self.target_model = model
        self.model = model
# endregion


# region Duelling Double Deep Agent
class DDDQNetwork:
    def __init__(self, _state_size, _action_size, _learning_rate, name):
        self.state_size = _state_size
        self.action_size = _action_size
        self.learning_rate = _learning_rate
        self.name = name

        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *_state_size], name='Inputs')

            self.is_weights = tf.placeholder(tf.float32, [None, 1], name='IS_Weights')

            self.actions_ = tf.placeholder(tf.float32, [None, _action_size], name='Actions')

            self.target_q = tf.placeholder(tf.float32, [None], name='Target')

            self.dense_one = tf.keras.layers.Dense(
                units=256,
                activation=tf.nn.relu,
                # kernel_regularizer=tf.keras.regularizers.l1(l=0.01),
                name='EingangsDense'
            )(self.inputs_)
            self.value_fc = tf.keras.layers.Dense(
                units=32,
                activation=tf.nn.relu,
                # kernel_regularizer=tf.keras.regularizers.l1(l=0.01),
                name='ValueFC'
            )(self.dense_one)
            self.value = tf.keras.layers.Dense(
                units=1,
                activation=None,
                name='Value'
            )(self.value_fc)
            self.advantage_fc = tf.keras.layers.Dense(
                units=32,
                activation=tf.nn.relu,
                # kernel_regularizer=tf.keras.regularizers.l1(l=0.01),
                name='AdvantageFC'
            )(self.dense_one)
            self.advantage = tf.keras.layers.Dense(
                units=self.action_size,
                activation=None,
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
    per_beta_increment = 0.000025
    abs_error_clip = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, _experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_error_clip

        self.tree.add(max_priority, _experience)

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
    def __init__(
            self,
            _eps_start,
            _eps_stop,
            _eps_decay,
            _learning_rate,
            _batch_size,
            _action_size,
            _state_size,
            _gamma,
            _memory_size,
            _session,
            _log_dir
    ):
        self.sess = _session
        self.epsilon = _eps_start
        self.eps_stop = _eps_stop
        self.eps_decay = _eps_decay
        self.batch_size = _batch_size
        self.gamma = _gamma
        self.curr_loss = -1
        self.possible_actions = np.identity(_action_size, dtype=int).tolist()
        self.dq_network = DDDQNetwork(_state_size, _action_size, _learning_rate, name='DQNetwork')
        self.target_network = DDDQNetwork(_state_size, _action_size, _learning_rate, name='TargetNetwork')
        self.memory = Memory(_memory_size)
        self.sess.run(tf.global_variables_initializer())
        # self.update_target()
        self.writer = tf.summary.FileWriter(_log_dir, self.sess.graph)
        with tf.name_scope('Modell'):
            self.v_rewards = tf.placeholder(tf.float32, shape=None, name='Belohnungen')
            self.reward_histo = tf.summary.histogram('Belohnungen', self.v_rewards)
            self.v_rewards_sum = tf.math.reduce_sum(self.v_rewards)
            self.summary_reward_sum = tf.summary.scalar('Gesamtbelohnung', self.v_rewards_sum)
            self.v_rewards_min = tf.math.reduce_min(self.v_rewards)
            self.summary_reward_min = tf.summary.scalar('MinBelohnung', self.v_rewards_min)
            self.v_reward_optimal = tf.placeholder(tf.float32, shape=None, name='OptimaleBelohnung')
            self.v_proz_reward = tf.math.divide(self.v_rewards_sum, self.v_reward_optimal)
            self.summary_opti_reward = tf.summary.scalar('OptBelohnung', self.v_reward_optimal)
            self.summary_proz_reward = tf.summary.scalar('ProzBelohnung', self.v_proz_reward)
            self.v_epsilon = tf.placeholder(tf.float32, shape=None, name='Epsilon')
            self.summary_epsilon = tf.summary.scalar('Epsilon', self.v_epsilon)
            self.v_loss = tf.placeholder(tf.float32, shape=None, name='Loss')
            self.summary_loss = tf.summary.scalar('Loss', self.v_loss)
            self.v_beta = tf.placeholder(tf.float32, shape=None, name='Beta')
            self.summary_beta = tf.summary.scalar('Beta', self.v_beta)
            self.v_target_updates = tf.placeholder(tf.float32, shape=None, name='TargetUpdates')
            self.summary_target_update = tf.summary.scalar('TargetUpdates', self.v_target_updates)
        with tf.name_scope('Bestand'):
            self.v_bestand = tf.placeholder(tf.float32, shape=None, name='Bestand')
            self.v_bestand_max = tf.math.reduce_max(self.v_bestand)
            self.summary_bestand_max = tf.summary.scalar('Maximum', self.v_bestand_max)
            self.v_bestand_mean = tf.math.reduce_mean(self.v_bestand)
            self.summary_bestand_mean = tf.summary.scalar('Durchschnitt', self.v_bestand_mean)
            self.v_actions = tf.placeholder(tf.float32, shape=None, name='Aktionen')
            self.action_histo = tf.summary.histogram('Aktionen', self.v_actions)
            self.v_actions_sum = tf.math.reduce_sum(self.v_actions)
            self.summary_actions_sum = tf.summary.scalar('Bestellmenge', self.v_actions_sum)
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
# endregion


# region Asynchronous Advantage Actor Critic
def normalized_columns_initializer(std=1.0):
    """ Vorgeschlagen von Arthur Juliani"""
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, _gamma):
    return lfilter([1], [1, -_gamma], x[::-1], axis=0)[::-1]


class A3CNetwork:
    def __init__(self, scope, _trainer, _state_size):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, *_state_size], dtype=tf.float32)
            self.hidden = tf.keras.layers.Dense(
                512,
                activation=tf.keras.activations.elu
            )(self.inputs)
            self.policy = tf.keras.layers.Dense(
                6,
                activation=tf.keras.activations.softmax,
                kernel_initializer=normalized_columns_initializer(0.01),
                bias_initializer=None
            )(self.hidden)
            self.policy += 1e-7
            # verhindert NANs wenn log(policy) berechnet wird. Führt zum Abbruch des Worker-Threads
            self.value = tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=normalized_columns_initializer(1.0),
                bias_initializer=None
            )(self.hidden)

        if scope != 'global':
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, 6, dtype=tf.float32)
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
    def __init__(self, env, name, _trainer, _model_path, _logging_path, _global_episodes, _state_size):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = _model_path
        self.trainer = _trainer
        self.global_episodes = _global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(os.path.join(_logging_path, "train_" + str(self.number)))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = A3CNetwork(self.name, _trainer, _state_size)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = np.identity(4, dtype=bool).tolist()
        self.env = env
        self.rewards_plus = None
        self.value_plus = None

    def train(self, rollout, _sess, _gamma, bootstrap_value):
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
        discounted_rewards = discount(self.rewards_plus, _gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + _gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, _gamma)

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

    def work(self, _gamma, _sess, _coord, _saver):
        episode_count = _sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with _sess.as_default(), _sess.graph.as_default():
            while not _coord.should_stop():
                _sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_states = []
                episode_reward = 0
                episode_step_count = 0
                done = False
                state, info = self.env.reset()
                episode_states.append(state)
                while not done:
                    # Take an action using probabilities from policy network output.
                    a_dist, v = _sess.run(
                        [self.local_AC.policy, self.local_AC.value],
                        feed_dict={self.local_AC.inputs: [state]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    reward, done, next_state = self.env.make_action(a)
                    episode_states.append(next_state)

                    episode_buffer.append([state, a, reward, next_state, done, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += reward
                    state = next_state
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and not done:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        target_v = _sess.run(
                            self.local_AC.value,
                            feed_dict={self.local_AC.inputs: [state]}
                        )[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, _sess, _gamma, target_v)
                        episode_buffer = []
                        _sess.run(self.update_local_ops)
                    if done:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, _sess, _gamma, 0.0)

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

# endregion
