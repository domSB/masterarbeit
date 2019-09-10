import random
import os
import datetime
from collections import deque

import numpy as np
import tensorflow as tf


class AgentOne:
    def __init__(self, 
                 memory_size, 
                 state_shape, 
                 action_space, 
                 gamma, 
                 learning_rate,
                 lr_decay,
                 batch_size, 
                 epsilon, 
                 epsilon_decay, 
                 epsilon_min, 
                 possible_actions, 
                 time_series_length,
                 run_description
                 ):
        self.memory_size = memory_size
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.possible_actions = possible_actions
        self.time_series_length = time_series_length
        self.memory = deque(maxlen=memory_size)
        self.model = self.create_model("Train")
        log_dir = './files/logging/AgentV1/' + run_description
        model_dir = './files/models/AgentV1/' + run_description
        if os.path.exists(log_dir):
            log_dir = './files/logging/AgentV1/' + datetime.datetime.today().date().__str__() + "-" \
                      + datetime.datetime.today().time().__str__()[:8].replace(":", ".")
            model_dir = './files/logging/AgentV1/' + datetime.datetime.today().date().__str__() + "-" \
                        + datetime.datetime.today().time().__str__()[:8].replace(":", ".")
        os.mkdir(log_dir)
        os.mkdir(model_dir)
        self.logdir = log_dir
        self.modeldir = model_dir
        self.target_model = self.create_model("Target")

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

            self.theo_bestand = tf.placeholder(tf.float32, shape=None, name="TheoBestand")
            self.fakt_bestand = tf.placeholder(tf.float32, shape=None, name="FaktBestand")
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

            self.theo_bestand_max = tf.get_variable("Bestand_Max", dtype=tf.float32, initializer=tf.constant(0.0))
            self.theo_bestand_max_op = self.theo_bestand_max.assign(tf.math.reduce_max(self.theo_bestand))

            self.theo_bestand_min = tf.get_variable("Bestand_Min", dtype=tf.float32, initializer=tf.constant(0.0))
            self.theo_bestand_min_op = self.theo_bestand_min.assign(tf.math.reduce_min(self.theo_bestand))

            self.theo_bestand_mean = tf.get_variable("Bestand_Mean", dtype=tf.float32, initializer=tf.constant(0.0))
            self.theo_bestand_mean_op = self.theo_bestand_mean.assign(tf.math.reduce_mean(self.theo_bestand))

            self.summary_theo_bestand_max = tf.summary.scalar("Max", self.theo_bestand_max_op)
            self.summary_theo_bestand_min = tf.summary.scalar("Min", self.theo_bestand_min_op)
            self.summary_theo_bestand_mean = tf.summary.scalar("Mean", self.theo_bestand_mean_op)

            self.summary_theo_bestand = tf.summary.histogram("TheoretischerBestand", self.theo_bestand)
            self.summary_fakt_bestand = tf.summary.histogram("FaktischerBestand", self.fakt_bestand)
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
                self.summary_theo_bestand,
                self.summary_fakt_bestand,
                self.summary_theo_bestand_max,
                self.summary_theo_bestand_min,
                self.summary_theo_bestand_mean,
                self.summary_actions,
                self.summary_epsilon
            ])
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

    def create_model(self, name):
        with tf.name_scope(name):
            inputs = tf.keras.Input(shape=(self.time_series_length, self.state_shape))
            x = tf.keras.layers.LSTM(
                256,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="LSTM"
            )(inputs)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(
                256,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="Dense_1"
            )(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(
                256,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="Dense_2"
            )(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(
                512,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="Dense_3"
            )(x)
            predictions = tf.keras.layers.Dense(self.action_space, activation='relu', name="Predictions")(x)
            model = tf.keras.Model(inputs=inputs, outputs=predictions)
            adam = tf.keras.optimizers.Adam(lr=self.learning_rate, decay=self.lr_decay)
            model.compile(optimizer=adam, loss='mse', metrics=["accuracy"])
        
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        self.epsilon *= self.epsilon_decay
        self.epsilon = np.max([self.epsilon, self.epsilon_min])

        samples = random.sample(self.memory, self.batch_size)

        states = [sample[0] for sample in samples]
        actions = [sample[1] for sample in samples]
        rewards = [sample[2] for sample in samples]
        new_states = [sample[3] for sample in samples]
        new_states = np.array(new_states)
        states = np.array(states)
        dones = [sample[4] for sample in samples]
        targets = self.target_model.predict(states)
        qs_new_states = self.target_model.predict(new_states)
        
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
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.sample(self.possible_actions, 1)[0]
        predictions = self.model.predict(state.reshape(1, self.time_series_length, self.state_shape))[0]
        return np.argmax(predictions)
    
    def save(self):
        self.target_model.save(os.path.join(self.modeldir, "model.h5"))
    
    def load(self, path):
        model = tf.keras.models.load_model(path, compile=False)
        adam = tf.keras.optimizers.Adam(lr=self.learning_rate, decay=self.lr_decay)
        model.compile(optimizer=adam, loss='mse', metrics=["accuracy"])
        self.target_model = model
        self.model = model


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif 3 <= epoch < 10:
        return 1e-4
    else:
        return 1e-5


class Predictor(object):
    def __init__(self):
        self.model = None

    def build_model(self, **_params):
        dynamic_inputs = tf.keras.Input(shape=(_params['time_steps'], _params['dynamic_state_shape']),
                                        name='dynamic_input')
        static_inputs = tf.keras.Input(shape=(_params['static_state_shape'],), name='static_input')
        dynamic_x = tf.keras.layers.LSTM(
            64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="LSTM_1",
            return_sequences=True
        )(dynamic_inputs)
        dynamic_x = tf.keras.layers.LSTM(
            64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="LSTM_2"
        )(dynamic_x)
        x = tf.keras.layers.concatenate([dynamic_x, static_inputs])
        x = tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="Dense_1"
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="Dense_2"
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="Dense_3"
        )(x)
        predictions = tf.keras.layers.Dense(_params['forecast_state'], activation='relu', name="predictions")(x)
        self.model = tf.keras.Model(inputs=[dynamic_inputs, static_inputs], outputs=predictions)
        rms = tf.keras.optimizers.RMSprop(lr=_params['learning_rate'])
        self.model.compile(
            optimizer=rms,
            loss='mean_squared_error',
            metrics=['mean_squared_error', 'mean_absolute_error']
        )

    def train(self, _dataset, _val_dataset, _params):
        if os.path.exists(os.path.join('files', 'logging', 'Predictor', _params['Name'])):
            name = datetime.datetime.now().__str__()
        else:
            name = _params['Name']
        os.mkdir(os.path.join('files', 'logging', 'Predictor', name))
        os.mkdir(os.path.join('files', 'models', 'Predictor', name))
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('files', 'logging', 'Predictor', name),
            histogram_freq=1,
            batch_size=32,
            write_graph=True,
            write_grads=True,
            update_freq='batch')
        nan_callback = tf.keras.callbacks.TerminateOnNaN()
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join('files', 'models', 'Predictor', name, 'weights.{epoch:02d}-{loss:.2f}.hdf5'),
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
        return y


class AgentTwo(object):
    def __init__(self, **kwargs):
        self.memory_size = kwargs['MemorySize']
        self.state_shape = kwargs['StateShape']
        self.action_space = kwargs['AktionSpace']
        self.gamma = kwargs['Gamma']
        self.learning_rate = kwargs['LearningRate']
        self.lr_decay = kwargs['LearningRateDecay']
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

            self.theo_bestand = tf.placeholder(tf.float32, shape=None, name="TheoBestand")
            self.fakt_bestand = tf.placeholder(tf.float32, shape=None, name="FaktBestand")
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

            self.theo_bestand_max = tf.get_variable("Bestand_Max", dtype=tf.float32, initializer=tf.constant(0.0))
            self.theo_bestand_max_op = self.theo_bestand_max.assign(tf.math.reduce_max(self.theo_bestand))

            self.theo_bestand_min = tf.get_variable("Bestand_Min", dtype=tf.float32, initializer=tf.constant(0.0))
            self.theo_bestand_min_op = self.theo_bestand_min.assign(tf.math.reduce_min(self.theo_bestand))

            self.theo_bestand_mean = tf.get_variable("Bestand_Mean", dtype=tf.float32, initializer=tf.constant(0.0))
            self.theo_bestand_mean_op = self.theo_bestand_mean.assign(tf.math.reduce_mean(self.theo_bestand))

            self.summary_theo_bestand_max = tf.summary.scalar("Max", self.theo_bestand_max_op)
            self.summary_theo_bestand_min = tf.summary.scalar("Min", self.theo_bestand_min_op)
            self.summary_theo_bestand_mean = tf.summary.scalar("Mean", self.theo_bestand_mean_op)

            self.summary_theo_bestand = tf.summary.histogram("TheoretischerBestand", self.theo_bestand)
            self.summary_fakt_bestand = tf.summary.histogram("FaktischerBestand", self.fakt_bestand)
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
                self.summary_theo_bestand,
                self.summary_fakt_bestand,
                self.summary_theo_bestand_max,
                self.summary_theo_bestand_min,
                self.summary_theo_bestand_mean,
                self.summary_actions,
                self.summary_epsilon
            ])
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

    def create_model(self, name):
        with tf.name_scope(name):
            inputs = tf.keras.Input(shape=(self.state_shape, ))
            x = tf.keras.layers.Dropout(0.2)(inputs)
            x = tf.keras.layers.Dense(
                64,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="Dense_1"
            )(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(
                64,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="Dense_2"
            )(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(
                128,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="Dense_3"
            )(x)
            predictions = tf.keras.layers.Dense(self.action_space, activation='relu', name="Predictions")(x)
            model = tf.keras.Model(inputs=inputs, outputs=predictions)
            adam = tf.keras.optimizers.Adam(lr=self.learning_rate, decay=self.lr_decay)
            model.compile(optimizer=adam, loss='mse', metrics=["accuracy"])

        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        self.epsilon *= self.epsilon_decay
        self.epsilon = np.max([self.epsilon, self.epsilon_min])

        samples = random.sample(self.memory, self.batch_size)

        states = [sample[0] for sample in samples]
        actions = [sample[1] for sample in samples]
        rewards = [sample[2] for sample in samples]
        new_states = [sample[3] for sample in samples]
        new_states = np.array(new_states)
        states = np.array(states)
        dones = [sample[4] for sample in samples]
        targets = self.target_model.predict(states)
        qs_new_states = self.target_model.predict(new_states)

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
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.sample(self.possible_actions, 1)[0]
        predictions = self.model.predict(state.reshape(1, self.state_shape))[0]
        return np.argmax(predictions)

    def save(self):
        self.target_model.save(os.path.join(self.modeldir, "model.h5"))

    def load(self, path):
        model = tf.keras.models.load_model(path, compile=False)
        adam = tf.keras.optimizers.Adam(lr=self.learning_rate, decay=self.lr_decay)
        model.compile(optimizer=adam, loss='mse', metrics=["accuracy"])
        self.target_model = model
        self.model = model