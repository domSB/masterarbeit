import random
import os
import datetime
from collections import deque

import numpy as np
import tensorflow as tf


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
        y = np.array(y).reshape(6, 16)
        return y


class Agent(object):
    def __init__(self, **kwargs):
        self.memory_size = kwargs['MemorySize']
        self.article_state_shape = kwargs['ArticleStateShape']
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
            sales_input = tf.keras.Input(shape=(6, 16), name='predicted_sales')
            stock_input = tf.keras.Input(shape=(3,), name='current_stock')
            article_input = tf.keras.Input(shape=(self.article_state_shape,), name='article_info')
            flat_sales_input = tf.keras.layers.Flatten()(sales_input)
            sales_hidden = tf.keras.layers.Dense(
                32,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="Dense_Sales"
            )(flat_sales_input)
            article_hidden = tf.keras.layers.Dense(
                16,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="Dense_Article"
            )(article_input)
            x = tf.keras.layers.concatenate([sales_hidden, stock_input, article_hidden])
            x = tf.keras.layers.Dense(
                20,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="Dense_Concat"
            )(x)
            x = tf.keras.layers.Dense(
                20,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="Dense_middle"
            )(x)
            x = tf.keras.layers.Dense(
                20,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="Dense_top"
            )(x)
            predictions = tf.keras.layers.Dense(self.action_space, activation='relu', name="Predictions")(x)
            model = tf.keras.Model(inputs=[sales_input, stock_input, article_input], outputs=predictions)
            rms = tf.keras.optimizers.RMSprop(lr=self.learning_rate)
            model.compile(optimizer=rms, loss='mse', metrics=["accuracy"])

        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        self.epsilon *= self.epsilon_decay
        self.epsilon = np.max([self.epsilon, self.epsilon_min])

        samples = random.sample(self.memory, self.batch_size)

        predicted_sales = [sample[0]['predicted_sales'] for sample in samples]
        current_stock = [sample[0]['current_stock'] for sample in samples]
        article_info = [sample[0]['article_info'] for sample in samples]
        states = {
            'predicted_sales': np.array(predicted_sales),
            'current_stock': np.array(current_stock),
            'article_info': np.array(article_info)
        }
        actions = [sample[1] for sample in samples]
        rewards = [sample[2] for sample in samples]
        predicted_sales = [sample[3]['predicted_sales'] for sample in samples]
        current_stock = [sample[3]['current_stock'] for sample in samples]
        article_info = [sample[3]['article_info'] for sample in samples]
        new_states = {
            'predicted_sales': np.array(predicted_sales),
            'current_stock': np.array(current_stock),
            'article_info': np.array(article_info)
        }
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
        predictions = self.model.predict(
            {
                'predicted_sales': np.expand_dims(state['predicted_sales'], axis=0),
                'current_stock': np.expand_dims(state['current_stock'], axis=0),
                'article_info': np.expand_dims(state['article_info'], axis=0)
            }
        )[0]
        return np.argmax(predictions)

    def save(self):
        self.target_model.save(os.path.join(self.modeldir, "model.h5"))

    def load(self, path):
        model = tf.keras.models.load_model(path, compile=False)
        rms = tf.keras.optimizers.RMSprop(lr=self.learning_rate)
        model.compile(optimizer=rms, loss='mse', metrics=["accuracy"])
        self.target_model = model
        self.model = model
