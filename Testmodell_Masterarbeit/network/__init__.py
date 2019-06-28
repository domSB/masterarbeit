import random
import os
import datetime
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard

from tensorflow import summary, Variable, Session




class DQN:
    def __init__(self, 
                 memory_size, 
                 state_shape, 
                 action_space, 
                 gamma, 
                 learning_rate, 
                 batch_size, 
                 epsilon, 
                 epsilon_decay, 
                 epsilon_min, 
                 possible_actions, 
                 time_series_length
                 ):
        self.memory_size = memory_size
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.possible_actions = possible_actions
        self.time_series_length = time_series_length
        self.memory = deque(maxlen=memory_size)
        self.model = self.create_model("Train")
        self.logdir = "./logs/" + datetime.datetime.today().date().__str__() + "-" \
                      + datetime.datetime.today().time().__str__()[:8].replace(":", ".")
        self.modeldir = "./model/" + datetime.datetime.today().date().__str__() + "-" \
                      + datetime.datetime.today().time().__str__()[:8].replace(":", ".")
        os.mkdir(self.modeldir)
        self.target_model = self.create_model("Target")

        with tf.name_scope("Eigene_Variablen"):
            #Training
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
            self.summary_rewards = summary.histogram("Rewards", self.rewards)
            self.summary_reward = summary.scalar("Sum", self.reward_sum_op)
            self.summary_reward_mean = summary.scalar("Mean", self.reward_mean_op)
            self.summary_reward_max = summary.scalar("Max", self.reward_max_op)
            self.summary_reward_min = summary.scalar("Min", self.reward_min_op)
        with tf.name_scope("Val_Reward_Stats"):
            self.summary_val_rewards = summary.histogram("Rewards", self.val_rewards)
            self.summary_val_reward = summary.scalar("Sum", self.val_reward_sum_op)
            self.summary_val_reward_mean = summary.scalar("Mean", self.val_reward_mean_op)
            self.summary_val_reward_max = summary.scalar("Max", self.val_reward_max_op)
            self.summary_val_reward_min = summary.scalar("Min", self.val_reward_min_op)

        with tf.name_scope("Bestand_Stats"):
            self.summary_actions = summary.histogram("Actions", self.actions)
            self.summary_theo_bestand = summary.histogram("TheoretischerBestand", self.theo_bestand)
            self.summary_fakt_bestand = summary.histogram("FaktischerBestand", self.fakt_bestand)
        with tf.name_scope("Model_Stats"):
            self.summary_loss = summary.scalar("Loss", self.loss)
            self.summary_mse = summary.scalar("Accuracy", self.accuracy)
            self.summary_epsilon = summary.scalar("Epsilon", self.tf_epsilon)
        self.merged = summary.merge(
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
                self.summary_actions,
                self.summary_epsilon
            ])
        self.sess = Session(config=tf.ConfigProto(log_device_placement=False))
        self.writer = summary.FileWriter(self.logdir, self.sess.graph)

    def create_model(self, name):
        #TODO: Struktur dynamisch gestalten, damit eine Klasse für alle Tests nutzbar.
        with tf.name_scope(name):
            #TODO: Weigth Regulization und Dropout-Layer einfügen
            inputs = Input(shape=(self.time_series_length, self.state_shape))
            x = LSTM(32, activation='relu', name="LSTM")(inputs)
            x = Dense(32, activation='relu', name="Dense_1")(x)
            x = Dense(64, activation='relu', name="Dense_2")(x)
            predictions = Dense(self.action_space, activation='relu', name="Predictions")(x)
            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer=RMSprop(lr=self.learning_rate), loss='mse', metrics=["accuracy"])
        
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
        model = tf.keras.models.load_model(path)
        self.target_model = model
        self.model = model





