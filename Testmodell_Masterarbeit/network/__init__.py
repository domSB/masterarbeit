import numpy as np

import random
from collections import deque

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import tensorflow as tf

import datetime



class DQN:
    def __init__(self, memory_size, state_shape, action_space, gamma, learning_rate, batch_size, epsilon, epsilon_decay, epsilon_min, possible_actions):
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
        self.memory = deque(maxlen=memory_size)
        self.model = self.create_model()
        self.logdir = "./logs/" + datetime.datetime.today().date().__str__() + "-"+ datetime.datetime.today().time().__str__()[:8].replace(":",".")
        self.target_model = self.create_model()
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.reward = tf.Variable(0.0, trainable=False, name="vReward")
        self.loss = tf.Variable(0.0, trainable=False, name="vLoss")
        self.accuracy = tf.Variable(0.0, trainable=False, name="vMSE")
        self.summary_reward = tf.summary.scalar("Reward", self.reward)
        self.summary_loss = tf.summary.scalar("Loss", self.loss)
        self.summary_mse = tf.summary.scalar("Accuracy", self.accuracy)
        self.merged = tf.summary.merge([self.summary_reward, self.summary_loss, self.summary_mse])

    def create_model(self):
        model = Sequential()
        # model.add(keras.layers.Input(shape=(state_shape, )))
        model.add(Dense(24, input_dim = self.state_shape, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(96, activation="relu"))
        model.add(Dense(self.action_space)) # Qs werden nicht standardisiert, da keine Custom Loss Funtion. So funktioniert Standard MSE
        model.compile(
            optimizer = Adam(lr=self.learning_rate), 
            loss="mse",
            metrics=["acc"]
            )
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)

        states = [sample[0] for sample in samples]
        actions = [sample[1] for sample in samples]
        rewards = [sample[2] for sample in samples]
        new_states = [sample[3] for sample in samples]
        new_states = np.array(new_states)
        states = np.array(states)
        dones = [sample[4] for sample in samples]
        targets = self.target_model.predict(states)
        Qs_new_states = self.target_model.predict(new_states)
        
        target_Qs_batch = []
        for i in range(self.batch_size):
            terminal = dones[i]

            if terminal:
                updated_target = targets[i]
                updated_target[actions[i]] = rewards[i]
                target_Qs_batch.append(updated_target)
            else:
                updated_target = targets[i]
                updated_target[actions[i]] = rewards[i] + self.gamma * np.max(Qs_new_states[i])
                target_Qs_batch.append(updated_target)

        targets = np.array([each for each in target_Qs_batch])

        tensorbard = TensorBoard(
            log_dir= self.logdir, 
            histogram_freq=0, 
            batch_size=512, 
            write_graph=False, 
            write_grads=False, 
            write_images=False, 
            embeddings_freq=0, 
            embeddings_layer_names=None, 
            embeddings_metadata=None, 
            embeddings_data=None, 
            update_freq='epoch'
            ) # Klappt nicht, da Epoche 0 immer überschrieben wird
        history = self.model.fit(states, targets, epochs=1, verbose=0, callbacks=[])
        return history.history

    def target_train(self):
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
        return np.argmax(self.model.predict(state.reshape(1, self.state_shape))[0])


#import tensorflow as tf





#a = tf.constant(15, name="a")
#b = tf.constant(5, name="b")



#y = tf.placeholder(tf.int64, shape=[1])
#c = tf.Variable(y, name="Scalar")


#tf.reset_default_graph()
#writer = tf.summary.FileWriter("./logs/graphs")

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    curr_rew = 18
#    tensor_curr_rew = tf.convert_to_tensor(curr_rew, dtype=tf.int64, name="Wert_Reward")
#    summary = tf.summary.scalar("Rewart", tensor_curr_rew)
#    sum = sess.run(summary)
#    writer.add_summary(sum)
#    tf.reset_default_graph()

#    sess.run(tf.global_variables_initializer())
#    curr_rew = 20
#    tensor_curr_rew = tf.convert_to_tensor(curr_rew, dtype=tf.int64, name="Wert_Reward")
#    summary = tf.summary.scalar("Rewart", tensor_curr_rew)
#    sum = sess.run(summary)
#    writer.add_summary(sum)


#sess = tf.Session()

#writer = tf.summary.FileWriter("./logs/neu", sess.graph)

#reward = tf.Variable(0.0, trainable=False, name="vReward")
#loss = tf.Variable(0.0, trainable=False, name="vLoss")
#mse = tf.Variable(0.0, trainable=False, name="vMSE")
#summary_reward = tf.summary.scalar("Reward", reward)
#summary_loss = tf.summary.scalar("Loss", loss)
#summary_mse = tf.summary.scalar("MeanSquaredError", mse)
#merged = tf.summary.merge([summary_reward, summary_loss, summary_mse])
#a = 10
#sess.run([reward.assign(a), loss.assign(1), mse.assign(100)])
#summary = sess.run(merged)
#writer.add_summary(summary)





