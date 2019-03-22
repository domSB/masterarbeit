import tensorflow as tf
import numpy as np
from collections import deque
import pickle
import random

### Modelhyperparameter
state_size = 5
action_size = 5
learning_rate = 0.0002
max_steps = 312

### Training Hyperparameter
total_epochs = 600
batch_size = 64

### Exploration Parameter
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

# Q Learning Parameter
gamma = 0.95

### Testing Parameter
single_product = 4

# Memory Hyperparameter
pretrain_lenght = batch_size * 10
memory_size = 10000

order_none = [1, 0, 0, 0, 0]
order_one = [0, 1, 0, 0, 0]
order_two = [0, 0, 1, 0, 0]
order_tree = [0, 0, 0, 1, 0]
order_four = [0, 0, 0, 0, 1]
possible_actions = [order_none, order_one, order_two, order_tree, order_four]

class StockSimulation:
    def __init__(self, sales, start_stock):
        assert type(sales) == np.ndarray, "Wrong type for sales"
        assert type(start_stock) == np.ndarray, "Wrong type for start_stock"
        self.sales = sales[:,single_product] # Zum üben nur 1 Produkt
        self.start_stock = start_stock[single_product]
        self.days = len(sales)
        self.current_day = 0
        self.current_product = 0
        self.sim_stock = start_stock[single_product].copy()
        self.product_count = 1

    def reset(self):
        self.sim_stock = self.start_stock.copy()
        self.current_day = 0
        self.current_product = 0
        self.sales_forecast = deque(maxlen=4)
        for i in range(4):
            self.sales_forecast.append(self.sales[i])
        new_state = np.append(self.start_stock, self.sales_forecast).reshape(5)
        return new_state

    def make_action(self, action):
        # assert len(action) == self.product_count, "len(actions) doesn't match lenght of product stock"
        assert self.current_day < self.days - 1 , "epoch is finished. Do Reset."
        action = np.array(action).astype(np.uint8)
        reward = 0.0
        self.sim_stock -= self.sales[self.current_day]
        if self.sim_stock < 0:
            reward -= -1
        if self.sim_stock > 0:
            reward += 1 / self.sim_stock
        self.sim_stock += np.argmax(action)
        
        if self.current_day + 4 < self.days:
            self.sales_forecast.append(self.sales[self.current_day + 4])
        else:
            self.sales_forecast.append(np.zeros(1).astype(int))
        new_state = np.append(self.sim_stock, self.sales_forecast).reshape(5)
        self.current_day += 1
        return reward, self.current_day == self.days - 1, new_state


class DQNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        super(DQNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name, reuse=False):
            # Platzhalter erstellen
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions")

            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """1. Dense Layer"""
            self.dense_1 = tf.layers.Dense(
                units = 32,
                activation = tf.nn.relu,
                name="dense_1"
                )(self.inputs_)

            """2. Dense Layer"""
            self.dense_2 = tf.layers.Dense(
                units = 64,
                activation = tf.nn.relu,
                name="dense_2"
                )(self.dense_1)

            """3. Dense Layer"""
            self.q_pred = tf.layers.Dense(
                units = 5,
                activation = tf.nn.softmax,
                name="dense_3"
                )(self.dense_2)


            # Q ist der verhergesagte Q Wert

            self.Q = tf.reduce_sum(tf.multiply(self.q_pred, self.actions_), axis = 1)

            # Loss zwischen vorhergesagtem Q und Q Target

            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)



class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size = batch_size, replace = False)
        return [self.buffer[i] for i in index]


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
    else:
        Qs = sess.run(net.q_pred, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})

        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
    return action, explore_probability

tf.reset_default_graph()

net = DQNetwork(state_size, action_size, learning_rate)

saver = tf.train.Saver()

with open("./data/sales.pickle", "rb") as file:
    sales = pickle.load(file)

with open("./data/inventory.pickle", "rb") as file:
    start_stock = pickle.load(file)

simulation = StockSimulation(sales, start_stock)

memory = Memory(max_size = memory_size)

for i in range(pretrain_lenght):
    if i == 0:
        state = simulation.reset()

    # Random Action
    action = random.choice(possible_actions)

    # Get the rewards
    reward, is_done, new_state = simulation.make_action(action)


    
    if is_done:
        memory.add((state, action, reward, new_state, is_done))
        state = simulation.reset()
    else:
        
        memory.add((state, action, reward, new_state, is_done))
        state = new_state

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./runs/3", sess.graph)
    writer.add_graph(tf.get_default_graph())
    tf.summary.scalar("Loss", net.loss)
    merged = tf.summary.merge_all()
    reward_pl = tf.placeholder(tf.float32)
    reward_summary = tf.summary.scalar("Reward", reward_pl)
    exploration_pl = tf.placeholder(tf.float32)
    exploration_summary = tf.summary.scalar("Exploration", exploration_pl)
    init = tf.global_variables_initializer()
    
    sess.run(init)

    decay_step = 0

    for epoch in range(total_epochs):
        step = 0
        epoch_rewards = []

        state = simulation.reset()

        while step < max_steps:
            step += 1
            decay_step += 1
            action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

            reward, is_done, new_state = simulation.make_action(action)
            epoch_rewards.append(reward)

            if is_done:
                
                step = max_steps
                total_reward = np.sum(epoch_rewards)
                feed = {reward_pl: total_reward}
                summary_str = sess.run(reward_summary, feed_dict=feed)
                writer.add_summary(summary_str, epoch)
                feed = {exploration_pl: explore_probability}
                summary_str_expl = sess.run(exploration_summary, feed_dict=feed)
                writer.add_summary(summary_str_expl, epoch)
                print(
                    'epoch: {}'.format(epoch),
                    'Total reward: {}'.format(total_reward),
                    'Training Loss: {:.4f}'.format(loss),
                    'Explore P: {:.4f}'.format(explore_probability)
                    )
                memory.add((state, action, reward, new_state, is_done))

            else:

                memory.add((state, action, reward, new_state, is_done))

                state = new_state

            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            new_states_mb = np.array([each[3] for each in batch])
            is_dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []

            Qs_new_state = sess.run(net.q_pred, feed_dict = {net.inputs_: new_states_mb})

            for i in range(len(batch)):
                terminal = is_dones_mb[i]

                if terminal:
                    target_Qs_batch.append(rewards_mb[i])
                else:
                    target = rewards_mb[i] + gamma * np.max(Qs_new_state[i])
                    target_Qs_batch.append(target)

            targets_mb = np.array([each for each in target_Qs_batch])

            loss, _ = sess.run([net.loss, net.optimizer], feed_dict={net.inputs_: states_mb,
                                                                                    net.target_Q: targets_mb,
                                                                                    net.actions_: actions_mb})
            summary = sess.run(merged, feed_dict={net.inputs_: states_mb,
                                                    net.target_Q: targets_mb,
                                                    net.actions_: actions_mb})
            writer.add_summary(summary, epoch)
            
            writer.flush()

        # Modell alle 5 epochn speichern
        if epoch % 5 == 0:
            try:
                save_path = saver.save(sess, "./model/checkpoints/3/model.ckpt")
                print("Model saved")
            except:
                print("Saving skipped")
