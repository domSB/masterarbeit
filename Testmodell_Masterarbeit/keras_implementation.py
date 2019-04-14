# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import random
from collections import deque
import pickle

import numpy as np

import keras





class DQN:
    def __init__(self):
        self.memory = deque(maxlen=memory_size)
        self.model = self.create_model()

        self.target_model = self.create_model()

    def create_model(self):
        model = keras.Sequential()
        # model.add(keras.layers.Input(shape=(state_shape, )))
        model.add(keras.layers.Dense(24, input_dim = state_shape, activation="relu"))
        model.add(keras.layers.Dense(48, activation="relu"))
        model.add(keras.layers.Dense(96, activation="relu"))
        model.add(keras.layers.Dense(action_space)) # Qs werden nicht standardisiert, da keine Custom Loss Funtion. So funktioniert Standard MSE
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss="mse")
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)

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
        for i in range(batch_size):
            terminal = dones[i]

            if terminal:
                updated_target = targets[i]
                updated_target[actions[i]] = rewards[i]
                target_Qs_batch.append(updated_target)
            else:
                updated_target = targets[i]
                updated_target[actions[i]] = rewards[i] + gamma * np.max(Qs_new_states[i])
                target_Qs_batch.append(updated_target)

        targets = np.array([each for each in target_Qs_batch])

        self.model.fit(states, targets, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def act(self, state):
        global epsilon, epsilon_decay, epsilon_min
        epsilon *= epsilon_decay
        epsilon = np.max([epsilon, epsilon_min])
        if random.random() < epsilon:
            return random.sample(possible_actions, 1)[0]
        return np.argmax(self.model.predict(state.reshape(1, 5))[0])

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
            reward = np.expm1(self.sim_stock/2)
            # Nichtnegativität des Bestandes
            self.sim_stock = 0
        if self.sim_stock >= 0:
            reward = np.exp(-self.sim_stock/5)
        
        # Morgen:  Bestellung kommt an
        self.sim_stock += action

        if self.current_day + 4 < self.days:
            self.sales_forecast.append(self.sales[self.current_day + 4])
        else:
            self.sales_forecast.append(np.zeros(1).astype(int))
        new_state = np.append(self.sim_stock, self.sales_forecast).reshape(5)
        self.current_day += 1
        return reward, self.current_day == self.days - 1, new_state

def load_simulation():
    with open("./data/sales.pickle", "rb") as file:
        sales = pickle.load(file)

    with open("./data/inventory.pickle", "rb") as file:
        start_stock = pickle.load(file)

    simulation = StockSimulation(sales, start_stock)

    return simulation

def main():

    simulation = load_simulation()

    agent = DQN()
    global_steps = 0
    for epoch in range(epochs):
        state = simulation.reset()
        current_rewards = []
        while True:
            action = agent.act(state)
            global_steps += 1
            reward, done, new_state = simulation.make_action(action)
            current_rewards.append(reward)
            agent.remember(state, action, reward, new_state, done)
            agent.replay()
            if global_steps % update_target_network == 0:
                agent.target_train()

            state = new_state

            if done:
                mean_reward = np.mean(current_rewards)
                sum_reward = np.sum(current_rewards)
                print("Epoche {}".format(epoch))
                print("\tMean reard: {} --- Total Reward: {} --- EXP-EXP: {}".format(mean_reward, sum_reward, epsilon))
                break

       
if __name__ == "__main__":
    memory_size = 300
    gamma = 1
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9997
    learning_rate = 0.001
    tau = 0.05
    batch_size = 32

    epochs = 100

    update_target_network = 1000

    single_product = 4

    state_shape = 5
    action_space = 5

    order_none = 0
    order_one = 1
    order_two = 2
    order_tree = 3
    order_four = 4
    possible_actions = [order_none, order_one, order_two, order_tree, order_four]
    main()
