# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from simulation import StockSimulation

import random
from collections import deque
import pickle

import numpy as np
import pandas as pd

import keras


import cProfile





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
        return np.argmax(self.model.predict(state.reshape(1, state_shape))[0])


def load_dataframe(path):
    df = pd.read_csv(
        path, 
        names=["Zeile", "Datum", "Artikel", "Absatz", "Warengruppe", "Abteilung"], 
        header=0, 
        parse_dates=[1], 
        index_col=[1, 2],
        memory_map=True
        )
    df.dropna(how='any', inplace=True)
    df["Warengruppe"] = df["Warengruppe"].astype(np.uint8)
    # df = df.drop(columns=['Abteilung', 'Zeile'])
    # Warengruppen auswählen
    # 13 Frischmilch
    # 14 Joghurt
    # 69 Tabak
    # 8 Obst Allgemen

    warengruppen = [8, 13, 14, 69 ]
    df = df[df['Warengruppe'].isin(warengruppen)]
    for i, wg in enumerate(warengruppen):
        df.loc[df.Warengruppe == wg, "Warengruppe"] = i
    df["Datum"] = df.index.get_level_values('Datum')
    df["Artikel"] = df.index.get_level_values('Artikel').astype(np.int32)
    df["Wochentag"] = df["Datum"].apply(lambda x:x.dayofweek)
    df["Jahrestag"] = df["Datum"].apply(lambda x:x.dayofyear)
    df["Jahr"] = df["Datum"].apply(lambda x:x.year)
    # df = df.drop(columns=['Datum'])
    df = df.sort_index()
    
    # Fürs erste
    df["OrderLeadTime"] = 1
    
    test_data = df[df["Jahr"]==2019]
    train_data = df[df["Jahr"]==2018]
    return test_data, train_data

def load_simulation(train_data, test_data):

    simulation = StockSimulation(train_data, sample_produkte)
    test_env = StockSimulation(test_data, sample_produkte)

    return simulation, test_env

def main():
    test_data, train_data = load_dataframe('F:/OneDrive/Dokumente/1 Universität - Master/6. Semester/Masterarbeit/Implemenation/Echtdaten/3 absatz_altforweiler.csv')

    simulation, test_env = load_simulation(train_data, test_data)

    print("Laden fertig")
    agent = DQN()
    global_steps = 0
    for epoch in range(epochs):
        state = simulation.reset()
        current_rewards = []
        while True:
            action = agent.act(state)
            global_steps += 1
            reward, artikel_fertig, new_state, state_neuer_artikel, episode_fertig = simulation.make_action(action)
            current_rewards.append(reward)
            agent.remember(state, action, reward, new_state, episode_fertig)
            if global_steps % n_step == 0:
                agent.replay()
            
            if global_steps % update_target_network == 0:
                agent.target_train()

            if artikel_fertig:
                state = state_neuer_artikel
            else:
                state = new_state

            if episode_fertig:
                mean_reward = np.mean(current_rewards)
                sum_reward = np.sum(current_rewards)
                print("Epoche {}".format(epoch))
                print("\tMean reard: {} --- Total Reward: {} --- EXP-EXP: {}".format(mean_reward, sum_reward, epsilon))
                break

       
if __name__ == "__main__":
    memory_size = 12000
    gamma = 1
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9999
    learning_rate = 0.0001
    tau = 0.05
    batch_size = 512
    n_step = 64

    epochs = 30

    update_target_network = 1000

    sample_produkte = 50

    #single_product = 4

    state_shape = 11
    action_space = 10

    order_none = 0
    order_one = 1
    order_two = 2
    order_tree = 3
    order_four = 4
    order_five = 5
    order_six = 6
    order_seven = 7
    order_eight = 8
    order_nine = 9

    possible_actions = [
        order_none, 
        order_one, 
        order_two, 
        order_tree, 
        order_four, 
        order_five, 
        order_six, 
        order_seven, 
        order_eight, 
        order_nine
        ]
    main()
    
    
