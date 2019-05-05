
from simulation import StockSimulation
from network import DQN

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf

import cProfile

memory_size = 1200000
gamma = 0.5
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9999
learning_rate = 0.00003
tau = 0.05
batch_size = 512
n_step = 64
log_frequency = 100 # jeder 100te n_step

epochs = 2

update_target_network = 1000

sample_produkte = 10

state_shape = 12
action_space = 10

time_series_lenght = 10

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

def load_weather(path):
    df = pd.read_csv(
        path, 
        index_col="date", 
        memory_map=True

        )
    df = df.drop(columns="Unnamed: 0")
    df = df.sort_index()
    # pd.to_datetime(df["Datum"]*24*3600, unit='s') liefert richtiges Datum

def load_prices(path):
    df = pd.read_csv(
        path, 
        names=["Zeile", "Preis","Artikelnummer","Datum"],
        header=0,
        index_col="Artikelnummer", 
        memory_map=True
        )
    df = df.sort_index()
    df = df.drop(columns=["Zeile"])
    return df

def load_sales(path):
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


data_dir = 'F:/OneDrive/Dokumente/1 Universität - Master/6. Semester/Masterarbeit/Implementation/Echtdaten'

prices = load_prices(os.path.join(data_dir, '3 preise_altforweiler.csv'))

test_data, train_data = load_sales(os.path.join(data_dir, '3 absatz_altforweiler.csv'))

simulation = StockSimulation(train_data, sample_produkte, prices, time_series_lenght)

agent = DQN(
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
    time_series_lenght
    )

global_steps = 0
stats = {"loss": [],"acc": [], "rew":[]}
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
            history = agent.replay()
            if history:
                curr_loss = history["loss"][0]
                curr_acc = history["acc"][0]
                stats["loss"].append(curr_loss)
                stats["acc"].append(curr_acc)
            
        if global_steps % update_target_network == 0:
            agent.target_train()

        if artikel_fertig:
            state = state_neuer_artikel
        else:
            state = new_state

        if episode_fertig:
            history = agent.replay()
            curr_loss = history["loss"][0]
            curr_acc = history["acc"][0]
            curr_rew = np.sum(current_rewards)
            curr_mean_rew = np.mean(current_rewards)
            agent.sess.run([agent.reward.assign(curr_rew), agent.reward_mean.assign(curr_mean_rew), agent.loss.assign(curr_loss), agent.accuracy.assign(curr_acc)])
            summary = agent.sess.run(agent.merged)
            agent.writer.add_summary(summary, epoch)
            print("Epoche {}".format(epoch))
            print("\tMean reard: {} --- Total Reward: {} --- EXP-EXP: {}".format(curr_mean_rew, curr_rew, agent.epsilon))
            break
agent.writer.close()
agent.sess.close()

       

    
    
