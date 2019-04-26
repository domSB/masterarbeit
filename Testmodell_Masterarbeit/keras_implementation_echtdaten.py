
from simulation import StockSimulation
from network import DQN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import cProfile

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
        possible_actions)
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
                    curr_rew = np.sum(current_rewards)
                    stats["loss"].append(curr_loss)
                    stats["acc"].append(curr_acc)
                    stats["rew"].append(curr_rew)
                    # print(curr_loss, curr_acc, curr_rew)
                    ergebnis = agent.sess.run([agent.reward.assign(curr_rew), agent.loss.assign(curr_loss), agent.accuracy.assign(curr_acc)])
                    print(ergebnis)
                    summary = agent.sess.run(agent.merged)
                    agent.writer.add_summary(summary)
            
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
    agent.writer.close()
    agent.sess.close()
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    for stat in stats:
        stats[stat] = np.array(stats[stat]).reshape(-1)
    ax1.plot(stats["loss"])
    ax1.set_title("Loss")
    ax2.plot(stats["acc"])
    ax2.set_title("Accuracy")
    ax3.plot(stats["rew"])
    ax3.set(title = "Reward", xlabel="Batches")
    
    plt.show()

       
if __name__ == "__main__":
    memory_size = 120000
    gamma = 0.5
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9999
    learning_rate = 0.0001
    tau = 0.05
    batch_size = 512
    n_step = 64

    epochs = 200

    update_target_network = 1000

    sample_produkte = 500

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
    
    
