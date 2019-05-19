
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
learning_rate = 0.0001
tau = 0.05
batch_size = 32
n_step = 64
log_frequency = 100 # jeder 100te n_step

epochs = 3

update_target_network = 1000

# sample_produkte = 20

state_shape = 24
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

data_dir = 'F:/OneDrive/Dokumente/1 Universität - Master/6. Semester/Masterarbeit/Implementation/Echtdaten'

simulation = StockSimulation(data_dir, time_series_lenght)

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
    state, info = simulation.reset()
    print(info)
    current_rewards = []
    while True:
        action = agent.act(state)
        global_steps += 1
        reward, fertig, new_state= simulation.make_action(action)
        current_rewards.append(reward)
        agent.remember(state, action, reward, new_state, fertig)
        if global_steps % n_step == 0:
            history = agent.replay()
            if history:
                curr_loss = history["loss"][0]
                curr_acc = history["acc"][0]
                stats["loss"].append(curr_loss)
                stats["acc"].append(curr_acc)
            
        if global_steps % update_target_network == 0:
            agent.target_train()

        state = new_state

        if fertig:
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
            agent.target_model.save("model/model.h5")
            break
agent.writer.close()
agent.sess.close()

       

    
    
