
from simulation import StockSimulation
from network import DQN

import os
import numpy as np
# import tensorflow as tf

import cProfile

""" Hyperparameters """
#region  Hyperparameter
do_train = True

use_saved_model = False

memory_size = 364*1000
gamma = 0.5
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9999
learning_rate = 0.0001
batch_size = 32
n_step = 64

epochs = 21

update_target_network = batch_size * 100

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

#endregion

""" Initialize Objects """
#region Initilize
try:
    os.environ["OS"] == "Windows_NT"
    # Bin am eigenen Desktop
    data_dir = 'F:/OneDrive/Dokumente/1 Universität - Master/6. Semester/Masterarbeit/Implementation/Echtdaten'
    #TODO: Daten auch auf dem Desktop im ./data Directory ablegen, für Einheitlichkeit.
except KeyError:
    # Bin auf der EC2 Linux Maschine 
    data_dir = './data'

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

if use_saved_model:
    agent.load()
    #TODO: Model laden ermöglichen. Derzeit Bug beim Laden. Eventuell nur Weights speichern.

#endregion

""" Training Loop """
#region Training Loop
if do_train:
    global_steps = 0
    stats = {"loss": [],"acc": [], "rew":[]}
    for epoch in range(epochs):
        state, info = simulation.reset()
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
                tf_summary = agent.sess.run(agent.merged)
                agent.writer.add_summary(tf_summary, epoch)
                if epoch % 10 == 0:
                    print("Epoche {}".format(epoch))
                    print("\tMean reard: {} --- Total Reward: {} --- EXP-EXP: {}".format(curr_mean_rew, curr_rew, agent.epsilon))
                    agent.save()
                    #TODO: Validate Model with a trial Period in a seperate Simulation
                break
    agent.writer.close()
    agent.sess.close()

#endregion
       

    
    
