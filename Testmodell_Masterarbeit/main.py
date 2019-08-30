
from simulation import StockSimulation
from network import DQN

import os

# import cProfile

""" Hyperparameters """
# region  Hyperparameter
do_train = True
use_model_path = os.path.join('model', '2019-06-28-15.06.17', 'model.h5')
use_saved_model = False
use_pickled = True
save_pickled = False
simulation_group = 'Time'

epochs = 1500
memory_size = 364*2*200
gamma = 0.7
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.999
learning_rate = 0.01
lr_decay = 0.01/epochs
batch_size = 64
n_step = 128

update_target_network = n_step * 25

state_shape = 100
action_space = 6

time_series_lenght = 6

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
    order_five
    ]

    # order_six,
    # order_seven,
    # order_eight,
    # order_nine
# endregion

""" Initialize Objects """
# region Initilize
data_dir = 'data'

simulation = StockSimulation(data_dir, time_series_lenght, use_pickled, save_pickled, True, simulation_group)

validator = StockSimulation(data_dir, time_series_lenght, use_pickled, save_pickled, False, simulation_group)

agent = DQN(
    memory_size, 
    state_shape, 
    action_space, 
    gamma,
    learning_rate,
    lr_decay,
    batch_size, 
    epsilon, 
    epsilon_decay, 
    epsilon_min, 
    possible_actions, 
    time_series_lenght
    )

if use_saved_model:
    agent.load(use_model_path)

# endregion

""" Training Loop """
# region Training Loop


def train():
    global_steps = 0
    stats = {"loss": [], "acc": [], "rew": []}
    # TODO: Memory Buffer initial füllen, vor dem Trainingsloog.
    for epoch in range(epochs):
        state, info = simulation.reset()
        val_state, _ = validator.reset()
        val_fertig = False
        current_rewards = []
        current_val_rewards = []
        current_actions = []
        while True:
            # Train
            action = agent.act(state)
            global_steps += 1
            reward, fertig, new_state = simulation.make_action(action)
            current_rewards.append(reward)
            current_actions.append(action)
            agent.remember(state, action, reward, new_state, fertig)

            # Validate
            if not val_fertig:  # Validation Zeitraum ggf. kürzer oder gleichlang
                val_action = agent.act(val_state)
                val_reward, val_fertig, new_val_state= validator.make_action(val_action)
                current_val_rewards.append(val_reward)
                val_state = new_val_state

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
                tf_summary = agent.sess.run(
                    agent.merged, 
                    feed_dict={
                        agent.loss: curr_loss, 
                        agent.accuracy: curr_acc,
                        agent.rewards: current_rewards,
                        agent.val_rewards: current_val_rewards,
                        agent.theo_bestand: simulation.stat_theo_bestand,
                        agent.fakt_bestand: simulation.stat_fakt_bestand,
                        agent.actions: current_actions,
                        agent.tf_epsilon: agent.epsilon
                        }
                    )
                agent.writer.add_summary(tf_summary, epoch)
                if epoch % 10 == 0:
                    print("Epoche {}".format(epoch))
                    # print("\tMean reard: {} --- Total Reward: {} --- EXP-EXP: {}".format(
                    # curr_mean_rew, curr_rew, agent.epsilon
                    # )
                    # )
                    agent.save()
                    # TODO: Validate Model with a trial Period in a seperate Simulation
                else:
                    print('.')
                break
    agent.writer.close()
    agent.sess.close()


# cProfile.run('train()', 'cpu_profile.pstat')
if do_train:
    train()
# endregion
