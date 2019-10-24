import tensorflow as tf
import numpy as np
import scipy.signal
import os
import threading
from time import sleep
from agents import Predictor
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays


def normalized_columns_initializer(std=1.0):
    """ Vorgeschlagen von Arthur Juliani"""
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, _gamma):
    return scipy.signal.lfilter([1], [1, -_gamma], x[::-1], axis=0)[::-1]


class Environment:
    def __init__(self):
        self.steps = 0
        pass

    def reset(self):
        self.steps = 0
        return np.random.random((3, 3))

    def make_action(self, action):
        assert self.steps < 10
        self.steps += 1
        return np.random.random((3, 3)), action / 4, self.steps == 10


class A3CNetwork:
    def __init__(self, scope, _trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, 65], dtype=tf.float32)
            self.hidden = tf.keras.layers.Dense(
                512,
                activation=tf.keras.activations.elu
            )(self.inputs)
            self.policy = tf.keras.layers.Dense(
                6,
                activation=tf.keras.activations.softmax,
                kernel_initializer=normalized_columns_initializer(0.01),
                bias_initializer=None
            )(self.hidden)
            self.value = tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=normalized_columns_initializer(1.0),
                bias_initializer=None
            )(self.hidden)

        if scope != 'global':
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, 6, dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

            # Loss functions
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
            self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
            self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = _trainer.apply_gradients(zip(grads, global_vars))


class Worker:
    def __init__(self, env, name, _trainer, _model_path, _logging_path, _global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = _model_path
        self.trainer = _trainer
        self.global_episodes = _global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(os.path.join(_logging_path, "train_" + str(self.number)))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = A3CNetwork(self.name, _trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = np.identity(4, dtype=bool).tolist()
        self.env = env
        self.rewards_plus = None
        self.value_plus = None

    def train(self, rollout, _sess, _gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        # next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, _gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + _gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, _gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.array(list(observations)),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages}
        v_l, p_l, e_l, g_n, v_n, _ = _sess.run(
            [
                self.local_AC.value_loss,
                self.local_AC.policy_loss,
                self.local_AC.entropy,
                self.local_AC.grad_norms,
                self.local_AC.var_norms,
                self.local_AC.apply_grads
            ],
            feed_dict=feed_dict
        )
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, _gamma, _sess, _coord, _saver):
        episode_count = _sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with _sess.as_default(), _sess.graph.as_default():
            while not _coord.should_stop():
                _sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_states = []
                episode_reward = 0
                episode_step_count = 0
                done = False
                state, info = self.env.reset()
                episode_states.append(state)
                while not done:
                    # Take an action using probabilities from policy network output.
                    a_dist, v = _sess.run(
                        [self.local_AC.policy, self.local_AC.value],
                        feed_dict={self.local_AC.inputs: [state]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    reward, done, next_state = self.env.make_action(a)
                    episode_states.append(next_state)

                    episode_buffer.append([state, a, reward, next_state, done, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += reward
                    state = next_state
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and not done:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        target_v = _sess.run(
                            self.local_AC.value,
                            feed_dict={self.local_AC.inputs: [state]}
                        )[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, _sess, _gamma, target_v)
                        episode_buffer = []
                        _sess.run(self.update_local_ops)
                    if done:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, _sess, _gamma, 0.0)

                # Periodically save model parameters and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        _saver.save(_sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    _sess.run(self.increment)
                episode_count += 1


# region Hyperparameter
gamma = .99  # discount rate for advantage estimation and reward discounting
load_model = False
model_path = os.path.join('files', 'models', 'A3C', '8')
logging_path = os.path.join('files', 'logging', 'A3C', '8')

simulation_params = {
    'InputDirectory': os.path.join('files', 'raw'),
    'OutputDirectory': os.path.join('files', 'prepared'),
    'ZielWarengruppen': [71],
    'StatStateCategoricals': {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6},
}

predictor_path = os.path.join('files', 'models', 'PredictorV2', '01RegWG71', 'weights.30-0.21.hdf5')
# endregion

pipeline = DataPipeLine(**simulation_params)
simulation_data = pipeline.get_regression_data()
train_data, test_data = split_np_arrays(*simulation_data, percentage=0.01)
predictor = Predictor()
predictor.build_model(
    dynamic_state_shape=simulation_data[1].shape[2],
    static_state_shape=simulation_data[2].shape[1]
)
predictor.load_from_weights(predictor_path)
print('Predicting')
pred = predictor.predict(
    {
        'dynamic_input': train_data[1],
        'static_input': train_data[2]
    }
)
print('Predicted')
# endregion
tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(logging_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = A3CNetwork('global', None)
    # num_workers = multiprocessing.cpu_count()
    num_workers = 8  # Arbeitsspeicher Restriktion
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(
            Worker(
                StockSimulation(train_data, pred, 2, 'Bestandsreichweite'),
                i,
                trainer,
                model_path,
                logging_path,
                global_episodes
            )
        )
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma, sess, coord, saver)
        t = threading.Thread(target=worker_work)
        t.start()
        sleep(1)
        worker_threads.append(t)
    coord.join(worker_threads)
    eingabe = input('Fertig? (j/n)')
    if eingabe == 'j':
        coord.request_stop()

