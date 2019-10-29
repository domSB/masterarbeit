import tensorflow as tf
import numpy as np
import os
import threading
from time import sleep
from agents import Predictor, A3CNetwork, Worker
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays


# region Hyperparameter

warengruppe = 80
state_size = np.array([18])
gamma = .99  # discount rate for advantage estimation and reward discounting
load_model = False
model_path = os.path.join('files', 'models', 'A3C', '04eval' + str(warengruppe))
logging_path = os.path.join('files', 'logging', 'A3C', '04eval' + str(warengruppe))

simulation_params = {
    'InputDirectory': os.path.join('files', 'raw'),
    'OutputDirectory': os.path.join('files', 'prepared'),
    'ZielWarengruppen': [warengruppe],
    'StatStateCategoricals': {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6},
}

predictor_dir = os.path.join('files',  'models', 'PredictorV2', '02RegWG' + str(warengruppe))
available_weights = os.listdir(predictor_dir)
available_weights.sort()
predictor_path = os.path.join(predictor_dir, available_weights[-1])
# endregion

pipeline = DataPipeLine(**simulation_params)
simulation_data = pipeline.get_regression_data()
state_size[0] += simulation_data[2].shape[1]
train_data, test_data = split_np_arrays(*simulation_data, percentage=0.01)
predictor = Predictor()
predictor.build_model(
    dynamic_state_shape=simulation_data[1].shape[2],
    static_state_shape=simulation_data[2].shape[1]
)
predictor.load_from_weights(predictor_path)
print('Warengruppe: ', warengruppe)
print('Predicting',  end='')
pred = predictor.predict(
    {
        'dynamic_input': train_data[1],
        'static_input': train_data[2]
    }
)
print(' and done ;)')
# endregion
tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(logging_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-5)
    master_network = A3CNetwork('global', None, state_size)
    # num_workers = multiprocessing.cpu_count()
    num_workers = 8 * 4  # Arbeitsspeicher Restriktion
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
                global_episodes,
                state_size
            )
        )
    saver = tf.train.Saver(max_to_keep=1)

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
        sleep(0.2)
        worker_threads.append(t)
    coord.join(worker_threads)
    eingabe = input('Fertig? (j/n)')
    if eingabe == 'j':
        coord.request_stop()

