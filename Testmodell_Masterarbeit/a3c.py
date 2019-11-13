import os
import threading
import multiprocessing
import tensorflow as tf

from time import sleep

from agents import Predictor, A3CNetwork, Worker
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from utils import Hyperparameter


# region Hyperparameter
hps = Hyperparameter(
    run_id=5,
    warengruppe=[55],
    detail_warengruppe=None,
    use_one_article=False,
    bestell_zyklus=3,
    state_size=[18],  # Zeitdimension, 6 Vorhersagen, Bestand, Abschriften, Fehlbestand
    action_size=6,
    learning_rate=0.001,
    memory_size=30000,
    episodes=10000,
    pretrain_episodes=5,
    batch_size=32,
    learn_step=1,
    max_tau=10000,
    epsilon_start=1,
    epsilon_stop=0.03,
    epsilon_decay=0.9999,
    gamma=0.95,
    do_train=True,
    reward_func='TDGewinn V2',
    sim_state_group=2,
    main_size=64,
    main_activation='tanh',
    main_regularizer='l2',
    value_size=32,
    value_activation='relu',
    value_regularizer='l2',
    avantage_size=32,
    advantage_activation='relu',
    advantage_regularizer='l2',
    per_epsilon=0.01,
    per_alpha=0.6,
    per_beta=0.4,
    per_beta_increment=0.00025,
    per_error_clip=1.0,
    use_importance_sampling=True,
    rest_laufzeit=14
)
load_model = False
dir_name = str(hps.run_id) + 'eval' + str(hps.warengruppe[0])
if hps.detail_warengruppe:
    dir_name = dir_name + '-' + str(hps.detail_warengruppe[0])
hps.add_hparam('model_dir', os.path.join('files', 'models', 'A3C', dir_name))
hps.add_hparam('log_dir', os.path.join('files', 'logging', 'A3C', dir_name))
if not os.path.exists(hps.log_dir):
    os.mkdir(hps.log_dir)
    os.mkdir(hps.model_dir)

predictor_dir = os.path.join('files',  'models', 'PredictorV2', '02RegWG' + str(hps.warengruppe[0]))
available_weights = os.listdir(predictor_dir)
available_weights.sort()
predictor_path = os.path.join(predictor_dir, available_weights[-1])
# endregion

pipeline = DataPipeLine(ZielWarengruppen=hps.warengruppe, DetailWarengruppe=hps.detail_warengruppe)
simulation_data = pipeline.get_regression_data()

if hps.sim_state_group > 1:
    state_size = hps.state_size
    state_size[0] += simulation_data[2].shape[1]
    hps.set_hparam('state_size', state_size)

train_data, test_data = split_np_arrays(*simulation_data)
predictor = Predictor()
predictor.build_model(
    dynamic_state_shape=simulation_data[1].shape[2],
    static_state_shape=simulation_data[2].shape[1]
)
predictor.load_from_weights(predictor_path)
print('Arbeite mit: ', dir_name)
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
hps.save(os.path.join(hps.log_dir, 'Hyperparameter.yaml'))

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(hps.learning_rate)
    master_network = A3CNetwork('global', None, hps)
    num_workers = multiprocessing.cpu_count()
    workers = []
    # Create worker classes
    for i in range(num_workers-1):
        workers.append(
            Worker(
                StockSimulation(train_data, pred, hps),
                i,
                trainer,
                global_episodes,
                hps
            )
        )
    workers.append(
        Worker(
            StockSimulation(test_data, pred, hps),
            num_workers-1,
            trainer,
            global_episodes,
            hps,
            use_as_validator=True
        )
    )
    saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(hps.model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(sess, coord, saver)
        t = threading.Thread(target=worker_work)
        t.start()
        sleep(0.3)
        worker_threads.append(t)
    coord.join(worker_threads)

