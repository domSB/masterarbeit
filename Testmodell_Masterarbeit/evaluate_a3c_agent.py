import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from agents import Predictor, A3CNetwork
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from tqdm import tqdm


# region Hyperparameter

warengruppe = 12
state_size = np.array([18])
possible_actions = list(range(6))
tage = 388
gamma = .99
for warengruppen in [1, 6, 12, 17, 28, 55, 71, 77, 80]:
    tf.keras.backend.clear_session()
    model_path = os.path.join('files', 'models', 'A3C', '03eval' + str(warengruppe))
    logging_path = os.path.join('files', 'logging', 'A3C', '03eval' + str(warengruppe))

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
    train_data, test_data = split_np_arrays(*simulation_data, percentage=0)
    predictor = Predictor()
    predictor.build_model(
        dynamic_state_shape=simulation_data[1].shape[2],
        static_state_shape=simulation_data[2].shape[1]
    )
    predictor.load_from_weights(predictor_path)
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
        saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        simulation = StockSimulation(train_data, pred, 2, 'Bestandsreichweite')
        artikels = simulation.possibles

        for artikel in tqdm(artikels):
            state, info = simulation.reset(artikel)
            done = False
            while not done:
                strategy, value = sess.run(
                    [master_network.policy, master_network.value],
                    feed_dict={
                        master_network.inputs: [state]
                    }
                )
                action = np.random.choice(possible_actions, p=strategy[0])
                reward, done, next_state = simulation.make_action(action)
                state = next_state

        statistik = np.zeros((len(artikels), 3))
        for id_art, art_mkt in enumerate(artikels):
            aktueller_artikel = int(str(art_mkt)[-6:])
            abschriften = np.sum(simulation.statistics.abschrift(aktueller_artikel))
            fehlmenge = np.sum(simulation.statistics.fehlmenge(aktueller_artikel))
            reward = np.sum(simulation.statistics.rewards(aktueller_artikel))
            absatz = np.sum(simulation.statistics.absaetze(aktueller_artikel))
            actions = np.sum(simulation.statistics.actions(aktueller_artikel))
            if actions > 0:
                abschrift_quote = abschriften/actions
            else:
                abschrift_quote = 0
            statistik[id_art] = [reward/tage, abschrift_quote, fehlmenge/absatz]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.set_title('Leistungsbilanz A3C-Agent in Warengruppe {wg}'.format(wg=warengruppe))
        ax1.hist(statistik[:, 0], bins=100, range=(-3, 3), label=r'$\emptyset$-Belohnung', color='orangered')
        ax1.legend()

        ax2.hist(statistik[:, 1], bins=100, range=(0, 1), label=r'$\emptyset$-Abschriften Quote', color='limegreen')
        ax2.set_ylabel('Anzahl Artikel')
        ax2.legend()

        ax3.hist(statistik[:, 2], bins=100, range=(0, 1), label=r'$\emptyset$-Fehlmengen Quote', color='dodgerblue')
        ax3.legend()

        plt.savefig(os.path.join('files', 'graphics', 'A3C-Agent Eval 01 Warengruppe {wg}').format(wg=warengruppe))


