import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import DDDQAgent, Predictor

plt.style.use('ggplot')

tf.get_logger().setLevel('ERROR')


# region Hyperparams
action_size = 6
learning_rate = 0

memory_size = 10
tage = 322

episodes = 10000
batch_size = 32

learn_step = 4
max_tau = learn_step * 10000

epsilon_start = 0
epsilon_stop = 0
epsilon_decay = 1

gamma = 0.99

for warengruppe in [1, 6, 12, 17, 28, 55, 71, 77, 80]:
    tf.keras.backend.clear_session()
    state_size = np.array([18])   # Zeitdimension, 6 Vorhersagen, Bestand, Abschriften, Fehlbestand

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
    agent_path = os.path.join('files', 'models', 'DDDQN', '01eval' + str(warengruppe))
    # endregion

    pipeline = DataPipeLine(**simulation_params)
    simulation_data = pipeline.get_regression_data()
    train_data, test_data = split_np_arrays(*simulation_data, percentage=0)

    state_size[0] += simulation_data[2].shape[1]

    predictor = Predictor()
    predictor.build_model(
        dynamic_state_shape=simulation_data[1].shape[2],
        static_state_shape=simulation_data[2].shape[1]
    )
    predictor.load_from_weights(predictor_path)
    print('Predicting', end='')
    pred = predictor.predict(
        {
            'dynamic_input': train_data[1],
            'static_input': train_data[2]
        }
    )
    print('and done ;)')

    simulation = StockSimulation(train_data, pred, 2, 'Bestandsreichweite')

    # endregion

    # region Initialisieren
    session = tf.Session()
    agent = DDDQAgent(
        epsilon_start,
        epsilon_stop,
        epsilon_decay,
        learning_rate,
        batch_size,
        action_size,
        state_size,
        gamma,
        memory_size,
        session,
        '.'
    )
    # endregion
    saver = tf.train.Saver()
    saver.restore(agent.sess, tf.train.latest_checkpoint(agent_path))
    artikels = simulation.possibles

    for artikel in tqdm(artikels):
        state, info = simulation.reset(artikel)
        done = False
        while not done:
            action = agent.act(state)
            reward, done, next_state = simulation.make_action(np.argmax(action))
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
    ax1.set_title('Leistungsbilanz DDDQN-Agent in Warengruppe {wg}'.format(wg=warengruppe))
    ax1.hist(statistik[:, 0], bins=100, range=(-3, 3), label=r'$\emptyset$-Belohnung', color='orangered')
    ax1.legend()

    ax2.hist(statistik[:, 1], bins=100, range=(0, 1), label=r'$\emptyset$-Abschriften Quote', color='limegreen')
    ax2.set_ylabel('Anzahl Artikel')
    ax2.legend()

    ax3.hist(statistik[:, 2], bins=100, range=(0, 1), label=r'$\emptyset$-Fehlmengen Quote', color='dodgerblue')
    ax3.legend()

    plt.savefig(os.path.join('files', 'graphics', 'DDDQN-Agent Eval 01 Warengruppe {wg}').format(wg=warengruppe))
