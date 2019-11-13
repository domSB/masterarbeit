import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import DDDQAgent, Predictor
from utils import Hyperparameter

plt.style.use('ggplot')

tf.get_logger().setLevel('ERROR')


# region Hyperparams
hps = Hyperparameter()
hps.load(os.path.join('files', 'logging', 'DDDQN', '38eval55-2363', 'Hyperparameter.yaml'))


for _ in [55]:
    tf.keras.backend.clear_session()

    predictor_dir = os.path.join('files',  'models', 'PredictorV2', '02RegWG' + str(hps.warengruppe[0]))
    available_weights = os.listdir(predictor_dir)
    available_weights.sort()
    predictor_path = os.path.join(predictor_dir, available_weights[-1])
    agent_path = os.path.join('files', 'models', 'DDDQN', '38eval55-2363')
    # endregion

    pipeline = DataPipeLine(ZielWarengruppen=hps.warengruppe, DetailWarengruppe=hps.detail_warengruppe)
    simulation_data = pipeline.get_regression_data()
    train_data, test_data = split_np_arrays(*simulation_data, percentage=0)

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

    simulation = StockSimulation(train_data, pred, hps)
    # endregion

    # region Initialisieren
    session = tf.Session()
    agent = DDDQAgent(session, hps)
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
        reward = np.mean(simulation.statistics.rewards(aktueller_artikel))
        absatz = np.sum(simulation.statistics.absaetze(aktueller_artikel))
        actions = np.sum(simulation.statistics.actions(aktueller_artikel))
        if actions > 0:
            abschrift_quote = abschriften/actions
        else:
            abschrift_quote = 0
        statistik[id_art] = [reward, abschrift_quote, fehlmenge/absatz]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_title('Leistungsbilanz DDDQN-Agent in Warengruppe {wg}-{dt_wg}'.format(
        wg=hps.warengruppe[0],
        dt_wg=hps.detail_warengruppe[0])
    )
    ax1.hist(statistik[:, 0], bins=100, range=(0, 3), label=r'$\emptyset$-Belohnung', color='orangered')
    ax1.legend()

    ax2.hist(statistik[:, 1], bins=100, range=(0, 1), label=r'$\emptyset$-Abschriften Quote', color='limegreen')
    ax2.set_ylabel('Anzahl Artikel')
    ax2.legend()

    ax3.hist(statistik[:, 2], bins=100, range=(0, 1), label=r'$\emptyset$-Fehlmengen Quote', color='dodgerblue')
    ax3.legend()

    plt.savefig(os.path.join('files', 'graphics', 'DDDQN-Agent Eval 38 Warengruppe {wg}-{dt_wg}'.format(
        wg=hps.warengruppe[0],
        dt_wg=hps.detail_warengruppe[0]))
    )
