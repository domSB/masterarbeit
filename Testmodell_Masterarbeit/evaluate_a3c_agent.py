import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from agents import Predictor, A3CNetwork
from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from utils import Hyperparameter
from tqdm import tqdm

plt.style.use('ggplot')
# region Hyperparameter

hps = Hyperparameter()
hps.load(os.path.join('files', 'logging', 'A3C', '2eval55-2363', 'Hyperparameter.yaml'))

predictor_dir = os.path.join('files',  'models', 'PredictorV2', '02RegWG' + str(hps.warengruppe[0]))
available_weights = os.listdir(predictor_dir)
available_weights.sort()
predictor_path = os.path.join(predictor_dir, available_weights[-1])
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
print('Predicting',  end='')
pred = predictor.predict(
    {
        'dynamic_input': train_data[1],
        'static_input': train_data[2]
    }
)
print(' and done ;)')
possible_actions = np.identity(hps.action_size, dtype=int).tolist()
# endregion

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    master_network = A3CNetwork('global', None, hps)
    saver = tf.train.Saver(max_to_keep=1, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global'))

with tf.Session() as sess:
    print('Loading Model...')

    ckpt = tf.train.get_checkpoint_state(hps.model_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    simulation = StockSimulation(train_data, pred, hps)
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
            action = np.random.choice(strategy[0], p=strategy[0])
            action = np.argmax(strategy == action)
            reward, done, next_state = simulation.make_action(action)
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
ax1.set_title('Leistungsbilanz A3C-Agent in Warengruppe {wg}-{dt_wg}'.format(
    wg=hps.warengruppe[0],
    dt_wg=hps.detail_warengruppe[0])
)
ax1.hist(statistik[:, 0], bins=100, range=(0, 3.5), label=r'$\emptyset$-Belohnung', color='orangered')
ax1.legend()

ax2.hist(statistik[:, 1], bins=100, range=(0, 1), label=r'$\emptyset$-Abschriften Quote', color='limegreen')
ax2.set_ylabel('Anzahl Artikel')
ax2.legend()

ax3.hist(statistik[:, 2], bins=100, range=(0, 1), label=r'$\emptyset$-Fehlmengen Quote', color='dodgerblue')
ax3.legend()

plt.savefig(os.path.join('files', 'graphics', 'A3C-Agent Eval 02 Warengruppe {wg}-{dt_wg}'.format(
    wg=hps.warengruppe[0],
    dt_wg=hps.detail_warengruppe[0]))
)
