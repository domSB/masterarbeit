import os
from time import sleep
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from simulation import StockSimulation
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import DDDQAgent


def name_run(number):
    return 'Run' + str(number)


def get_artikel_wahl():
    artikel_wahl = input('Soll die Artikelwahl deterministisch oder zufällig erfolgen? (d/z)')
    if artikel_wahl == 'd':
        artikel = input('Bitte Artikelnummer eingeben:')
        markt = input('Bitte Marktnummer eingeben:')
        _wahl = (int(artikel), int(markt))
    elif artikel_wahl == 'z':
        _wahl = None
    else:
        _wahl = get_artikel_wahl()

    return _wahl


action_size = 6
time_steps = 6
state_size = np.array([9+6+3])
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
train_data, test_data = split_np_arrays(*simulation_data)
simulation = StockSimulation(train_data, predictor_path)

run_id = input('Welche Run id soll es sein?')
model_path = os.path.join('files', 'models', 'DDDQN', name_run(int(run_id)))
files = os.listdir(model_path)
files = [file for file in files if file[-5:] == 'index']
epsis = [file.split('.')[0][6:] for file in files]
epsi = input('Welche Epsiode soll benutzt werden? ' + str(epsis))
assert epsi in epsis, 'Unbekannte Episode'
checkpoint_path = os.path.join(model_path, 'model_' + epsi + '.ckpt')
session = tf.Session()
agent = DDDQAgent(
    0,
    0,
    1,
    0.001,
    32,
    action_size,
    state_size,
    time_steps,
    0.99,
    1000,
    session,
    '.'
)

saver = tf.train.Saver()
saver.save(agent.sess, checkpoint_path)

wahl = get_artikel_wahl()
weiterspielen = True
while weiterspielen:
    step = 0
    state, info = simulation.reset(wahl)
    recurrent_state = deque(maxlen=time_steps)
    for i in range(time_steps):
        recurrent_state.append(state)
    done = False
    while not done:
        step += 1
        print('Zustand:')
        print(np.array(recurrent_state))
        value, advantage, qs_of_state = agent.sess.run(
            [agent.dq_network.value, agent.dq_network.advantage, agent.dq_network.output],
            feed_dict={
                agent.dq_network.inputs_: np.array(recurrent_state).reshape((1, time_steps, state_size[0]))
            })
        print('State-Value: {val} - Advantage: {adv}\nQ-Values: {qvals}'.format(
            val=value, adv=advantage, qvals=qs_of_state
        ))
        action = np.argmax(qs_of_state)
        reward, done, next_state = simulation.make_action(action)
        print('Wähle Aktion {act} und erhalte Belohnung {rew}'.format(act=action, rew=reward))
        next_recurrent_state = recurrent_state
        next_recurrent_state.append(next_state)
        recurrent_state = next_recurrent_state
        sleep(2)

    # TODO: Ergebnis am Ende plotten
    weiterspielen_eing = input('Möchten sie eine weitere Ausführung starten? (j/n)')
    if weiterspielen_eing == 'j':
        weiterspielen = True
    else:
        weiterspielen = False

agent.sess.close()
