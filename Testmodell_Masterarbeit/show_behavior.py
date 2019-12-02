import os
from time import sleep

import numpy as np
import tensorflow as tf

from agents import DDDQAgent, Predictor
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from simulation import StockSimulation


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


warengruppe = 6

action_size = 6
state_size = np.array([18])
simulation_params = {
    'InputDirectory': os.path.join('files', 'raw'),
    'OutputDirectory': os.path.join('files', 'prepared'),
    'ZielWarengruppen': [71],
    'StatStateCategoricals': {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6},
}
predictor_dir = os.path.join('files', 'models', 'PredictorV2', '02RegWG' + str(warengruppe))
available_weights = os.listdir(predictor_dir)
available_weights.sort()
predictor_path = os.path.join(predictor_dir, available_weights[-1])
agent_path = os.path.join('files', 'models', 'DDDQN', 'Run44', 'model_9975.ckpt')
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

run_id = input('WelcheWG soll es sein?')
model_path = os.path.join('files', 'models', 'DDDQN', '01eval' + str(run_id))
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
    done = False
    while not done:
        step += 1
        print('Zustand:')
        print(np.array(state))
        value, advantage, qs_of_state = agent.sess.run(
            [agent.dq_network.value, agent.dq_network.advantage, agent.dq_network.output],
            feed_dict={
                agent.dq_network.inputs_: np.array(state).reshape((1, state_size[0]))
            })
        print('State-Value: {val} - Advantage: {adv}\nQ-Values: {qvals}'.format(
            val=value, adv=advantage, qvals=qs_of_state
        ))
        action = np.argmax(qs_of_state)
        reward, done, next_state = simulation.make_action(action)
        print('Wähle Aktion {act} und erhalte Belohnung {rew}'.format(act=action, rew=reward))
        state = next_state
        sleep(0.5)
        if step == 20:
            break

    # TODO: Ergebnis am Ende plotten
    weiterspielen_eing = input('Möchten sie eine weitere Ausführung starten? (j/n)')
    if weiterspielen_eing == 'j':
        weiterspielen = True
    else:
        weiterspielen = False

agent.sess.close()
