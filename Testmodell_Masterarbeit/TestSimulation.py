import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import entropy
from collections import defaultdict, Counter

from data.access import DataPipeLine
from data.preparation import split_np_arrays
from simulation import StockSimulation

# region Simulation Laden
simulation_params = {
    'InputDirectory': os.path.join('files', 'raw'),
    'OutputDirectory': os.path.join('files', 'prepared'),
    'ZielWarengruppen': [71],
    'StatStateCategoricals': {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6},
}
inv_markt_index = {
    0: 27,
    1: 67,
    2: 87,
    3: 128,
    4: 129,
    5: 147
}
predictor_path = os.path.join('files', 'models', 'PredictorV2', '01RegWG71', 'weights.30-0.21.hdf5')

pipeline = DataPipeLine(**simulation_params)
simulation_data = pipeline.get_regression_data()
train_data, test_data = split_np_arrays(*simulation_data)
simulation = StockSimulation(train_data, predictor_path)
# endregion

# endregion

# region ursprüngliche Absätze laden
# warenausgang = pd.read_csv(
#     os.path.join(simulation_params['InputDirectory'], '0 Warenausgang.Markt.csv'),
#     header=1,
#     names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
# )
# warenausgang['Datum'] = pd.to_datetime(warenausgang['Datum'], format='%d.%m.%y')
# warenausgang = warenausgang[warenausgang.Belegtyp.isin(['UMSATZ_AKTION', 'UMSATZ_SCANNING'])]
# warenausgang = warenausgang.groupby(['Markt', 'Artikel', 'Datum']).sum()
# warenausgang.reset_index(inplace=True)
# endregion

# region Preise
# preise = pd.read_csv(
#     os.path.join(simulation_params['InputDirectory'], '0 Preise.Markt.csv'),
#     header=1,
#     names=['Preis', 'Artikel', 'Datum']
# )
# endregion
# region Testschleife
x = []
y = []
for i in range(3000):
    states = []
    new_state, info = simulation.reset()
    artikel, markt = info['Artikel'], inv_markt_index[info['Markt']]
    # relevanter_absatz = warenausgang[(warenausgang.Artikel == artikel) & (warenausgang.Markt == markt)]
    # korrigierter_absatz = relevanter_absatz[relevanter_absatz.Datum.dt.weekday != 3]
    epsilon = i/3000
    rewards = []
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.random.choice(6)
        else:
            action = int(new_state[1])
        reward, done, new_state = simulation.make_action(action)
        states.append(new_state)
        rewards.append(reward)
    x.append(epsilon)
    y.append(np.sum(rewards))
    # a_values, a_count = np.unique(simulation.statistics.actions(), return_counts=True)
    # abs_values, abs_count = np.unique(simulation.statistics.absaetze(), return_counts=True)
    # actions_df = pd.DataFrame(data={'Actions': a_count}, index=a_values)
    # absatz_df = pd.DataFrame(data={'Absatz': abs_count}, index=abs_values)
    # probas = pd.merge(actions_df, absatz_df, how='outer', left_index=True, right_index=True)
    # probas.fillna(0, inplace=True)
    # print('Action Entropy:', entropy(probas.Actions, qk=probas.Absatz))
    # print(np.sum(rewards))

plt.style.use('ggplot')
plt.plot(x, y, '.')
plt.ylabel('Gesamtbelohnung')
plt.xlabel('1-Epsilon')
plt.title('Belohnungsentwicklung')
plt.show()
# endregion

# absatz_sammlung = defaultdict(list)
# # region Absatzproblem erörtern
# for i in range(10):
#     state, info = simulation.reset()
#     artikel, markt = info['Artikel'], inv_markt_index[info['Markt']]
#     key = str(artikel) + '-' + str(markt)
#     absatz_sammlung[key].append(simulation.artikel_absatz.sum())



# endregion