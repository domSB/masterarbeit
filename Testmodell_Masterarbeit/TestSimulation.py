import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
warenausgang = pd.read_csv(
    os.path.join(simulation_params['InputDirectory'], '0 Warenausgang.Markt.csv'),
    header=1,
    names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
)
warenausgang['Datum'] = pd.to_datetime(warenausgang['Datum'], format='%d.%m.%y')
warenausgang = warenausgang[warenausgang.Belegtyp.isin(['UMSATZ_AKTION', 'UMSATZ_SCANNING'])]
warenausgang = warenausgang.groupby(['Markt', 'Artikel', 'Datum']).sum()
warenausgang.reset_index(inplace=True)
# endregion

# region Preise
preise = pd.read_csv(
    os.path.join(simulation_params['InputDirectory'], '0 Preise.Markt.csv'),
    header=1,
    names=['Preis', 'Artikel', 'Datum']
)
# endregion
# region Testschleife
absatz = np.array([0])
for i in range(200):
    states = []
    state, info = simulation.reset()
    # print('Artikel:', info['Artikel'], '\tMarkt:', info['Markt'])
    # artikel, markt = info['Artikel'], inv_markt_index[info['Markt']]
    # relevanter_absatz = warenausgang[(warenausgang.Artikel == artikel) & (warenausgang.Markt == markt)]
    # korrigierter_absatz = relevanter_absatz[relevanter_absatz.Datum.dt.weekday != 3]
    # states.append(state)
    done = False
    while not done:
        reward, done, new_state = simulation.make_action(2)
        states.append(new_state)
    absatz = np.concatenate((absatz, simulation.statistics.abschrift()), axis=0)
    # print(simulation.statistics.absaetze().sum(), 'Laut Simulation')
    # print(relevanter_absatz['Menge'].sum(), 'Laut relevantem Warenausgang')
    # print(korrigierter_absatz['Menge'].sum(), 'Laut korrigiertem Warenausgang')
    # arr = simulation.statistics.absaetze()
    # sim = arr[~np.equal(arr, 0)]
    # tru = korrigierter_absatz[korrigierter_absatz.Datum < '2019-06-01'].Menge.to_numpy()
    # print(sim.shape)
    # print(korrigierter_absatz.shape)
    # plt.bar(np.array(list(range(1, len(sim) + 1)))-0.25, sim, width=0.5, label='Simulation')
    # plt.bar(np.array(list(range(1, len(tru) + 1)))+0.25, tru, width=0.5, label='True')
    # plt.show()
    # print(arr[~np.equal(arr, 0)][-10:])
    # print(korrigierter_absatz.tail(10)['Menge'].to_numpy())
    # print(korrigierter_absatz.tail(10)['Datum'])
absatz = np.array(absatz)
zuwachs = absatz.cumsum()
plt.plot(list(range(0, len(zuwachs))), zuwachs)
plt.title(np.mean(absatz))
plt.show()
# endregion
