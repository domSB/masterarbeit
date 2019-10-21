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
# endregion

# region Testschleife
for i in range(5):
    states = []
    state, info = simulation.reset()
    print('Artikel:', info['Artikel'], '\nMarkt:', info['Markt'])
    artikel, markt = info['Artikel'], inv_markt_index[info['Markt']]
    relevanter_absatz = warenausgang[(warenausgang.Artikel == artikel) & (warenausgang.Markt == markt)]
    states.append(state)
    done = False
    while not done:
        reward, done, new_state = simulation.make_action(0)
        states.append(new_state)
    # endregion
    print('Laut Simulation:', simulation.statistics.absaetze().sum())
    print('Laut Warenausgang:', relevanter_absatz['Menge'].sum())
    print('Datum', relevanter_absatz['Datum'].min(), relevanter_absatz['Datum'].max())
