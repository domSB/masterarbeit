import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Testmodell_Masterarbeit.data.access import DataPipeLine

pipeline = DataPipeLine()
# TODO: Baseline in Klasse einbauen
# bewegung = self.dynamic_state.loc[:, ['Markt', 'Artikel', 'Datum', 'Menge', 'UNIXDatum']].copy()
# bewegung.reset_index(inplace=True, drop=True)
# bewegung['Prediction'] = bewegung.groupby(['Markt', 'Artikel'])['Menge'].shift(1)
# bewegung['AError'] = np.abs(bewegung['Menge'] - bewegung['Prediction'])
# bewegung['SError'] = np.square(bewegung['AError'])
# bewegung.dropna(inplace=True)
# bewegung['MAE'] = bewegung['AError'].rolling(prediction_days).mean()
# bewegung['MSE'] = bewegung['SError'].rolling(prediction_days).mean()
# self.mae = np.mean(bewegung['MAE'])
# self.mse = np.mean(bewegung['MSE'])
# print('BASELINE\n---\nMean Average Error: {mae} \nMean Squared Error: {mse}'.format(
#     mae=self.mae,
#     mse=self.mse
# ))