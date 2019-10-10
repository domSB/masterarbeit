from Testmodell_Masterarbeit.data.access import DataPipeLine
import os

print('Ändere Skript, um Daten mit dieser ausführbaren Datei vorzubereiten')
data_dir = os.path.join('files', 'raw')
output_dir = os.path.join('files', 'prepared')
# [1, 12, 55, 80, 17, 77, 71, 6, 28]
warengruppen_maske = [55]
stat_state_category_cols = {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None}
simulation_params = {
    'InputPath': data_dir,
    'OutputPath': output_dir,
    'ZielWarengruppen': warengruppen_maske,
    'StatStateCategoryCols': stat_state_category_cols,
}
pipeline = DataPipeLine(**simulation_params)
a, b, c = pipeline.get_regression_data()
