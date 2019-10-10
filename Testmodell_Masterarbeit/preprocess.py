from Testmodell_Masterarbeit.data.access import DataPipeLine
import os

print('Ändere Skript, um Daten mit dieser ausführbaren Datei vorzubereiten')
data_dir = os.path.join('files', 'raw')
output_dir = os.path.join('files', 'prepared')
# [1, 12, 55, 80, 17, 77, 71, 6, 28]
warengruppen_maske = [17]
stat_state_category_cols = {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6}
simulation_params = {
    'InputDirectory': data_dir,
    'OutputDirectory': output_dir,
    'ZielWarengruppen': warengruppen_maske,
    'StatStateCategoricals': stat_state_category_cols,
}
pipeline = DataPipeLine(**simulation_params)
a, b, c = pipeline.get_regression_data()
