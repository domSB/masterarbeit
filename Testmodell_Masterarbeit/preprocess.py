from Testmodell_Masterarbeit.data.access import DataPipeLine
import os

data_dir = os.path.join('files', 'raw')
output_dir = os.path.join('files', 'prepared')
for warengruppen_maske in [1, 12, 55, 80, 17, 77, 71, 6, 28]:
    print('Starte mit Warengruppe', warengruppen_maske)
    stat_state_category_cols = {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6}
    simulation_params = {
        'InputDirectory': data_dir,
        'OutputDirectory': output_dir,
        'ZielWarengruppen': [warengruppen_maske],
        'StatStateCategoricals': stat_state_category_cols,
    }
    pipeline = DataPipeLine(**simulation_params)
    _, _, _, _ = pipeline.get_regression_data()
