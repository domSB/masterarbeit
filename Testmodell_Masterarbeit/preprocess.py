from Testmodell_Masterarbeit.data.access import DataPipeLine
import os

print('Ändere Skript, um Daten mit dieser ausführbaren Datei vorzubereiten')
data_dir = os.path.join('files', 'raw')
output_dir = os.path.join('files', 'prepared')
warengruppen_maske = [17]
dyn_state_scalar_cols = ['Menge', 'MaxTemp_1D', 'MinTemp_1D', 'Wolken_1D',
                         'Regen_1D', 'MaxTemp_2D', 'MinTemp_2D', 'Wolken_2D', 'Regen_2D',
                         'Preis', 'relRabatt', 'absRabatt', 'vDauer']
dyn_state_label_cols = ['in1', 'in2', 'in3', 'in4', 'in5', 'in6']
dyn_state_category_cols = {'Wochentag': 7, 'Kalenderwoche': 54}
stat_state_scalar_cols = ['Eigenmarke', 'GuG', 'OSE', 'Saisonal', 'Kern', 'Bio', 'Glutenfrei',
                          'Laktosefrei']
stat_state_category_cols = {'MHDgroup': 7, 'Warengruppe': 9, 'Detailwarengruppe': None, 'Einheit': None}
simulation_params = {
    'InputPath': data_dir,
    'OutputPath': output_dir,
    'ZielWarengruppen': warengruppen_maske,
    'DynStateScalarCols': dyn_state_scalar_cols,
    'DynStateLabelCols': dyn_state_label_cols,
    'DynStateCategoryCols': dyn_state_category_cols,
    'StatStateScalarCols': stat_state_scalar_cols,
    'StatStateCategoryCols': stat_state_category_cols,
}
pipeline = DataPipeLine(**simulation_params)
a, c = pipeline.get_simulation_data()
