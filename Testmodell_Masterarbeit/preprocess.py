from data.preparation.clean import Datapipeline
import os
data_dir = os.path.join('files', 'raw')
output_dir = os.path.join('files', 'prepared')
warengruppen_maske = [1, 12, 55, 80, 17, 77, 71, 6, 28]
dyn_state_scalar_cols = ['Menge', 'MaxTemp_1D', 'MinTemp_1D', 'Wolken_1D',
                         'Regen_1D', 'MaxTemp_2D', 'MinTemp_2D', 'Wolken_2D', 'Regen_2D',
                         'Preis', 'relRabatt', 'absRabatt']
dyn_state_label_cols = ['in1', 'in2', 'in3', 'in4', 'in5']
dyn_state_category_cols = {'Wochentag': 7, 'Kalenderwoche': 54}
stat_state_scalar_cols = ['Eigenmarke', 'GuG', 'OSE', 'Saisonal', 'Kern', 'Bio', 'Glutenfrei',
                          'Laktosefrei']
stat_state_category_cols = {'MHDgroup': 7, 'Warengruppe': 9, 'Detailwarengruppe': None, 'Einheit': None}

# region Markt.Train
# markt = Datapipeline(
#     InputPath=data_dir,
#     OutputPath=output_dir,
#     ZielWarengruppen=warengruppen_maske,
#     Type='Markt',
#     DynStateScalarCols=dyn_state_scalar_cols,
#     DynStateLabelCols=dyn_state_label_cols,
#     DynStateCategoryCols=dyn_state_category_cols,
#     StatStateScalarCols=stat_state_scalar_cols,
#     StatStateCategoryCols=stat_state_category_cols
# )
# markt.read_files()
# markt.save_regression_numpy(
#     StartDatum='2017-01-01',
#     EndDatum='2017-12-31',
#     StepSize=6
# )
# # endregion
#
# # region Markt.Test
# markt = Datapipeline(
#     InputPath=data_dir,
#     OutputPath=output_dir,
#     ZielWarengruppen=warengruppen_maske,
#     Type='Markt',
#     DynStateScalarCols=dyn_state_scalar_cols,
#     DynStateLabelCols=dyn_state_label_cols,
#     DynStateCategoryCols=dyn_state_category_cols,
#     StatStateScalarCols=stat_state_scalar_cols,
#     StatStateCategoryCols=stat_state_category_cols
# )
# markt.read_files()
# markt.save_regression_numpy(
#     StartDatum='2018-01-01',
#     EndDatum='2018-12-31',
#     StepSize=6
# )
# # endregion
#
# # region Time.Train
# time = Datapipeline(
#     InputPath=data_dir,
#     OutputPath=output_dir,
#     ZielWarengruppen=warengruppen_maske,
#     Type='Time',
#     DynStateScalarCols=dyn_state_scalar_cols,
#     DynStateLabelCols=dyn_state_label_cols,
#     DynStateCategoryCols=dyn_state_category_cols,
#     StatStateScalarCols=stat_state_scalar_cols,
#     StatStateCategoryCols=stat_state_category_cols
# )
# time.read_files()
# time.save_regression_numpy(
#     StartDatum='2016-01-01',
#     EndDatum='2017-12-31',
#     StepSize=6
# )
# # endregion
#
# # region Time.Test
# time = Datapipeline(
#     InputPath=data_dir,
#     OutputPath=output_dir,
#     ZielWarengruppen=warengruppen_maske,
#     Type='Time',
#     DynStateScalarCols=dyn_state_scalar_cols,
#     DynStateLabelCols=dyn_state_label_cols,
#     DynStateCategoryCols=dyn_state_category_cols,
#     StatStateScalarCols=stat_state_scalar_cols,
#     StatStateCategoryCols=stat_state_category_cols
# )
# time.read_files()
# time.save_regression_numpy(
#     StartDatum='2018-01-01',
#     EndDatum='2018-12-31',
#     StepSize=6
# )
# # endregion
