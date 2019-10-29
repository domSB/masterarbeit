import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from data.access import DataPipeLine

plt.style.use('ggplot')
# region Tensorboard Predictor Training
#
# for warengruppe in [1, 6, 12, 17, 28, 55, 71, 77, 80]:
#
#     data_dir = os.path.join('files', 'prepared', 'Logging', 'Predictor')
#     wg_dir = os.path.join(data_dir, 'WG' + str(warengruppe))
#
#     epoch_loss = pd.read_json(os.path.join(wg_dir, 'run-.-tag-epoch_loss.json'))
#     epoch_val_loss = pd.read_json(os.path.join(wg_dir, 'run-.-tag-epoch_val_loss.json'))
#
#     acc_1d = pd.read_json(os.path.join(wg_dir, 'run-.-tag-epoch_val_1day_categorical_accuracy.json'))
#     acc_2d = pd.read_json(os.path.join(wg_dir, 'run-.-tag-epoch_val_2day_categorical_accuracy.json'))
#     acc_3d = pd.read_json(os.path.join(wg_dir, 'run-.-tag-epoch_val_3day_categorical_accuracy.json'))
#     acc_4d = pd.read_json(os.path.join(wg_dir, 'run-.-tag-epoch_val_4day_categorical_accuracy.json'))
#     acc_5d = pd.read_json(os.path.join(wg_dir, 'run-.-tag-epoch_val_5day_categorical_accuracy.json'))
#     acc_6d = pd.read_json(os.path.join(wg_dir, 'run-.-tag-epoch_val_6day_categorical_accuracy.json'))
#
#     metrics = pd.DataFrame(
#         data={
#             'Epoch': epoch_loss.index.values + 1,
#             'Loss': epoch_loss[2],
#             'Val_Loss': epoch_val_loss[2],
#             'Acc_1d': acc_1d[2],
#             'Acc_2d': acc_2d[2],
#             'Acc_3d': acc_3d[2],
#             'Acc_4d': acc_4d[2],
#             'Acc_5d': acc_5d[2],
#             'Acc_6d': acc_6d[2],
#         },
#         index=[i for i in range(epoch_loss.shape[0])]
#     )
#     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#     ax1.plot(metrics.Epoch, metrics.Loss, label='Fehler')
#     ax1.plot(metrics.Epoch, metrics.Val_Loss, label='Validierungs Fehler')
#     ax1.set_ylabel('Fehlerterm')
#     ax1.legend()
#
#     ax2.plot(metrics.Epoch, metrics.Acc_1d, label='1-Tages Vorhersage')
#     ax2.plot(metrics.Epoch, metrics.Acc_2d, label='2-Tages Vorhersage')
#     ax2.plot(metrics.Epoch, metrics.Acc_3d, label='3-Tages Vorhersage')
#     ax2.plot(metrics.Epoch, metrics.Acc_4d, label='4-Tages Vorhersage')
#     ax2.plot(metrics.Epoch, metrics.Acc_5d, label='5-Tages Vorhersage')
#     ax2.plot(metrics.Epoch, metrics.Acc_6d, label='6-Tages Vorhersage')
#     ax2.set_xlabel('Epoche')
#     ax2.set_ylabel('Trefferquote')
#     ax2.legend()
#     fig.suptitle('Vorhersage-Ergebnisse Warengruppe {wg}'.format(wg=warengruppe))
#     plt.savefig(os.path.join(wg_dir, 'Prädiktor Ergebnis Warengruppe {wg}.png'.format(wg=warengruppe)))
# endregion

# region Analyse Absatz
simulation_params = {
    'InputDirectory': os.path.join('files', 'raw'),
    'OutputDirectory': os.path.join('files', 'prepared'),
    'ZielWarengruppen': [71],
    'StatStateCategoricals': {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6},
}
pipeline = DataPipeLine(**simulation_params)
simulation_data = pipeline.get_regression_data()
print([arr.shape for arr in simulation_data])
dyn = simulation_data[1]
absatz = dyn[:, 0, 0]
uni, cnt = np.unique(absatz, return_counts=True)
i = np.argwhere(uni == 0)[0][0]
print(cnt[i]/cnt.sum())
# endregion
