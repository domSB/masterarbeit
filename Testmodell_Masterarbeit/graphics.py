import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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
# simulation_params = {
#     'InputDirectory': os.path.join('files', 'raw'),
#     'OutputDirectory': os.path.join('files', 'prepared'),
#     'ZielWarengruppen': [17],
#     'StatStateCategoricals': {'MHDgroup': 7, 'Detailwarengruppe': None, 'Einheit': None, 'Markt': 6},
# }
# pipeline = DataPipeLine(**simulation_params)
# simulation_data = pipeline.get_regression_data()
# print([arr.shape for arr in simulation_data])
# dyn = simulation_data[1]
# absatz = dyn[:, 0, 0]
# uni, cnt = np.unique(absatz, return_counts=True)
# i = np.argwhere(uni == 0)[0][0]
# print(cnt[i]/cnt.sum())
# endregion

# region 03evalWG77/80
for warengruppe in[77, 80]:
    data_dir = os.path.join('files', 'prepared', 'Logging', 'A3C', '03evalWG' + str(warengruppe))

    action_entropy = pd.read_json(os.path.join(data_dir, 'run-train_2-tag-Losses_Entropy.json'))
    action_entropy.set_index(1, inplace=True)
    fehlmenge = pd.read_json(os.path.join(data_dir, 'run-train_2-tag-Model_FehlmengeQuote.json'))
    fehlmenge.set_index(1, inplace=True)

    metrics = pd.DataFrame(
        data={
            'Epoch': action_entropy.index.values,
            'ActionEntropy': action_entropy[2],
            'Fehlmenge': fehlmenge[2],
        },
        index=action_entropy.index.values
    )

    plt.plot(metrics.loc[:2000, 'ActionEntropy'], label='Aktions-Entropie')
    plt.plot(metrics.loc[:2000, 'Fehlmenge'], label='Fehlmenge')
    plt.legend()
    plt.xlabel('Episode')
    plt.title('Konvergenz A3C-Agent Warengruppe ' + str(warengruppe))
    plt.savefig(os.path.join('files', 'Konvergenz A3C-Agent Warengruppe {wg}.png'.format(wg=warengruppe)))
    plt.show()

# endregion


# region Belohnungsfunktion
def belohnung(ausfall, abschrift):
    """
    Monte Carlo Belohnung mit guten Gradienten und gleicher Bestrafung von Fehlmenge und Abschriften
    """
    z = np.log(3/(ausfall**2+abschrift**2+1))/4 + 3
    # z = 3 / (ausfall + abschrift + 1) ** 0.5 - 0.01 * abs(ausfall - abschrift) ** 1.1 - 0.01 * (ausfall + abschrift)
    return z


x = y = np.arange(0, 100)
X, Y = np.meshgrid(x, y)
zs = np.array([belohnung(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
Gx, Gy = np.gradient(Z)  # gradients with respect to x and y
G = (Gx**2+Gy**2)**.5  # gradient magnitude
N = np.clip(G, 0, 0.04)
N = N/N.max()

# plt.style.use('ggplot')
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.seismic(N), linewidth=0, antialiased=False, shade=False)
ax.set_xlabel('Fehlmenge')
ax.set_ylabel('Abschrift')
ax.set_zlabel('Belohnung')
ax.set_title('Monte Carlo Belohnungsfunktion')
plt.show()
# endregion
