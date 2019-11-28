import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import random
from data.access import DataPipeLine
from data.preparation import split_np_arrays
from agents import Predictor

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
# for warengruppe in[77, 80]:
#     data_dir = os.path.join('files', 'prepared', 'Logging', 'A3C', '03evalWG' + str(warengruppe))
#
#     action_entropy = pd.read_json(os.path.join(data_dir, 'run-train_2-tag-Losses_Entropy.json'))
#     action_entropy.set_index(1, inplace=True)
#     fehlmenge = pd.read_json(os.path.join(data_dir, 'run-train_2-tag-Model_FehlmengeQuote.json'))
#     fehlmenge.set_index(1, inplace=True)
#
#     metrics = pd.DataFrame(
#         data={
#             'Epoch': action_entropy.index.values,
#             'ActionEntropy': action_entropy[2],
#             'Fehlmenge': fehlmenge[2],
#         },
#         index=action_entropy.index.values
#     )
#
#     plt.plot(metrics.loc[:2000, 'ActionEntropy'], label='Aktions-Entropie')
#     plt.plot(metrics.loc[:2000, 'Fehlmenge'], label='Fehlmenge')
#     plt.legend()
#     plt.xlabel('Episode')
#     plt.title('Konvergenz A3C-Agent Warengruppe ' + str(warengruppe))
#     plt.savefig(os.path.join('files', 'Konvergenz A3C-Agent Warengruppe {wg}.png'.format(wg=warengruppe)))
#     plt.show()

# endregion


# region Belohnungsfunktion
# def belohnung(ausfall, abschrift, bestand):
#     """
#     Monte Carlo Belohnung mit guten Gradienten und gleicher Bestrafung von Fehlmenge und Abschriften
#     """
#     z = np.log(3/(bestand**2+ausfall**2+abschrift**2+1))/4 + 3
#     # z = 3 / (ausfall + abschrift + 1) ** 0.5 - 0.01 * abs(ausfall - abschrift) ** 1.1 - 0.01 * (ausfall + abschrift)
#     return z
#
#
# for cmap, bestand in zip([cm.seismic, cm.jet, cm.hot],[0, 5, 100]):
#     x = y = np.arange(0, 100)
#     X, Y = np.meshgrid(x, y)
#     zs = np.array([belohnung(x, y, bestand) for x, y in zip(np.ravel(X), np.ravel(Y))])
#     Z = zs.reshape(X.shape)
#     Gx, Gy = np.gradient(Z)  # gradients with respect to x and y
#     G = (Gx**2+Gy**2)**.5  # gradient magnitude
#     N = np.clip(G, 0, 0.04)
#     N = N/N.max()
#
#     # plt.style.use('ggplot')
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cmap(N), linewidth=0, antialiased=False, shade=False)
#     ax.set_xlabel('Fehlmenge')
#     ax.set_ylabel('Abschrift')
#     ax.set_zlabel('Belohnung')
#     ax.set_title('Monte Carlo Belohnungsfunktion')
#     plt.show()
# endregion

# region Predictor Accuracy
# warengruppe = 55
# state_size = np.array([18])
#
# predictor_dir = os.path.join('files',  'models', 'PredictorV2', '02RegWG' + str(warengruppe))
# available_weights = os.listdir(predictor_dir)
# available_weights.sort()
# predictor_path = os.path.join(predictor_dir, available_weights[-1])
#
# pipeline = DataPipeLine(ZielWarengrupppe=[warengruppe])
# simulation_data = pipeline.get_regression_data()
# train_data, test_data = split_np_arrays(*simulation_data)
#
# print([tr.shape for tr in train_data])
# state_size[0] += simulation_data[2].shape[1]
#
# predictor = Predictor()
# predictor.build_model(
#     dynamic_state_shape=simulation_data[1].shape[2],
#     static_state_shape=simulation_data[2].shape[1]
# )
# predictor.load_from_weights(predictor_path)
# print('Predicting', end='')
# pred = predictor.predict(
#     {
#         'dynamic_input': train_data[1],
#         'static_input': train_data[2]
#     }
# )
# lab, dyn, stat, ids = train_data
# possibles = np.unique(ids)
#
# for i in range(3):
#     position_wahl = np.random.choice(len(possibles))
#     ids_wahl = np.argwhere(np.isin(ids, possibles[position_wahl])).reshape(-1)
#     static_state = stat[ids_wahl]
#     dynamic_state = dyn[ids_wahl]
#     predicted_state = pred[ids_wahl]
#     true_state = lab[ids_wahl]
#     artikel_absatz = (dyn[ids_wahl, 0, 0] * 8).astype(np.int64)
#     vorhersagen = np.argmax(predicted_state, axis=2)
#     df = pd.DataFrame(
#         {
#             '1-Tages Prognose': vorhersagen[:-1, 0],
#             'Absatz': artikel_absatz[1:]
#         }
#     )
#     df.plot.bar(rot=90)
#     fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
#     ax1.plot(artikel_absatz[2:], '.', label='Absatz')
#     ax1.plot(vorhersagen[1:-1, 0], '2', label='1-Tages-Prognose')
#     ax1.plot(vorhersagen[0:-2, 1], '*', label='2-Tages-Prognose')
#     ax1.set_ylabel('Menge')
#     ax1.legend()
#     ax2.plot(vorhersagen[1:-1, 0] - artikel_absatz[2:], '.', label='Prognose Abweichung')
#     ax2.legend()
#     ax2.set_xlabel('Verkaufstage')
#     ax2.set_ylabel('Menge')
#     plt.show()

# endregion

# region Q Konvergenz
# data_dir = os.path.join('files', 'prepared', 'Logging', 'DDDQN')
#
# big_q = pd.read_json(os.path.join(data_dir, 'BigQ.json'))
# small_q = pd.read_json(os.path.join(data_dir, 'ClippedQ.json'))
# big_q.set_index(1, inplace=True)
# small_q.set_index(1, inplace=True)
#
# metrics = pd.DataFrame(
#     data={
#         'Epoch': big_q.index.values,
#         'RegularQ': big_q[2],
#         'ScaledQ': small_q[2],
#     },
#     index=big_q.index.values
# )
#
# plt.plot(metrics.loc[:, 'RegularQ'], label='Q bei Standard-Belohnung ')
# plt.plot(metrics.loc[:, 'ScaledQ'], label='Q bei skalierter Belohnung')
# plt.legend()
# plt.xlabel('Episode')
# plt.title('Konvergenz der Q-Werte')
# plt.savefig(os.path.join('files', 'graphics', 'Konvergenz der Q-Werte.png'))
# plt.show()
# endregion

# region Werbehäufigkeit

artikelstamm = pd.read_csv(
    'files/raw/0 ArtikelstammV4.csv',
    header=0,
    names=['Artikel', 'Warengruppe', 'Detailwarengruppe', 'Bezeichnung',
           'Eigenmarke', 'Einheit', 'Verkaufseinheit', 'MHD',
           'GuG', 'OSE', 'OSEText', 'Saisonal',
           'Kern', 'Bio', 'Glutenfrei',
           'Laktosefrei', 'MarkeFK', 'Region']
)
warenausgang = pd.read_csv(
    'files/raw/0 Warenausgang.Time.csv',
    header=1,
    names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
)
warenausgang['Datum'] = pd.to_datetime(warenausgang['Datum'], format='%d.%m.%y')
wareneingang = pd.read_csv(
    'files/raw/0 Wareneingang.Time.csv',
    header=1,
    names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
)

wareneingang['Datum'] = pd.to_datetime(wareneingang['Datum'], format='%d.%m.%y')

bestand = pd.read_csv(
    'files/raw/0 Warenbestand.Time.csv',
    header=1,
    names=['Markt', 'Artikel', 'Bestand', 'EK', 'VK', 'Anfangsbestand', 'Datum']
)
bestand['Datum'] = pd.to_datetime(bestand['Datum'], format='%d.%m.%y')
aktionspreise = pd.read_csv(
    'files/raw/0 Aktionspreise.Time.csv',
    header=1,
    names=['Artikel', 'DatumAb', 'DatumBis', 'Aktionspreis']
)
artikel = warenausgang.Artikel.unique()
aktionspreise = aktionspreise[aktionspreise.Artikel.isin(artikel)]
aktionspreise['DatumAb'] = pd.to_datetime(aktionspreise['DatumAb'], format='%d.%m.%Y')
aktionspreise['DatumBis'] = pd.to_datetime(aktionspreise['DatumBis'], format='%d.%m.%Y')
aktions_menge = aktionspreise[['Artikel', 'Aktionspreis']].groupby('Artikel').count()
plt.style.use('ggplot')
plt.hist(aktions_menge.Aktionspreis, bins=100)
plt.xlabel('Anzahl Werbeaktionen')
plt.ylabel('Anzahl Artikel')
plt.title('Werbehäufigkeit der Artikel in 2 Jahren')

absatz = warenausgang[warenausgang.Belegtyp.isin(['UMSATZ_SCANNING', 'UMSATZ_AKTION'])]
absatz.drop(columns=['Markt'], inplace=True)
absatz = absatz.groupby(['Artikel', 'Datum']).sum()
absatz.reset_index(inplace=True)
artikelstamm.set_index('Artikel', inplace=True)
absatz = pd.merge(absatz, artikelstamm.Warengruppe, how='left', left_on='Artikel', right_index=True)
absatz = absatz.groupby(['Warengruppe', 'Datum']).sum()
absatz.reset_index(inplace=True)
absatz.drop(columns=['Artikel'], inplace=True)
test = pd.pivot_table(absatz, values='Menge', index=['Datum'], columns=['Warengruppe'], aggfunc=np.sum)
test.fillna(0, inplace=True)
arr = test.drop(columns=[82, 22]).T.to_numpy()
y_tick_labels = test.drop(columns=[82, 22]).T.index.values
x_tick_labels = test.drop(columns=[82, 22]).T.columns.strftime('%b-%Y')
y_ticks = np.arange(0, len(y_tick_labels), 3)
x_ticks = np.arange(0, len(x_tick_labels), 101)
y_ticks_labels = y_tick_labels[y_ticks]
x_ticks_labels = x_tick_labels[x_ticks]

fig, ax = plt.subplots()
im = ax.imshow(arr, cmap=cm.gist_rainbow, aspect='auto')
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks_labels)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks_labels)
plt.setp(ax.get_xticklabels(), rotation=-30, ha='left', va='top',
         rotation_mode="anchor")
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Tagesabsatz', rotation=-90, va="bottom")
ax.set_xlabel('Datum')
ax.set_ylabel('Warengruppe')
ax.set_title('Verteilung der Warengruppenabsätze')
plt.savefig('files/graphics/Verteilung der Warengruppenabsätze.png')
plt.show()

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
flierprops = dict(marker='.', markerfacecolor='blue', markersize=2)
bp = ax.boxplot(test.to_numpy(), flierprops=flierprops)
x_tick_labels = test.columns.values
x_ticks = np.arange(0, len(x_tick_labels), 3)
x_ticks_labels = x_tick_labels[x_ticks]
ax.set_xticks(x_ticks)  # Diagramm umgedreht
ax.set_xticklabels(x_ticks_labels)
ax.set_xlabel('Warengruppe')
ax.set_title('Streubreite der Tagesabsätze je Warengruppe')
plt.savefig('files/graphics/Warengruppen Boxplot.png')
plt.show()

# endregion

# region Skalierung Abschriften
# indir = os.path.join('Testmodell_Masterarbeit', 'files', 'prepared', 'Logging', 'DDDQN')
# df = pd.read_csv(os.path.join(indir, 'Skalierung Abschriften.csv'))
# dd = df.ewm(com=0.6).mean()
# plt.style.use('ggplot')
# plt.plot(df.Absatz, label='Absatz')
# plt.plot(df.Fehlmenge, label='Fehlmenge')
# plt.plot(df.Abschrift, label='Abschrift')
# plt.legend()
# plt.title('Skalierung der Messgrößen Abschrift & Fehlmenge')
# plt.savefig(
#     os.path.join('Testmodell_Masterarbeit', 'files', 'graphics', 'Skalierung der Messwerte Abschrift & Fehlmenge.png'))
# plt.show()
# endregion
