import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

from Testmodell_Masterarbeit.data.preparation.clean import Datapipeline


def show_timelines(start, end):
    idx = np.random.randint(0, len(artikel_mit_ose), 1)
    mask = artikel_mit_ose[idx]
    testabsatz = warenausgang[warenausgang.Artikel.isin(mask)]
    strahl = pd.date_range(start, end, freq='d')
    testabsatz.set_index('Datum', inplace=True)
    testabsatz = testabsatz.groupby(['Markt', 'Belegtyp']).apply(lambda x: x.reindex(strahl))
    maerkte_mit_absatz = pd.unique(testabsatz.Markt)[~np.isnan(pd.unique(testabsatz.Markt))]
    max_absatz = testabsatz.loc[
                 (slice(None), slice('UMSATZ_AKTION', 'UMSATZ_SCANNING'), slice(None)),
                 :
                 ].Menge.max()
    fig, axes = plt.subplots(nrows=len(maerkte_mit_absatz), ncols=1, sharex=True)
    fig.suptitle(
        'Absätze je Markt für Artikel {} OSE {}'.format(
            artikelstamm.loc[mask[0]].Bezeichnung,
            artikelstamm.loc[mask[0]].Inhalt),
        y=0.99)
    for i, (axis, markt) in enumerate(zip(axes, maerkte_mit_absatz)):
        try:
            plot_scanning = axis.plot(
                testabsatz.loc[(markt, 'UMSATZ_SCANNING')].index,
                testabsatz.loc[(markt, 'UMSATZ_SCANNING')].Menge,
                color='#0072B2',
                marker='.',
                linestyle='None'
            )
        except KeyError:
            pass
        try:
            plot_aktion = axis.plot(
                testabsatz.loc[(markt, 'UMSATZ_AKTION')].index,
                testabsatz.loc[(markt, 'UMSATZ_AKTION')].Menge,
                color='#D55E00',
                marker='.',
                linestyle='None'
            )
        except KeyError:
            pass
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axis.xaxis.set_major_formatter(DateFormatter('%m/%y'))
        axis.set_xlim([strahl[0], strahl[-1] + pd.DateOffset(1)])
        axis.set_ylim([0, max_absatz + 1])
        axis.set_yticks(range(int(max_absatz) + 1), minor=True)
        axis.set_yticks(range(0, int(max_absatz) + 1, 5), minor=False)
        if i == len(maerkte_mit_absatz) - 1:
            axis.set(xlabel='Datum')
        if i == int(len(maerkte_mit_absatz) / 2):
            axis.set(ylabel='Menge')
    plt.show()


plt.style.use('ggplot')
data_dir = os.path.join('files', 'raw')
artikelstamm = pd.read_csv(
            os.path.join(data_dir, '0 ArtikelstammV4.csv'),
            header=0,
            names=['Artikel', 'Warengruppe', 'Detailwarengruppe', 'Bezeichnung',
                   'Eigenmarke', 'Einheit', 'Verkaufseinheit', 'MHD',
                   'GuG', 'OSE', 'OSEText', 'Saisonal',
                   'Kern', 'Bio', 'Glutenfrei',
                   'Laktosefrei', 'MarkeFK', 'Region']
            )
warengruppenstamm = pd.read_csv(
            os.path.join(data_dir, '0 Warengruppenstamm.csv'),
            header=1,
            names=['WG', 'WGNr', 'WGBez', 'Abt', 'AbtNr', 'AbtBez']
        )
warenausgang = pd.read_csv(
    os.path.join(data_dir, '0 Warenausgang.Markt.csv'),
    header=1,
    names=['Markt', 'Artikel', 'Belegtyp', 'Menge', 'Datum']
)
ordersatz = pd.read_csv(
    os.path.join(data_dir, 'Ordersatz Mopro.csv'),
    sep=';',
    header=1
)
ordersatz.dropna(how='all', inplace=True)
ordersatz.dropna(axis=1, how='all', inplace=True)
ordersatz.drop(columns=['Unnamed: 4'], inplace=True)
ordersatz.dropna(how='any', inplace=True)
artikelstamm = artikelstamm[artikelstamm.Warengruppe.isin([68, 17])]
artikelstamm = pd.merge(artikelstamm, ordersatz, how='left', on='Bezeichnung')
artikelmaske = pd.unique(artikelstamm.Artikel)
artikelstamm.set_index('Artikel', inplace=True, drop=False)
warenausgang = warenausgang[warenausgang.Artikel.isin(artikelmaske)]
warenausgang['Datum'] = pd.to_datetime(warenausgang['Datum'], format='%d.%m.%y')
warenausgang = warenausgang.groupby(['Markt', 'Artikel', 'Datum', 'Belegtyp'], as_index=False).sum()
benutzte_artikel = pd.unique(warenausgang.Artikel)
artikelstamm = artikelstamm.loc[benutzte_artikel].copy()
artikelstamm = artikelstamm[~artikelstamm.Artikelnummer.isna()]
artikel_mit_ose = artikelstamm.Artikel.to_numpy()


show_timelines(warenausgang.Datum.min(), warenausgang.Datum.max())
# df = testabsatz.pivot_table(index='Datum', columns='Artikel', values='Menge').reindex(strahl)
# plt.figure()
# df.plot(style='.', subplots=True)
# plt.show()
#
# df['Jahr'] = df.index.year
# df['Monat'] = df.index.month
# df['Kalenderwoche'] = df.index.week
# df_year = df.groupby('Jahr').sum().drop(columns=['Kalenderwoche', 'Monat'])
# plt.figure()
# df_year.plot.bar()
# plt.show()
#
# df_month = df.groupby('Monat').sum().drop(columns=['Jahr', 'Kalenderwoche'])
# plt.figure()
# df_month.plot(subplots=True)
# plt.show()
#
# df_week = df.groupby('Kalenderwoche').sum().drop(columns=['Jahr', 'Monat'])
# plt.figure()
# df_week.plot(subplots=True)
# plt.show()

# TODO: Absätze aus mehreren Märkten aggregieren, um Zeitreihenverhalten zu analysieren und dann Grafik mit \
# Prozentualen anteilen der Märkte erstellen (Stacked Area Plot), um die Abhängigkeit von Menschen zu zeigen.
