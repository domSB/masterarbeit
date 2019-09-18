import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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
# ordersatz = pd.read_csv(
#     os.path.join(data_dir, 'Ordersatz Mopro.csv'),
#     sep=';',
#     header=1
# )
# ordersatz.dropna(how='all', inplace=True)
# ordersatz.dropna(axis=1, how='all', inplace=True)
# ordersatz.drop(columns=['Unnamed: 4'], inplace=True)
# ordersatz.dropna(how='any', inplace=True)
artikelstamm = artikelstamm[artikelstamm.Warengruppe.isin([68, 17])]
# erw_artikelstamm = pd.merge(artikelstamm, ordersatz, how='left', on='Bezeichnung')
artikelmaske = pd.unique(artikelstamm.Artikel)
warenausgang = warenausgang[warenausgang.Artikel.isin(artikelmaske)]
warenausgang['Datum'] = pd.to_datetime(warenausgang['Datum'], format='%d.%m.%y')
benutzte_artikel = pd.unique(warenausgang.Artikel)
# interesse = warenausgang.groupby('Artikel').sum()
#
idx = np.random.randint(0, len(benutzte_artikel), 1)
mask = benutzte_artikel[idx]
testabsatz = warenausgang[warenausgang.Artikel.isin(mask)]
strahl = pd.date_range('2016-01-01', '2019-6-30', freq='d')
testabsatz.set_index('Datum', inplace=True)
testabsatz = testabsatz.groupby(['Markt', 'Belegtyp']).apply(lambda x: x.reindex(strahl))
maerkte_mit_absatz = pd.unique(testabsatz.Markt)[~np.isnan(pd.unique(testabsatz.Markt))]
fig, axes = plt.subplots(nrows=len(maerkte_mit_absatz), ncols=1)
for axis, markt in zip(axes, maerkte_mit_absatz):
    try:
        plot_scanning = axis.plot(
            testabsatz.loc[(markt, 'UMSATZ_SCANNING')].index,
            testabsatz.loc[(markt, 'UMSATZ_SCANNING')].Menge,
            color='ob'
        )
        axis.legend(
            (plot_scanning[0],),
            ('Regulärer Umsatz',)
        )
    except KeyError:
        pass
    try:
        plot_aktion = axis.plot(
            testabsatz.loc[(markt, 'UMSATZ_AKTION')].index,
            testabsatz.loc[(markt, 'UMSATZ_AKTION')].Menge,
            'or'
        )
        axis.legend(
            (plot_aktion[0],),
            ('Aktionsumsatz',)
        )
    except KeyError:
        pass

plt.show()
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
