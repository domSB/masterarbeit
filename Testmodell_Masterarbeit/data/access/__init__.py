import os

import numpy as np
import pandas as pd

from ..preparation.clean import create_frame_from_raw_data
from ..preparation.to_numpy import create_numpy_from_frame


class Parameter(object):
    """
    Objekt zur Koordination der Daten-Aufbereitungs-Parameter
    """

    def __init__(self, **kwargs):
        """
        Speichert alle relevanten Parameter der Datenaufbereitung und kann an beliebige Funktionen weitergegeben werden.
        Wenn keine Parameter beim Initialisieren übergeben werden, werden Standardwerte verwendet.
        :param kwargs:
        """
        self.output_dir = kwargs.get('OutputDirectory',
                                     os.path.join('files', 'prepared'))

        self.input_dir = kwargs.get('InputDirectory',
                                    os.path.join('files', 'raw'))

        self.warengruppen_maske = kwargs.get('ZielWarengruppen', [55])

        self.detail_warengruppen_maske = kwargs.get('DetailWarengruppe', None)

        self.stat_state_category_cols = kwargs.get(
            'StatStateCategoricals',
            {
                'MHDgroup': 7,
                'Detailwarengruppe': None,
                'Einheit': None,
                'Markt': 6
            }
        )

        self.val_split = kwargs.get('ValidationSplit', 0.3)

    @property
    def name(self):
        """
        Dieser Name dient der Wiedererkennung zwischengespeicherter Ergebnisse mit den selben Parametern.
        :return: str aus Warengruppe und Detailwarengruppe, falls vorhanden.
        """
        if self.detail_warengruppen_maske:
            name = '-'.join([str(self.warengruppen_maske[0]),
                             str(self.detail_warengruppen_maske[0])])
        else:
            name = str(self.warengruppen_maske[0])
        return name

    @property
    def h5name(self):
        """
        Dieser Name dient der Wiedererkennung zwischengespeicherter Ergebnisse mit den selben Parametern.
        :return: str aus name und h5-Endung.
        """
        return self.name + ' store.h5'

    @property
    def npz_name(self):
        """
        Dieser Name dient der Wiedererkennung zwischengespeicherter Ergebnisse mit den selben Parametern.
        :return: str aus name und npz-Endung.
        """
        return self.name + ' store.npz'


class DataPipeLine(object):
    """
    Objekt für den Datenzugriff aus den einzelnen ausführbaren Dateien.
    Erleichtert Bedienung und ermöglicht identischen Zugriff auf vorberechnete oder nicht vorberechnete Daten.
    So müssen zur Ausführung nur die Rohdaten auf das System kopiert werden.
    """

    def __init__(self, **kwargs):
        """
        Die Initialisierung benötigt die Eigenschaften der zu untersuchenden Daten.
        Insbesondere die Warengruppe und die Detailwarengruppe. Auch können vom Standardverzeichnis abweichende
        Speicherpfade angegeben werden.
        :param kwargs:
        """
        self.params = Parameter(**kwargs)

    def check_for_raw_data(self):
        """
        Kurzes Script zum checken, ob alle Rohdaten auf dem System verfügbar sind.
        Wirft AssertationError bei fehlenden Daten.
        :return:
        """
        files = os.listdir(self.params.input_dir)
        assert '0 ArtikelstammV4.csv' in files, 'Artikelstamm fehlt'
        assert '0 Warenausgang.Markt.csv' in files, 'Warenausgang fehlt'
        assert '0 Wareneingang.Markt.csv' in files, 'Wareneingang fehlt'
        assert '0 Warenbestand.Markt.csv' in files, 'Warenbestand fehlt'
        assert '0 Aktionspreise.Markt.csv' in files, 'Aktionspreise fehlen'
        assert '1 Wetter.csv' in files, 'Wetter fehlt'
        assert '0 Preise.Markt.csv' in files, 'Preise fehlen'

    def get_regression_data(self):
        """
        Hauptmethode für den Datenzugriff.
        :return: lab, dyn, stat, split_helper
        """
        if self.params.npz_name in os.listdir(self.params.output_dir):
            print(
                'Vorberechnete Daten vorhanden\nLese Numpy Archiv-Dateien ...')
            files = np.load(
                os.path.join(self.params.output_dir, self.params.npz_name))
            lab = files['lab']
            dyn = files['dyn']
            stat = files['stat']
            split_helper = files['split_helper']
        elif self.params.h5name in os.listdir(self.params.output_dir):
            print(
                'Teilweise vorberechnete Daten vorhanden\n(1/2)\tLese HDF-Store ...')
            with pd.HDFStore(os.path.join(self.params.output_dir,
                                          self.params.h5name)) as store:
                absatz = store.get('Absatz')
                artikelstamm = store.get('Artikelstamm')
            print('(2/2)\tErstelle Numpy-Arrays aus DataFrames')
            lab, dyn, stat, split_helper = create_numpy_from_frame(self.params,
                                                                   absatz,
                                                                   artikelstamm)
            np.savez(
                os.path.join(self.params.output_dir, self.params.npz_name),
                lab=lab, dyn=dyn, stat=stat, split_helper=split_helper)

        else:
            print(
                'Keine vorberechneten Daten\n(1/2)\tErstelle DataFrames aus Rohdaten')
            absatz, bewegung, artikelstamm = create_frame_from_raw_data(
                self.params)
            print('Speichere neu berechnete Frames')
            with pd.HDFStore(os.path.join(self.params.output_dir,
                                          self.params.h5name)) as store:
                store.put('Artikelstamm', artikelstamm, format="table")
                store.put('Absatz', absatz)
                store.put('Bewegung', bewegung)
            print('(2/2)\tErstelle Numpy-Arrays aus DataFrames')
            lab, dyn, stat, split_helper = create_numpy_from_frame(self.params,
                                                                   absatz,
                                                                   artikelstamm)
            np.savez(
                os.path.join(self.params.output_dir, self.params.npz_name),
                lab=lab, dyn=dyn, stat=stat, split_helper=split_helper)

        return lab, dyn, stat, split_helper

    def get_statistics_data(self):
        """
        Methode gibt die Daten als Pandas-Dataframe zurück, damit diese weiter analysiert werden können.
        :return: absatz, bewegung, artikelstamm
        """
        if self.params.h5name in os.listdir(self.params.output_dir):
            print('Vorberechnete Daten vorhanden\nLese HDF-Store ...')
            with pd.HDFStore(os.path.join(self.params.output_dir,
                                          self.params.h5name)) as store:
                absatz = store.get('Absatz')
                bewegung = store.get('Bewegung')
                artikelstamm = store.get('Artikelstamm')
        else:
            print(
                'Keine vorberechneten Daten\nErstelle DataFrames aus Rohdaten')
            absatz, bewegung, artikelstamm = create_frame_from_raw_data(
                self.params)

            print('Speichere neu berechnete Frames')
            with pd.HDFStore(os.path.join(self.params.output_dir,
                                          self.params.h5name)) as store:
                store.put('Artikelstamm', artikelstamm, format="table")
                store.put('Absatz', absatz)
                store.put('Bewegung', bewegung)

        return absatz, bewegung, artikelstamm
