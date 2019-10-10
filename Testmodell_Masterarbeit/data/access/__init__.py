import os
import numpy as np
import pandas as pd

from Testmodell_Masterarbeit.data.preparation.clean import create_frame_from_raw_data
from Testmodell_Masterarbeit.data.preparation.to_numpy import create_numpy_from_frame


class Parameter(object):
    def __init__(self, **kwargs):
        """
        Speichert alle relevanten Parameter der Datenaufbereitung und kann an beliebige Funktionen weitergegeben werden.
        Wenn keine Parameter beim Initialisieren übergeben werden, werden Standardwerte verwendet.
        :param kwargs:
        """
        if 'OutputDirectory' in kwargs:
            self.output_dir = kwargs.get('OutputDirectory')
        else:
            self.output_dir = os.path.join('files', 'prepared')

        if 'InputDirectory' in kwargs:
            self.input_dir = kwargs.get('InputDirectory')
        else:
            self.input_dir = os.path.join('files', 'raw')

        if 'ZielWarengruppen' in kwargs:
            self.warengruppenmaske = kwargs.get('ZielWarengruppen')
        else:
            self.warengruppenmaske = [17]

        if 'StatStateCategoricals' in kwargs:
            self.stat_state_category_cols = kwargs.get('StatStateCategoricals')
        else:
            self.stat_state_category_cols = {
                'MHDgroup': 7,
                'Warengruppe': 9,
                'Detailwarengruppe': None,
                'Einheit': None,
                'Markt': 6
            }

        if 'ValidationSplit' in kwargs:
            self.val_split = kwargs.get('ValidationSplit')
        else:
            self.val_split = 0.3

    def get_name(self):
        """
        Dieser Name dient der Wiedererkennung zwischengespeicherter Ergebnisse mit den selben Parametern.
        Da zu viele Parameter für einen String-Namen benutzt werden, werden alle Parameter gehashed und so aggregiert.
        :return: Hash aller Parameter, der als Dateiname dient.
        """
        name = self.warengruppenmaske
        return str(name)


class DataPipeLine(object):
    def __init__(self, **kwargs):
        """
        Dieses Objekt solle von allen ausführbaren Skripten verwendet werden, um auf die Daten zuzugreifen.
        Das verhindert redundante Datenaufbereitungen für verschiedene Aufgaben. Zudem können Zwischenergebnisse
        persistent gespeichert werden und müssen bei wiederholten Ausführungen nur gelesen werden.
        :param kwargs:
        """
        self.params = Parameter(**kwargs)
        pass

    def get_regression_data(self):
        filename = str(self.params.warengruppenmaske) + ' store'
        if (filename + '.npz') in os.listdir(self.params.output_dir):
            print('Vorberechnete Daten vorhanden\nLese Numpy Archiv-Dateien ...')
            files = np.load(os.path.join(self.params.output_dir, (filename + '.npz')))
            lab = files['lab']
            dyn = files['dyn']
            stat = files['stat']
            split_helper = files['split_helper']
        elif (filename + '.h5') in os.listdir(self.params.output_dir):
            print('Teilweise vorberechnete Daten vorhanden\n(1/2)\tLese HDF-Store ...')
            with pd.HDFStore(os.path.join(self.params.output_dir, (filename + '.h5'))) as store:
                absatz = store.get('Absatz')
                artikelstamm = store.get('Artikelstamm')
            print('(2/2)\tErstelle Numpy-Arrays aus DataFrames')
            lab, dyn, stat, split_helper = create_numpy_from_frame(self.params, absatz, artikelstamm)
            np.savez(
                os.path.join(self.params.output_dir, (filename + '.npz')),
                lab=lab, dyn=dyn, stat=stat, split_helper=split_helper)

        else:
            print('Keine vorberechneten Daten\n(1/2)\tErstelle DataFrames aus Rohdaten')
            absatz, bewegung, artikelstamm = create_frame_from_raw_data(self.params)
            print('Speichere neu berechnete Frames')
            with pd.HDFStore(os.path.join(self.params.output_dir, (filename + '.h5'))) as store:
                store.put('Artikelstamm', artikelstamm, format="table")
                store.put('Absatz', absatz)
                store.put('Bewegung', bewegung)
            print('(2/2)\tErstelle Numpy-Arrays aus DataFrames')
            lab, dyn, stat, split_helper = create_numpy_from_frame(self.params, absatz, artikelstamm)
            np.savez(
                os.path.join(self.params.output_dir, (filename + '.npz')),
                lab=lab, dyn=dyn, stat=stat, split_helper=split_helper)

        return lab, dyn, stat, split_helper

    def get_simulation_data(self):
        filename = str(self.params.warengruppenmaske) + ' store.h5'
        if filename in os.listdir(self.params.output_dir):
            print('Vorberechnete Daten vorhanden\nLese HDF-Store ...')
            with pd.HDFStore(os.path.join(self.params.output_dir, filename)) as store:
                absatz = store.get('Absatz')
                artikelstamm = store.get('Artikelstamm')
        else:
            print('Keine vorberechneten Daten\nErstelle DataFrames aus Rohdaten')
            absatz, bewegung, artikelstamm = create_frame_from_raw_data(self.params)

            print('Speichere neu berechnete Frames')
            with pd.HDFStore(os.path.join(self.params.output_dir, filename)) as store:
                store.put('Artikelstamm', artikelstamm, format="table")
                store.put('Absatz', absatz)
                store.put('Bewegung', bewegung)

        return absatz, artikelstamm

    def get_statistics_data(self):
        filename = str(self.params.warengruppenmaske) + ' store.h5'
        if filename in os.listdir(self.params.output_dir):
            print('Vorberechnete Daten vorhanden\nLese HDF-Store ...')
            with pd.HDFStore(os.path.join(self.params.output_dir, filename)) as store:
                absatz = store.get('Absatz')
                bewegung = store.get('Bewegung')
                artikelstamm = store.get('Artikelstamm')
        else:
            print('Keine vorberechneten Daten\nErstelle DataFrames aus Rohdaten')
            absatz, bewegung, artikelstamm = create_frame_from_raw_data(self.params)

            print('Speichere neu berechnete Frames')
            with pd.HDFStore(os.path.join(self.params.output_dir, filename)) as store:
                store.put('Artikelstamm', artikelstamm, format="table")
                store.put('Absatz', absatz)
                store.put('Bewegung', bewegung)

        return absatz, bewegung, artikelstamm
