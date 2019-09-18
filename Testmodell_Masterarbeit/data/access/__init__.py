import os
import numpy as np
import pandas as pd

from Testmodell_Masterarbeit.data.preparation.clean import create_frame_from_raw_data, create_numpy_from_frame


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
            self.warengruppenmaske = [1, 12, 55, 80, 17, 77, 71, 6, 28]
        if 'DynStateScalars' in kwargs:
            self.dyn_state_scalar_cols = kwargs.get('DynStateScalars')
        else:
            self.dyn_state_scalar_cols = ['Menge', 'MaxTemp_1D', 'MinTemp_1D', 'Wolken_1D',
                                          'Regen_1D', 'MaxTemp_2D', 'MinTemp_2D', 'Wolken_2D', 'Regen_2D',
                                          'Preis', 'relRabatt', 'absRabatt']
        if 'DynStateLabels' in kwargs:
            self.dyn_state_label_cols = kwargs.get('DynStateLabels')
        else:
            self.dyn_state_label_cols = ['in1', 'in2', 'in3', 'in4', 'in5']
        if 'DynStateCategoricals' in kwargs:
            self.dyn_state_category_cols = kwargs.get('DynStateCategoricals')
        else:
            self.dyn_state_category_cols = {'Wochentag': 7, 'Kalenderwoche': 54}
        if 'StatStateScalars' in kwargs:
            self.stat_state_scalar_cols = kwargs.get('StatStateScalars')
        else:
            self.stat_state_scalar_cols = ['Eigenmarke', 'GuG', 'OSE', 'Saisonal', 'Kern', 'Bio', 'Glutenfrei',
                                           'Laktosefrei']
        if 'StatStateCategoricals' in kwargs:
            self.stat_state_category_cols = kwargs.get('StatStateCategoricals')
        else:
            self.stat_state_category_cols = {
                'MHDgroup': 7,
                'Warengruppe': 9,
                'Detailwarengruppe': None,
                'Einheit': None
            }
        if 'DataGroup' in kwargs:
            self.data_group = kwargs.get('DataGroup')
        else:
            self.data_group = 'Markt'
        if 'TimeSeriesLength' in kwargs:
            self.ts_length = kwargs.get('TimeSeriesLength')
        else:
            self.ts_length = 6
        if 'TrainStartStop' in kwargs:
            self.train_start, self.train_stop = kwargs.get('TrainStartStop')
        else:
            self.train_start, self.train_stop = ('2017-01-01', '2017-12-31')
        if 'TestStartStop' in kwargs:
            self.test_start, self.test_stop = kwargs.get('TestStartStop')
        else:
            self.test_start, self.test_stop = ('2018-01-01', '2018-12-31')
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
        name = hash(
            (
                hash(frozenset(self.warengruppenmaske)),
                hash(frozenset(self.dyn_state_scalar_cols)),
                hash(frozenset(self.dyn_state_label_cols)),
                hash(frozenset(self.dyn_state_category_cols)),
                hash(frozenset(self.stat_state_scalar_cols)),
                hash(frozenset(self.stat_state_category_cols)),
                hash(self.data_group),
                hash(self.ts_length),
                hash((self.train_start, self.train_stop)),
                hash((self.test_start, self.test_stop)),
                hash(self.val_split)
            )
        )
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
        filename = self.params.get_name()
        if (filename + '.npz') in os.listdir(self.params.output_dir):
            print('Vorberechnete Daten vorhanden\nLese Numpy Archiv-Dateien ...')
            files = np.load(os.path.join(self.params.output_dir, (filename + '.npz')))
            lab = files['lab']
            dyn = files['dyn']
            stat = files['stat']
        elif (filename + '.h5') in os.listdir(self.params.output_dir):
            print('Teilweise vorberechnete Daten vorhanden\n(1/2)\tLese HDF-Store ...')
            with pd.HDFStore(os.path.join(self.params.output_dir, (filename + '.h5'))) as store:
                absatz = store.get('absatz')
                artikelstamm = store.get('artikelstamm')
            print('(2/2)\tErstelle Numpy-Arrays aus DataFrames')
            lab, dyn, stat = create_numpy_from_frame(self.params, absatz, artikelstamm)
        else:
            print('Keine vorberechneten Daten\n(1/2)\tErstelle DataFrames aus Rohdaten')
            absatz, _, artikelstamm = create_frame_from_raw_data(self.params)
            print('(2/2)\tErstelle Numpy-Arrays aus DataFrames')
            lab, dyn, stat = create_numpy_from_frame(self.params, absatz, artikelstamm)

        return lab, dyn, stat

    def get_simulation_data(self):
        filename = self.params.get_name()
        if (filename + '.h5') in os.listdir(self.params.output_dir):
            print('Vorberechnete Daten vorhanden\nLese HDF-Store ...')
            with pd.HDFStore(os.path.join(self.params.output_dir, (filename + '.h5'))) as store:
                absatz = store.get('absatz')
                artikelstamm = store.get('artikelstamm')
        else:
            print('Keine vorberechneten Daten\nErstelle DataFrames aus Rohdaten')
            absatz, _, artikelstamm = create_frame_from_raw_data(self.params)

        return absatz, artikelstamm

    def get_statistics_data(self):
        filename = self.params.get_name()
        if (filename + '.h5') in os.listdir(self.params.output_dir):
            print('Vorberechnete Daten vorhanden\nLese HDF-Store ...')
            with pd.HDFStore(os.path.join(self.params.output_dir, (filename + '.h5'))) as store:
                absatz = store.get('absatz')
                bewegung = store.get('bewegung')
                artikelstamm = store.get('artikelstamm')
        else:
            print('Keine vorberechneten Daten\nErstelle DataFrames aus Rohdaten')
            absatz, bewegung, artikelstamm = create_frame_from_raw_data(self.params)

        return absatz, bewegung, artikelstamm
