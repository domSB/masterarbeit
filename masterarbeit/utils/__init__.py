"""
Modul mit mehreren klein Hilfsfunktionen und Hilfsklassen
"""

from collections import deque

import numpy as np
import yaml
from tensorflow.contrib.training import HParams


class Hyperparameter(HParams):
    """
    Klasse zum speichern der Hyperparemeter
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, file_path):
        """
        Läd die Parameter aus einer YAML-Datei
        :param file_path: Dateipfad
        :return:
        """
        with open(file_path, 'r') as file:
            for k, v in yaml.load(file, Loader=yaml.FullLoader).items():
                self.add_hparam(k, v)

    def save(self, file_path):
        """
        Speichert die Hyperparameter in eine YAML-Datei
        :param file_path:
        :return:
        """
        dic = self.values()
        with open(file_path, 'w') as file:
            _ = yaml.dump(dic, file, sort_keys=True)


class StateOperator:
    """
    Klasse zum identischen behandeln von LSTM-States und einfachen States
    """

    def __init__(self, hparams):
        self.use_lstm = hparams.use_lstm
        self.state_size = hparams.state_size
        self.time_steps = hparams.get('time_steps', 1)
        self.current_state = deque(maxlen=self.time_steps)
        self.preceding_state = deque(maxlen=self.time_steps)

    def start(self, initial_state):
        """
        Initialize Timeseries-State with first element
        :param initial_state:
        :return:
        """
        self.preceding_state = deque(maxlen=self.time_steps)
        for _ in range(self.time_steps):
            self.current_state.append(initial_state)

    def add(self, state):
        """
        Extend Timeseries state with new state-element
        :param state:
        :return:
        """
        self.preceding_state = self.current_state.copy()
        self.current_state.append(state)

    @property
    def state(self):
        """
        Current state as Numpy array
        :return: state
        """
        return np.array(self.current_state)

    @property
    def pre_state(self):
        """
        Preceding state as Numpy array
        :return: state
        """
        return np.array(self.preceding_state)
