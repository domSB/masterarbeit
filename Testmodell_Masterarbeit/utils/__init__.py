"""
Modul mit mehreren klein Hilfsfunktionen und Hilfsklassen
"""

import yaml
import os
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
