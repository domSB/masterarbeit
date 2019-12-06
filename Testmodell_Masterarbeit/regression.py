"""
Hier sollen die Inputdaten auf den Absatz regressiert werden, um die
Vorhersagekraft zu bestimmen.
Ein Modell soll je Warengruppe erstellt werden.

1		BACKZUTATEN
6		TEXTILIEN
12		FERTIGGERICHTE
17		QUARK,JOGHURT
28		TIERNAHRUNG
55		GETRÄNKE ALKOHOLFREI
71		SUPPEN, SOSSEN
77		SPEISE- EIS
80		MINERALWASSER

"""

from agents import Predictor
from data.access import DataPipeLine
from data.preparation import split_np_arrays, create_dataset

warengruppe = 1
params = {
    'forecast_state': 6,
    'learning_rate': 0.0001,
    'time_steps': None,
    'dynamic_state_shape': None,
    'static_state_shape': None,
    'epochs': 50,
    'batch_size': 512
}

pipeline = DataPipeLine(ZielWarengruppen=[warengruppe])
lab, dyn, stat, split_helper = pipeline.get_regression_data()
train_data, test_data = split_np_arrays(lab, dyn, stat, split_helper)
params.update({
    'steps_per_epoch': int(train_data[1].shape[0] / params['batch_size']),
    'val_steps_per_epoch': int(test_data[1].shape[0] / params['batch_size']),
    'dynamic_state_shape': dyn.shape[2],
    'static_state_shape': stat.shape[1],
    'Name': '02RegWG' + str(warengruppe)
})
dataset = create_dataset(*train_data[:3], params)
val_dataset = create_dataset(*test_data[:3], params)
predictor = Predictor()
predictor.build_model(**params)
hist = predictor.train(dataset, val_dataset, params)
