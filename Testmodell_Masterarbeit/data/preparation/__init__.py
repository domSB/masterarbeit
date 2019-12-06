import numpy as np
import tensorflow as tf


def split_np_arrays(lab, dyn, stat, split_helper, by_time=False, percentage=0.3,
                    only_one=False):
    """
    Hilfsfunktion um die Numpy-Arrays mit den Ausgangsdaten in einen Trainings-
    und Testdatensatz aufzuteilen. Generell wird nach Artikeln gespalten, dass
    Trainings- und Testdatensatz volle Zeitreihen von unterschiedlichen Artikeln
    enthalten.
    Ein Split nach der Zeitachse ist auch möglich, sodass jede Artikel-Zeitreihe
    in einen Trainings- und einen Testzeitraum unterteilt werden.
    Diese Methode muss verwendet werden, wenn der Agent mit nur einem Artikel
    trainiert wird.
    :param lab:
    :param dyn:
    :param stat:
    :param split_helper:
    :param by_time:
    :param percentage:
    :param only_one:
    :return:
    """
    assert max(split_helper[:, 0]) < 1000000

    idx = split_helper[:, 0] + split_helper[:, 1] * 1000000
    # Das splithelper Element enthält die Werte Markt-Nr. und Artikel-Nr.
    # Diese werden als eigener Array übergeben, da Numpy nur einen Index
    # verwendet und die Arrays eigentlich einen Tripel-Index haben.
    possibles = np.unique(idx)

    if by_time:
        uni, cnt = np.unique(idx, return_counts=True)
        tage = cnt[0]
        assert (cnt == tage).all(), 'Nicht die gleichen Zeitreihenlängen'
        test_tage = int(percentage * tage)
        train_tage = tage - test_tage
        single_mask = np.concatenate((np.ones((train_tage,), dtype=bool),
                                      np.zeros((test_tage,), dtype=bool)),
                                     axis=0)
        train_mask = np.tile(single_mask, len(possibles))
        test_mask = ~train_mask
    else:
        wahl = np.random.choice(len(possibles),
                                int(len(possibles) * percentage))
        test_mask = np.argwhere(np.isin(idx, possibles[wahl])).reshape(-1)
        train_mask = np.argwhere(
            np.isin(idx, possibles[wahl], invert=True)).reshape(-1)

    if only_one:
        train_artikel = idx[train_mask][np.random.choice(len(idx[train_mask]))]
        if by_time:
            test_artikel = train_artikel
            test_artikel_mask = np.isin(idx, test_artikel)
            test_mask = np.logical_and(test_artikel_mask, test_mask)
            train_mask = np.logical_and(test_artikel_mask, train_mask)
        else:
            test_artikel = idx[test_mask][np.random.choice(len(idx[test_mask]))]
            test_mask = np.argwhere(np.isin(idx, test_artikel)).reshape(-1)
            train_mask = np.argwhere(np.isin(idx, train_artikel)).reshape(-1)

    lab_test = lab[test_mask]
    lab_train = lab[train_mask]
    dyn_test = dyn[test_mask]
    dyn_train = dyn[train_mask]
    stat_test = stat[test_mask]
    stat_train = stat[train_mask]

    assert not np.isnan(lab_test).any()
    assert not np.isnan(dyn_test).any()
    assert not np.isnan(stat_test).any()
    return (lab_train, dyn_train, stat_train, idx[train_mask]), (
    lab_test, dyn_test, stat_test, idx[test_mask])


def create_dataset(_lab, _dyn, _stat, _params):
    """
    Erzeugt ein Tensorflow Dataset aus den Numpy Arrays
    :param _lab:
    :param _dyn:
    :param _stat:
    :param _params:
    :return:
    """
    def gen():
        while True:
            rand_idx = np.random.randint(0, _lab.shape[0])
            labels = _lab[rand_idx]
            yield {'dynamic_input': _dyn[rand_idx],
                   'static_input': _stat[rand_idx]}, \
                  {
                      '1day': labels[0],
                      '2day': labels[1],
                      '3day': labels[2],
                      '4day': labels[3],
                      '5day': labels[4],
                      '6day': labels[5],
                  }

    _dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=(
            {'dynamic_input': tf.float32, 'static_input': tf.int8},
            {
                '1day': tf.int8,
                '2day': tf.int8,
                '3day': tf.int8,
                '4day': tf.int8,
                '5day': tf.int8,
                '6day': tf.int8,
            }
        ),
        output_shapes=(
            {'dynamic_input': tf.TensorShape(
                [_params['time_steps'], _params['dynamic_state_shape']]),
             'static_input': tf.TensorShape([_params['static_state_shape']])},
            {
                '1day': tf.TensorShape([16]),
                '2day': tf.TensorShape([16]),
                '3day': tf.TensorShape([16]),
                '4day': tf.TensorShape([16]),
                '5day': tf.TensorShape([16]),
                '6day': tf.TensorShape([16]),
            }
        )
    )
    _dataset = _dataset.batch(_params['batch_size'])
    _dataset = _dataset.repeat()
    _dataset = _dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return _dataset
