import numpy as np


def split_np_arrays(lab, dyn, stat, split_helper, percentage=0.3):
    assert max(split_helper[:, 0]) < 1000000
    idx = split_helper[:, 0] + split_helper[:, 1] * 1000000
    possibles = np.unique(idx)
    wahl = np.random.choice(len(possibles), int(len(possibles) * percentage))
    args_test = np.argwhere(np.isin(idx, possibles[wahl])).reshape(-1)
    args_train = np.argwhere(np.isin(idx, possibles[wahl], invert=True)).reshape(-1)
    lab_test = lab[args_test]
    lab_train = lab[args_train]
    dyn_test = dyn[args_test]
    dyn_train = dyn[args_train]
    stat_test = stat[args_test]
    stat_train = stat[args_train]
    return (lab_train, dyn_train, stat_train), (lab_test, dyn_test, stat_test)
