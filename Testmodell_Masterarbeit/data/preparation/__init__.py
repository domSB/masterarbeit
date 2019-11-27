import numpy as np


def split_np_arrays(lab, dyn, stat, split_helper, by_time=False, percentage=0.3, only_one=False):
    assert max(split_helper[:, 0]) < 1000000

    idx = split_helper[:, 0] + split_helper[:, 1] * 1000000
    possibles = np.unique(idx)

    if by_time:
        uni, cnt = np.unique(idx, return_counts=True)
        tage = cnt[0]
        assert (cnt == tage).all(), 'Artikel haben nicht alle die gleiche Zeitreihenlänge'
        test_tage = int(percentage * tage)
        train_tage = tage - test_tage
        single_mask = np.concatenate((np.ones((train_tage,), dtype=bool), np.zeros((test_tage,), dtype=bool)), axis=0)
        train_mask = np.tile(single_mask, len(possibles))
        test_mask = ~train_mask
    else:
        wahl = np.random.choice(len(possibles), int(len(possibles) * percentage))
        test_mask = np.argwhere(np.isin(idx, possibles[wahl])).reshape(-1)
        train_mask = np.argwhere(np.isin(idx, possibles[wahl], invert=True)).reshape(-1)

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
    return (lab_train, dyn_train, stat_train, idx[train_mask]), (lab_test, dyn_test, stat_test, idx[test_mask])
