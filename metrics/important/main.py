import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def gini_score(target, pred):
    return round(2 * roc_auc_score(target, pred) - 1, 4)


def roc_auc_score_round(target, pred):
    return round(roc_auc_score(target, pred), 4)


def f1_round(target, pred):
    return round(f1_score(target, pred > 0.5), 4)


def accuracy_score_round(target, pred):
    return round(accuracy_score(target, pred > 0.5), 4)


def all_metrics(target, pred):
    return [gini_score(target, pred), roc_auc_score_round(target, pred), f1_round(target, pred),
            accuracy_score_round(target, pred)]


def main(train=None, nm_train_target=None, nm_train_pred=None,
         oos=None, nm_oos_target=None, nm_oos_pred=None,
         oot=None, nm_oot_target=None, nm_oot_pred=None):
    results = dict()

    train_target = train[nm_train_target]
    train_pred = train[nm_train_pred]

    dict_score = {'sample 1': all_metrics(train_target, train_pred)}

    if oos is not None:
        oos_target = oos[nm_oos_target]
        oos_pred = oos[nm_oos_pred]
        dict_score['sample 2'] = all_metrics(oos_target, oos_pred)

    if oot is not None:
        oot_target = oot[nm_oot_target]
        oot_pred = oot[nm_oot_pred]
        dict_score['sample 3'] = all_metrics(oot_target, oot_pred)

    table_score = pd.DataFrame(data=dict_score).T

    table_score = table_score.reset_index().rename(columns={'index': '', 0: 'gini', 1: 'auc', 2: 'f1', 3: 'accuracy'})

    results['output_table_data'] = {'table_data': table_score.to_dict('records')}

    return results
