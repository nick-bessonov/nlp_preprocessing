from sklearn.metrics import *
import numpy as np
import pandas as pd


def mape(ytrue, ypred):
    return np.mean(np.abs((ytrue - ypred) / ytrue)) * 100


def smape(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred) / (np.abs(ytrue) + np.abs(ypred))) * 200


def median_absolute_percentage_error(ytrue, ypred):
    return np.median(np.abs((ytrue - ypred) / ytrue)) * 100


def recall_score_bin(ytrue, ypred):
    return recall_score(ytrue, np.round(ypred))


def precision_score_bin(ytrue, ypred):
    return precision_score(ytrue, np.round(ypred))


def f1_score_bin(ytrue, ypred):
    return f1_score(ytrue, np.round(ypred))


def accuracy_score_bin(ytrue, ypred):
    return accuracy_score(ytrue, np.round(ypred))


def f1_micro(ytrue, ypred):
    return f1_score(ytrue, ypred, average='micro')


def f1_macro(ytrue, ypred):
    return f1_score(ytrue, ypred, average='macro')


def precision_micro(ytrue, ypred):
    return precision_score(ytrue, ypred, average='micro')


def precision_macro(ytrue, ypred):
    return precision_score(ytrue, ypred, average='macro')


def recall_micro(ytrue, ypred):
    return precision_score(ytrue, ypred, average='micro')


def recall_macro(ytrue, ypred):
    return precision_score(ytrue, ypred, average='macro')


dct = {
    'ROC AUC': roc_auc_score,
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error,
    'F1': f1_score_bin,
    'F1 micro': f1_micro,
    'F1 macro': f1_macro,
    'R2': r2_score,
    'Recall': recall_score_bin,
    'Recall macro': recall_macro,
    'Recall micro': recall_micro,
    'Precision': precision_score_bin,
    'Precision micro': precision_micro,
    'Precision macro': precision_macro,
    'MSLE': mean_squared_log_error,
    'MedAE': median_absolute_error,
    'MAPE': mape,
    'SMAPE': smape,
    'MedAPE': median_absolute_percentage_error,
    'Accuracy': accuracy_score_bin,
    'AP': average_precision_score
}


def main(row=None, dev=None, prep=None, lemm=None, spell=None, row_target=None, row_score=None,
         dev_target=None, dev_score=None, prep_target=None, prep_score=None,
         lemm_target=None, lemm_score=None, spell_target=None, spell_score=None, metric=None,
         higher_is_better=True, comp=None):
    results = dict()

    if row is not None:
        row_metric = np.round(dct[metric](row[row_target], row[row_score]), 2)
    else:
        row_metric = '-'
    if dev is not None:
        dev_metric = np.round(dct[metric](dev[dev_target], dev[dev_score]), 2)
    else:
        dev_metric = '-'
    if prep is not None:
        prep_metric = np.round(dct[metric](prep[prep_target], prep[prep_score]), 2)
    else:
        prep_metric = '-'
    if lemm is not None:
        lemm_metric = np.round(dct[metric](lemm[lemm_target], lemm[lemm_score]), 2)
    else:
        lemm_metric = '-'
    if spell is not None:
        spell_metric = np.round(dct[metric](spell[spell_target], spell[spell_score]), 2)
    else:
        spell_metric = '-'

    indexes = ['Без обработки', 'Исходная обработка', 'Первичная обработка', 'Лемматизация и замена сущностей',
               'Исправление опечаток']
    metrics = [row_metric, dev_metric, prep_metric, lemm_metric, spell_metric]

    goal = 'Цель теста. Проверяется корректность и оптимальность используемых техник предобработки текста. '

    nothing = 0

    dct_metrics = {'Nothing': nothing,
                   'Without preprocessing': row_metric,
                   'Simple preprocessing': prep_metric,
                   'Lemmatization and NER': lemm_metric,
                   'Spell checking': spell_metric}

    if higher_is_better == True and dct_metrics[comp] != nothing:
        if ((dct_metrics[comp] - dev_metric) / dev_metric) <= 0.05:
            color = 'green'
            comment = goal + 'Альтернативная предобработка не приводит к повышению метрики качества на валидации.'
        else:
            if ((dct_metrics[comp] - dev_metric) / dev_metric) <= 0.1:
                color = 'amber'
                comment = goal + 'Альтернативная предобработка приводит к повышению метрики качества на валидации в пределах 5-10% от изначального уровня качества.'
            else:
                color = 'red'
                comment = goal + 'Альтернативная предобработка приводит к повышению метрики качества на валидации более чем на 10% от изначального уровня качества.'
    elif higher_is_better == False and dct_metrics[comp] != nothing:
        if ((dct_metrics[comp] - dev_metric) / dev_metric) >= -0.05:
            color = 'green'
            comment = goal + 'Альтернативная предобработка не приводит к повышению метрики качества на валидации.'
        else:
            if ((dct_metrics[comp] - dev_metric) / dev_metric) >= -0.1:
                color = 'amber'
                comment = goal + 'Альтернативная предобработка приводит к повышению метрики качества на валидации в пределах 5-10% от изначального уровня качества.'
            else:
                color = 'red'
                comment = goal + 'Альтернативная предобработка приводит к повышению метрики качества на валидации более чем на 10% от изначального уровня качества.'
    else:
        color = 'grey'
        comment = goal

    table = pd.DataFrame({'Вариант обработки': indexes, 'Метрика качества': metrics})

    results['table_data'] = {'table': table.to_dict('records')}
    results['semaphore'] = {'value': color,
                            'title': comment}

    return results