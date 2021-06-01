import pandas as pd
import ast

from sklearn.linear_model import SGDClassifier, LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


def main(train=None, factors=None, train_nm_target=None,
         test=None, params='{}', type_model=None):  # обучение по выбору на одной из трех моделей

    results = dict()

    factors = [factors] if isinstance(factors, str) else factors
    y_train = train[train_nm_target]
    X_train = train[factors]
    X_test = test[factors]

    params = ast.literal_eval(params)

    if type_model == 'LGBMClassifier':
        model = LGBMClassifier(**params)
    elif type_model == 'SGDClassifier':
        model = SGDClassifier(**params)
    elif type_model == 'LogisticRegression':
        model = LogisticRegression(**params)
    elif type_model == 'RandomForestClassifier':
        model = RandomForestClassifier(**params)

    model.fit(X_train, y_train)

    y_train_pred = pd.DataFrame(data=model.predict_proba(X_train)[:, 1], columns=['predicted column'])
    y_test_pred = pd.DataFrame(data=model.predict_proba(X_test)[:, 1], columns=['predicted column'])

    train_pred = pd.concat([train, y_train_pred], axis=1)
    test_pred = pd.concat([test, y_test_pred], axis=1)

    results['model'] = model
    results['output_train_pred'] = train_pred
    results['output_test_pred'] = test_pred

    return results
