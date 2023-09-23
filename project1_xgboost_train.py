import os
import time
import sys
import pandas as pd
import numpy as np
from pathlib import Path

import glob
import re

# Data split & Tuning params
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error


# XGBoost
import xgboost

from xgboost import XGBRegressor

# save model
import joblib

import hyperopt
from hyperopt import fmin, tpe, hp, anneal, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score


df = pd.read_csv("./train_processed_dropped_features_3.csv",index_col=0)
print(df.head())
df = df.drop(['PatientID'], axis=1)
df = df.reset_index(drop=True)

y = df['HeartDisease']
x = df.drop(['HeartDisease'], axis=1).reset_index(drop=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=65)

learning_rate_range = np.arange(0.001, 0.1, 0.002)
test_XG = [] 
train_XG = []

scale_pos_weight_value = sum(y_train == 0) / sum(y_train == 1)
param_space = {
    'objective':'binary:logistic',
    'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1.0),
    'learning_rate': hp.loguniform('learning_rate', -5, 1),
    'max_depth': hp.choice('max_depth', range(1, 40)),
    'reg_alpha' :  hp.quniform('reg_alpha', 0, 10, 0.01),
    'n_estimators': hp.choice('n_estimators', range(10, 400,1)),
    'gamma': hp.uniform ('gamma', 0, 9),
    'subsample': hp.uniform('subsample', 0.1, 1),
    'eta': hp.quniform('eta', 0.01, 1.0, 0.01),
    'reg_lambda' : hp.quniform('reg_lambda', 0, 10, 0.01),
    'scale_pos_weight': scale_pos_weight_value
}

def objective(params):
    model = xgboost.XGBClassifier(**params)
    # Use cross-validation for robustness
    cv_scores = cross_val_score(model, x, y, cv=5, scoring='f1')
    # Since we want to maximize accuracy, we return its negative as a loss to minimize
    return {'loss': -cv_scores.mean(), 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=500,  # can be increased for more exhaustive search
            trials=trials)

print("Best hyperparameters:\n", best)

best_model = xgboost.XGBClassifier(
    n_estimators=int(best['n_estimators']),
    eta=best['eta'],
    learning_rate=best['learning_rate'],
    max_depth=int(best['max_depth']), 
    gamma=best['gamma'],
    subsample=best['subsample'],
    colsample_bytree=best['colsample_bytree'],
    reg_alpha= best['reg_alpha'], 
    reg_lambda=best['reg_lambda'],
)

best_model.fit(x, y)

joblib.dump(best_model, 'project1_0920_4.joblib')
#joblib.dump(best_model, 'project1_0920_3.joblib') train_processed_dropped_features_2