from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn_genetic.callbacks import DeltaThreshold, TimerStopping
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import chain, combinations

import pandas as pd
from datetime import datetime

from .. import data

def subpowerset(iterable, minlen=0, maxlen=None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if maxlen is None:
        maxlen=len(s)
    return list(chain.from_iterable(combinations(s, r) for r in range(minlen, maxlen+1)))


def run_experiment(exp_i, exp, device_params, results_dir='results'):
    (X_train, y_train), (X_test, y_test) = data.utils.load_dataset(exp['dataset'],
        exp['classes'], new_sequential_classes=True)
    now_str = datetime.now().strftime("%Y%m%dT%H%M%S.%s")
    exp_dir = f'{results_dir}/exp{exp_i}_{now_str}'
    print(f"staring experiment {exp_i}. Saving in {exp_dir}")
    os.makedirs(exp_dir)
    with open(f'{exp_dir}/exp.str', 'w') as f:
        f.write(str(exp))
    with open(f'{exp_dir}/exp.repr', 'w') as f:
        f.write(repr(exp))
    
    os.makedirs(f'{exp_dir}/estimators')
    clf = exp['estimator'](_save_path=f'{exp_dir}/estimators', _X_test=X_test, _y_test=y_test,
        **exp['estimator_params'], **device_params['estimator_params'])
#    clf.save_path = exp_dir
#    clf._X_test = X_test
#    clf._y_test = y_test
#    clf.exp_i = exp_i
#    clf.exp = exp
    
    #cv = StratifiedKFold(n_splits=3, shuffle=True)
    
    delta_callback = DeltaThreshold(threshold=1e-4, metric="fitness")
    timer_callback = TimerStopping(total_seconds=800)
    callbacks = [delta_callback, timer_callback]

    for trail_i in range(exp['trials']):
#        clf.trail_i = trail_i
        evolved_estimator = exp['search'](
            clf, param_grid=exp['estimator_params_grid'],
            **exp['search_params'], **device_params['search_params'])
        starttime=time.time()
        evolved_estimator.fit(X_train, y_train, callbacks = callbacks)
        endtime=time.time()
        y_predict_ga = evolved_estimator.predict(X_test)
        accuracy = accuracy_score(y_test, y_predict_ga)
        trail_dir = f'{exp_dir}/trail{trail_i}'
        os.makedirs(trail_dir)
        try:
            cv_results_df = pd.DataFrame.from_dict(evolved_estimator.cv_results_)
            cv_results_df.to_csv(f'{trail_dir}/cv_results_.csv', index=False)   
        except e:
            print("cv_results_.csv not saved, becuase:", e)

        d = data.serialization.search_estimator2dict(evolved_estimator)
        d['extra__exp_i'] = exp_i
        d['extra__trail_i'] = trail_i
        d['extra__starttime'] = starttime
        d['extra__endtime'] = endtime
        d['extra__test_accuracy'] = accuracy
        data.serialization.savedict(f'{trail_dir}/evolved_estimator.json', d)

