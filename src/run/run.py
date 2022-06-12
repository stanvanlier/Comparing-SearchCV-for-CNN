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
import pprint
from glob import glob
import shutil

from .. import data

def subpowerset(iterable, minlen=0, maxlen=None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if maxlen is None:
        maxlen=len(s)
    return list(chain.from_iterable(combinations(s, r) for r in range(minlen, maxlen+1)))


def run_experiment(exp_i, exp, device_params, results_dir='results', extra_str=''):
    (X_train, y_train), (X_test, y_test) = data.utils.load_dataset(exp['dataset'],
        exp['classes'], new_sequential_classes=True)
    now_str = datetime.now().strftime("%Y%m%dT%H%M%S.%s")
    exp_str = f'exp{exp_i}_{now_str}'
    exp_dir = f'{results_dir}/{exp_str}'
    print(f"staring experiment {exp_i}. Saving in {exp_dir}")
    os.makedirs(exp_dir)
    with open(f'{exp_dir}/exp.dict', 'w') as f:
        f.write(pprint.pformat(exp))
    
    for trial_i in range(exp['trials']):
        now_str = datetime.now().strftime("%Y%m%dT%H%M%S.%s")
        trial_dir = f'{exp_dir}/trial{trial_i}_{now_str}'
        estimators_dir = f'{trial_dir}/estimators' 
        os.makedirs(estimators_dir)
        clf = exp['estimator'](_save_path=estimators_dir, _X_test=X_test, _y_test=y_test,
            **exp['estimator_params'], **device_params['estimator_params'])
        evolved_estimator = exp['search'](
            clf, param_grid=exp['estimator_params_grid'],
            **exp['search_params'], **device_params['search_params'])
        starttime=time.time()
        evolved_estimator.fit(X_train, y_train, callbacks=exp['search_callbacks'])
        endtime=time.time()

        (pd.concat([pd.read_csv(x) for x in glob(f'{estimators_dir}/*')], ignore_index=True)
        ).to_csv(f'{estimators_dir}.csv', index=False)

        try:
            cv_results_df = pd.DataFrame.from_dict(evolved_estimator.cv_results_)
            cv_results_df.to_csv(f'{trial_dir}/cv_results_.csv', index=False)   
        except e:
            print("cv_results_.csv not saved, becuase:", e)

        pred_proba = evolved_estimator.predict_proba(X_test)
        np.save(f'{trial_dir}/best_pred_proba', pred_proba, allow_pickle=False)
        np.savetxt(f'{trial_dir}/best_pred_proba', pred_proba)

        y_predict_ga = evolved_estimator.predict(X_test)
        accuracy = accuracy_score(y_test, y_predict_ga)
        d = data.serialization.search_estimator2dict(evolved_estimator)
        d['extra__extra_str'] = extra_str
        d['extra__dataset'] = exp['dataset']
        d['extra__classes'] = exp['classes']
        d['extra__exp_i'] = exp_i
        d['extra__trial_i'] = trial_i
        d['extra__starttime'] = starttime
        d['extra__endtime'] = endtime
        d['extra__test_accuracy'] = accuracy
        data.serialization.savedict(f'{trial_dir}/evolved_estimator.json', d)
    os.makedirs('ready_to_be_downloaded', exist_ok=True)
    sort_prefix = datetime.now().strftime("%dT%H%M%S")
#    shutil.make_archive(f'{exp_dir}', 'zip', f'ready_to_be_downloaded/{sort_prefix}_{exp_str}')
    shutil.make_archive(f'ready_to_be_downloaded/{sort_prefix}_{exp_str}', 'zip', f'{exp_dir}')
    print(f'  !! exp{exp_i} ready!       {sort_prefix}_{exp_str}.zip can now be downloaded manualy via the menu from ready_to_be_downloaded/. ({extra_str})')
#    try:
#        from google.colab import files
#        files.download(f"{exp_dir}.zip")
#    except:
#        print(f'{exp_dir}.zip was not downloaded automticaly. Download it manualy from the menu.')

