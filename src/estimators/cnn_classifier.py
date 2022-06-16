import numpy as np
import pandas as pd
import torch
import time
from datetime import datetime

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from .. import models

class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, _save_path, _X_test=None, _y_test=None, 
            tr_criterion='NLLLoss', 
            tr_optimizer='Adam', 
            tr_lr=0.001,
            tr_epochs=10, 
            tr_batch_size=32,
            tr_device='cpu', 

            mo_n_conv_layers=2, 
        #Note: first_channels can not be set, since it is determined on given input data.
            mo_last_channels=20, 
            mo_first_kernel_size=5,
            mo_last_kernel_size=3, 
            mo_n_linear_layers=2, 
        #Note: n_classes can not be set, since it is determined on number of targets in the dataset.
     ):
        self._save_path = _save_path
        now_str = datetime.now().strftime("%Y%m%dT%H%M%S.%f")
        self._X_test = _X_test
        self._y_test = _y_test

        self.tr_criterion = tr_criterion
        self.tr_optimizer = tr_optimizer
        self.tr_lr = tr_lr
        self.tr_epochs = tr_epochs
        self.tr_batch_size = tr_batch_size
        self.tr_device = tr_device

        self.mo_n_conv_layers = mo_n_conv_layers
        self.mo_last_channels = mo_last_channels
        self.mo_first_kernel_size = mo_first_kernel_size
        self.mo_last_kernel_size = mo_last_kernel_size
        self.mo_n_linear_layers = mo_n_linear_layers

        self._ith_use = 0

        self._tr_param_names = [x for x in dir(self) if x.startswith('tr_')] 
        self._mo_param_names = [x for x in dir(self) if x.startswith('mo_')]

    def fit(self, X, y):
        # Store the classes seen during fit
        # make sequential labels, and save original classes
        self._ith_use += 1
        
        self.classes_ = unique_labels(y)
        mo_kwargs = {k: getattr(self,k) for k in self._mo_param_names} 
        arg_dict = {k: getattr(self,k) for k in self._tr_param_names}
        arg_dict.update(mo_kwargs)
        arg_dict['classes_'] = self.classes_
        arg_dict['n_classes'] = len(self.classes_)
        arg_dict['X_shape'] = X.shape
        arg_dict['ith_use'] = self._ith_use

        # remove mo_ from the keywoard to pass it to the pytroch module
        mo_kwargs = {k[3:]: v for k,v in mo_kwargs.items()}
        now_dt = datetime.now()
        now_str = now_dt.strftime("%Y%m%dT%H%M%S.%f")
        self.model_ = models.cnn.CNN(mmap_path=f'{self._save_path}/mmap/{now_str}',
                                     first_channels=X.shape[1],
                                     n_classes=len(self.classes_),
                                     **mo_kwargs)

        self.model_.to(self.tr_device)
        Optim = getattr(torch.optim, self.tr_optimizer)
        self.optimizer_ = Optim(self.model_.parameters(), lr=self.tr_lr)

        Loss = getattr(torch.nn.modules.loss, self.tr_criterion)
        self.criterion_ = Loss()

        # TODO? add seed?

        arg_dict['train_start_time'] = time.time()
        train_loss, train_acc = models.utils.train(self.model_, self.optimizer_,
                                      self.criterion_, self.tr_epochs,
                                      self.tr_batch_size, self.tr_device, X, y)
        arg_dict['train_end_time'] = time.time()

        if self._X_test is not None and self._y_test is not None:
            arg_dict['test_start_time'] = time.time()
            test_loss, test_acc = models.utils.evaluate(self.model_, self.tr_batch_size*2, self.tr_device, 
                                           self._X_test, self._y_test, self.criterion_)
            arg_dict['test_end_time'] = time.time()
        else: 
            arg_dict['test_start_time'] = None
            arg_dict['test_end_time'] = None
            test_loss, test_acc = None, None
        arg_dict['train_loss'] = train_loss
        arg_dict['test_loss'] = test_loss
        arg_dict['train_accuracy'] = train_acc
        arg_dict['test_accuracy'] = test_acc
        arg_dict['now_dt'] = now_dt
        arg_dict['now_str'] = now_str
        pd.DataFrame([arg_dict]).to_csv(f'{self._save_path}/estimators/{now_str}.csv', index=False)
        # Return the classifier
        return self

    def _get_model_output(self, X):
        check_is_fitted(self)
        return models.utils.predict(self.model_, self.tr_batch_size*2, self.tr_device, X).numpy()

    def predict_proba(self, X):
        # Check if fit has been called
        return self._get_model_output(X)
    
    def predict(self, X):
        output = self._get_model_output(X)
        #TODO use self.classes_ in case that there are gaps in the classlabels
        # return output.argmax(dim=1)
        #return self.classes_[np.argmax(output, axis=1)]
        return np.argmax(output, axis=1)
