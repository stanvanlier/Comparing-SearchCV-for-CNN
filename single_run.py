from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn_genetic import callbacks
from sklearn_genetic import GASearchCV

from src.estimators.cnn_classifier import CNNClassifier
from src.run import run
from src import models

batch_size=256

device_params = {
    'estimator_params': {
        'tr_device':'cpu',
        'tr_batch_size':batch_size,
    },
    'search_params': {
        'n_jobs':-1,
        'pre_dispatch':'2*n_jobs',
        'verbose':True,
        'error_score':'raise',
#            'log_config':None,
    }
}

problem_classes=[0,1]
exp = {
        # --- Number of times this experiment is repeated ---
        'trials': 1, 
        # --- Dataset as a subset of some it's targets
        'dataset': 'MNIST',
        'classes': problem_classes,
        # --- Estimator with corresponding hyperparameters ---
        # An estimator's hyperparameter should occur either in 'estimator_params' or in 'estimator_params_grid'
        'estimator': CNNClassifier,
        # fixed hyperparameters
        'estimator_params': {
            # training process hyperparameters
            'tr_criterion': 'NLLLoss',
            'tr_optimizer': 'Adam',
#            'tr_lr': 0.01,
            #'tr_epochs': 1,
            # model's hyperparameters
            'mo_n_conv_layers': 3,
            'mo_last_channels': 50,
            'mo_first_kernel_size': 5,
            'mo_last_kernel_size': 2,
            'mo_n_linear_layers': 2,
        },
        # varying hyperparameters
        'estimator_params_grid': {
            # training process hyperparameters
#            'tr_criterion': Categorical(['NLLLoss']),
#            'tr_optimizer': Categorical(['Adam']),
            'tr_lr': Continuous(0, 0.5),
            'tr_epochs': Integer(1,2),
            # model's hyperparameters
#            'mo_n_conv_layers': Integer(1,3),
#            'mo_last_channels': Integer(10,100),
#            'mo_first_kernel_size': Integer(4,6),
#            'mo_last_kernel_size': Integer(2,5),
#            'mo_n_linear_layers': Integer(1,5),
        },
        # --- Search method with corresponding parameters ---
        'search': GASearchCV,
        'search_params': dict( 
#            cv=model_selection.StratifiedKFold(n_splits=3, shuffle=True)
            cv=model_selection.StratifiedShuffleSplit(n_splits=2, test_size=0.3),
            scoring="accuracy",
            population_size=6,
            generations=3,
            crossover_probability=0.9,
            mutation_probability=0.05,
            algorithm="eaSimple",
        ),
        'search_callbacks': [ callbacks.DeltaThreshold(threshold=1e-2, metric="fitness"),
                              callbacks.TimerStopping(total_seconds=400) ],
    }
exp_i = 1

evolved_estimator = run.run_experiment(exp_i, exp, device_params, extra_str='cpu')
