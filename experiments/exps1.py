from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn_genetic.callbacks import DeltaThreshold, TimerStopping

from sklearn_genetic import GASearchCV

from src.estimators.cnn_classifier import CNNClassifier
from src.run.run import subpowerset

experiments = [{
        # --- Number of times this experiment is repeated ---
        'trails': 1, 

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
            'tr_epochs': 3,
            # model's hyperparameters
#            'mo_n_conv_layers': 3,
#            'mo_last_channels': 20,
            'mo_first_kernel_size': 5,
            'mo_last_kernel_size': 2,
            'mo_n_linear_layers': 2,
        },
        # varying hyperparameters
        'estimator_params_grid': {
            # training process hyper parameters
#            'tr_criterion': Categorical(['NLLLoss']),
#            'tr_optimizer': Categorical(['Adam']),
            'tr_lr': Continuous(0, 0.5),
#            'tr_epochs': Integer(1,10),
            # model's hyperparameters
            'mo_n_conv_layers': Integer(1,3),
            'mo_last_channels': Integer(10,100),
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
    } for problem_classes in subpowerset([1,2,3,4,5], 2,2)]
