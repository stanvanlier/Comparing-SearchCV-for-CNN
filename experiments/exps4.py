from multiprocessing import Queue
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn_genetic import callbacks
from sklearn_genetic import GASearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.estimators.cnn_classifier import CNNClassifier
from src.run.run import subpowerset

problems = subpowerset([1,7,5,6,8,9], minlen=2, maxlen=3)
print(f'Making {len(problems)} experiments for each of these class subselections of the dataset: ')
print(problems)

experiments = Queue()
for i, problem_classes in enumerate(problems):
    experiments.put((i, {
        # --- Number of times this experiment is repeated ---
        'trials': 3, 

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
            # training process hyperparameters
#            'tr_criterion': Categorical(['NLLLoss']),
#            'tr_optimizer': Categorical(['Adam']),
            #'tr_lr': Continuous(0, 0.5),
#            'tr_epochs': Integer(1,10),
            # model's hyperparameters
            'mo_n_conv_layers': [1,2,3],
            'mo_last_channels': [10,20,30,40],
#            'mo_first_kernel_size': Integer(4,6),
#            'mo_last_kernel_size': Integer(2,5),
#            'mo_n_linear_layers': Integer(1,5),
        },

        # --- Search method with corresponding parameters ---
        'search': GridSearchCV,
        'search_params': dict( 
            #cv=model_selection.StratifiedKFold(n_splits=3, shuffle=True)
            cv=model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.3),
            scoring="accuracy",
            #population_size=10,
            #generations=10,
            #crossover_probability=0.9,
            #mutation_probability=0.03,
            #algorithm="eaSimple",
            return_train_score=True,
        ),
        'search_callbacks': [ callbacks.DeltaThreshold(threshold=1e-4, metric="fitness"),
                              callbacks.TimerStopping(total_seconds=800) ],
    }))

