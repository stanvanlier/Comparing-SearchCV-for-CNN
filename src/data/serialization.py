import json
from numpyencoder import NumpyEncoder

def savedict(filename, d):
    with open(filename, 'w') as f:
        json.dump(d, f, cls=NumpyEncoder, indent=2)

def search_estimator2dict(est):
    attrs_to_get = [
        '_n_iterations',
        '_pop',
        'algorithm',
        'best_index_',
        'best_params_',
        'best_score_',
        'criteria',
        'criteria_sign',
        'crossover_probability',
        'cv_results_',
        'elitism',
        'error_score',
        'generations',
        'history',
        'hof',
        'keep_top_k',
        'log_config',
        'logbook',
        'metrics_list',
        'multimetric_',
        'mutation_probability',
        'n_jobs',
        'n_splits_',
        'population_size',
        'pre_dispatch',
        'refit',
        'refit_metric',
        'refit_time_',
        'return_train_score',
        'scoring',
        'tournament_size',
        'verbose',
    ]
    return {x: getattr(est, x) for x in attrs_to_get if hasattr(est, x)}

