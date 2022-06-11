from  . import run

def main(exps, batch_size=256):
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
    for i, e in enumerate(exps):
        run.run_experiment(i, e, device_params)
