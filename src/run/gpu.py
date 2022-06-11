from . import run

def main(exps, batch_size=512, device='cuda'):
    device_params = {
        'estimator_params': {
            'tr_device':device,
            'tr_batch_size':batch_size,
        },
        'search_params': {
            'n_jobs':1,
            'pre_dispatch':'1*n_jobs',
            'verbose':True,
            'error_score':'raise',
        }
    }
    for i, e in enumerate(exps):
        run.run_experiment(i, e, device_params)
