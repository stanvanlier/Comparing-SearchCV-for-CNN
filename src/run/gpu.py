from . import run

def main(exp_q, batch_size=512, device='cuda'):
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
    while not exp_q.empty():
        i, exp = exp_q.get()
        run.run_experiment(i, exp, device_params, extra_str='gpu')
