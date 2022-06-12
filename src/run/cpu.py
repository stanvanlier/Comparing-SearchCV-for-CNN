from  . import run

def main(exp_q, batch_size=256):
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
    while not exp_q.empty():
        i, exp = exp_q.get()
        run.run_experiment(i, exp, device_params, extra_str='cpu')
