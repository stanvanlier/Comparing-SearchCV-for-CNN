# TPU imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

#SERIAL_EXEC = xmp.MpSerialExecutor()

from time import sleep

from . import run

from multiprocessing import Queue

def main(exps, batch_size=1024):
    exp_q = Queue()
    for i, exp in enumerate(exps):
        exp_q.put((i,exp))

    FLAGS = {}
    FLAGS['batch_size'] = batch_size
    # FLAGS['log_steps'] = 20
    # FLAGS['metrics_debug'] = False

    def _mp_fn(rank, flags, exp_q):
        torch.set_default_tensor_type('torch.FloatTensor')
        device = xm.xla_device()
        print(rank, device, repr(device), str(device))
        device_params = {
            'estimator_params': {
                'tr_device':device,
                'tr_batch_size':flags['batch_size'],
            },
            'search_params': {
                'n_jobs':1,
                'pre_dispatch':1,
                'verbose':False,
            }
        }
        exp_counter = 0
        while not exp_q.empty():
            i, exp = exp_q.get()
            exp_counter += 1
            print(rank, "is going to run exp", i,":", exp)
            run.run_experiment(i, exp, device_params, results_dir=f'resutls_tpu{rank}')
            print(rank, "exp", i ,"done, total ", exp_counter, "done until now")

        print("core", rank, 'is done, it ran', exp_counter, 'experiments')
        xm.rendezvous('init')
        if rank == 0:
            sleep(1)

    xmp.spawn(_mp_fn, args=(FLAGS, exp_q), nprocs=8 if os.environ.get('TPU_NAME', None) else 1,
              start_method='fork')
