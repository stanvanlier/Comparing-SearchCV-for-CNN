import pandas as pd
import numpy as np
from glob import glob
import os
import json
import shutil

import matplotlib.pyplot as plt
import seaborn as sns


RAW_RESULTS = 'results'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

#%%
results = []
for ee in glob(f'{RAW_RESULTS}/*/*/search.json'):
    with open(ee, 'r') as f:
        d = json.load(f)
    row = {
        **{x: d[x] for x in [
            'extra__dataset',
            'extra__classes', 
            'extra__test_accuracy'
                             ]},
        **d['best_params_'],
        'Search Method': 'Genetic' if 'algorithm' in d.keys() else 'Random',
    }
    results.append(row)
results = pd.DataFrame(results)
results['N Classes'] = results.extra__classes.map(len)
results['Dataset'] = results.extra__dataset
results['Test Accuracy'] = results.extra__test_accuracy

sns.lineplot(data=results, x='N Classes', y='Test Accuracy', style='Dataset', hue='Search Method')
plt.savefig(f'{FIG_DIR}/Acc_over_N_Classes.png', dpi=300)

print(results.columns)

histplot_for_params = [
('tr_lr', 'Learning Rate'),
('mo_n_conv_layers', 'Number of Convolutional Layers'),
('mo_last_channels', 'Final Convolutional Layer Channels'),
#('mo_first_kernel_size',),
#('mo_n_linear_layers', ),
('mo_pooling', 'Pooling Method'),
('mo_activation', 'Activation'),
('mo_conv_order', 'Order after Each Convolution'),
]

plot_for_datasets = ['SVHN', 'MNIST']

# logy = True
logy = False
for ci, (col, nicename) in enumerate(histplot_for_params):
    fig, ax_ds = plt.subplots(1, len(plot_for_datasets), figsize=(7,3))
    subr = results[results.extra__dataset==ds].sort_values(col)
    for di, ds in enumerate(plot_for_datasets):
        ax = ax_ds[di]
        islastplot = di==len(plot_for_datasets)-1
        sns.histplot(ax=ax, data=subr, x=col,
                    hue='N Classes', multiple='stack', log_scale=(False, logy),
                    legend=islastplot,
                    )
        if islastplot:
            sns.move_legend(ax, (1.03,0))
        ax.set_title(ds)
        ax.set_xlabel(nicename)
        ax.set_ylabel('Occurance')
        if results[col].dtype == np.dtype('O'):
            ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/BestEsts_having_{col}.png', dpi=300)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11

plot_for_algos = ['Genetic', 'Random']
         
logy = False
for ci, (col, nicename) in enumerate(histplot_for_params):
    fig, ax_ds = plt.subplots(1, len(plot_for_algos), figsize=(7,3))
    subr = results[results['Search Method']==ds].sort_values(col)
    for di, ds in enumerate(plot_for_algos):
        ax = ax_ds[di]
        islastplot = di==len(plot_for_algos)-1
        sns.histplot(ax=ax, data=subr, x=col,
                    hue='N Classes', multiple='stack', log_scale=(False, logy),
                    legend=islastplot,
                    )
        if islastplot:
            sns.move_legend(ax, (1.03,0))
        ax.set_title(ds)
        ax.set_xlabel(nicename)
        ax.set_ylabel('Occurance')
        if results[col].dtype == np.dtype('O'):
            ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/AlgoBestEsts_having_{col}.png', dpi=300)

#####################################################


renamed = results.rename(columns={k:v for k,v in histplot_for_params})
renamed

#combi_vars = [v for _, v in histplot_for_params] + ['N Classes']
combi_vars = [
 'Learning Rate',
 'Number of Convolutional Layers',
 'Final Convolutional Layer Channels',
 #'Pooling Method',
 #'Activation',
 #'Order after Each Convolution',
 'N Classes',
]

for di, ds in enumerate(plot_for_datasets):
    g = sns.pairplot(
    data=renamed[renamed['Dataset'] == ds], 
    hue='Search Method',
    kind='scatter',
    vars=combi_vars,
    diag_kind='kde',
    plot_kws=dict(alpha=0.3),
    diag_kws=dict(alpha=0),
    corner=True,
    )
    plt.savefig(f'{FIG_DIR}/pairplot_{ds}.png', dpi=300)


shutil.make_archive(f'figures', 'zip', f'{FIG_DIR}')

