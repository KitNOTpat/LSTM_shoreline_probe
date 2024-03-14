# miscellaneous functions

import pickle
import torch
import matplotlib.pyplot as plt

#####################################################
#####################################################

def destandardize(arr, scalers, target):
    _arr_ = (arr*scalers[f'{target}_std'])+scalers[f'{target}_m']
    return _arr_

#####################################################
#####################################################

def standardize(arr, scalers, target):
    _arr_ = (arr - scalers[f'{target}_m'])/scalers[f'{target}_std']
    return _arr_

#####################################################
#####################################################

def plot_train_test(data, sl_var_name):

    fig, ax = plt.subplots(figsize=(12,4),facecolor='white')
    fig.tight_layout(pad=5.0)
    ax.grid(linestyle = '--', linewidth = 1, axis='both')
    ax.scatter(data.train.index, data.train[sl_var_name], c = 'royalblue', marker = 's', s = 10, label = 'Training')
    ax.scatter(data.test.index, data.test[sl_var_name], c = 'crimson', marker = 's', s = 10, label = 'Test')
    ax.legend(loc = 3);