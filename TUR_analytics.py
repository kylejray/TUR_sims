import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.expanduser('~/source'))

from kyle_tools.utilities import inv_xtanhx


def plot_normal_TUR(N_samples, s_mean=50, std_range=10, constant_bounds=True):
    stds = np.random.uniform(std_range, size=N_samples)
    
    sigma_dist = np.random.normal(s_mean, stds, (10_000,N_samples) )
    s_mean = s_mean*(1-np.exp(-s_mean))

    sigma_dist = sigma_dist * (1-np.exp(-sigma_dist))
    min_bound = 1/np.average(np.tanh(sigma_dist/2),axis=0) -1

    fig, ax = plt.subplots()

    if constant_bounds is False:
        sigma_mean = np.average(sigma_dist, axis=0)
        HVV = 2/(np.exp(sigma_mean)-1)
        HG = 2/(sigma_mean)
        

        inv = []
        for i,item in enumerate(sigma_mean):
            print('\r {}of{}'.format(i,len(sigma_mean)),end='')
            inv.append(inv_xtanhx(item/2))
        new_inv = np.zeros(len(inv))
        for i, item in enumerate(inv):
            new_inv[i] = item

        TGGL = 1/(np.sinh(new_inv))**2
        for item in [HVV, HG, TGGL, min_bound]:
            ax.scatter(stds, item, s=1, marker=',')
    else:  
        ax.axhline(2/(np.exp(s_mean)-1), c='k', linestyle='--')
        ax.axhline(2/s_mean, c='r', linestyle='--')
        ax.axhline(1/(np.sinh(inv_xtanhx(s_mean/2))**2), c='g', linestyle='--')
        ax.scatter(stds, min_bound, s=1, marker=',' )
    
    ax.set_title('<sigma>={:.2f}'.format(s_mean)+'normally distributed')
    ax.legend(['HVV','HG', 'TGGL', 'J_min'])
    ax.set_yscale('log')
    ax.set_xlabel('std of sigma')
    ax.set_ylabel('$\\epsilon^2$ bound')
    plt.show()
