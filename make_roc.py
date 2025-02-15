################################################################################
# make_roc.py
################################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from SegmentationUtilities import translate_dict
from SmaAtUNet.models.SmaAt_UNet import SmaAt_UNet

from scipy.special import softmax

import torch
import json
import argparse

def main(postfix, output_type='map'):
    """Here the postfix is something like `_catval_exp10_c30_br20_nsub1_hst` """
    
    assert output_type in ['map', 'binary']

    classes = ['Background',
               '$10^6$ - $10^{6.5}$', '$10^{6.5}$ - $10^7$',
               '$10^7$ - $10^{7.5}$', '$10^{7.5}$ - $10^8$',
               '$10^8$ - $10^{8.5}$', '$10^{8.5}$ - $10^9$',
               '$10^9$ - $10^{9.5}$', '$10^{9.5}$ - $10^{10}$',
               '$10^{10}$ - $10^{10.5}$', '$10^{10.5}$ - $10^{11}$']


    matplotlib.rcParams.update({'font.size': 15})

    assert postfix[0] == '_'
    out_range = range(1010000, 1020000)
        
    data_dir = '/n/holyscratch01/dvorkin_lab/Users/atsang/mif/'
    out_folder = 'stdout'+postfix
    coord_folder = 'coord'+postfix
    if '_hstelt3' in postfix:
        pixnum = 640
        pixsize = .01
    elif '_hstelt' in postfix:
        pixnum = 320
        pixsize = .02
    elif '_hst' in postfix:
        pixnum = 80
        pixsize = .08
    else:
        print('resolution/telescope not recognized')
    n_mass_class = 7

    # `datalist` is a list of (maximum) probabilities sorted by subhalo mass class
    datalist = [[] for _ in range(n_mass_class)]

    for seed in out_range:
        myout = np.load(os.path.join(data_dir, out_folder, f'{seed}.npy'))
        mycoord = np.load(os.path.join(data_dir, coord_folder, f'{seed}.npy'))

        if output_type == 'map':
            true_class = int(np.max(mycoord))
            sub_probs = np.sum(myout[1:], axis=0)
            max_prob = np.max(sub_probs)
            # sum_prob = np.sum(sub_probs)

            datalist[true_class].append(float(max_prob))
        else:
            raise RuntimeError('`output_type` not recognized')


    data_arrs = [np.array(l) for l in datalist]

    with open(f'tmp/rocdata{postfix}.json', 'w') as f:
        json.dump(datalist, f)

    print('Postfix:', postfix)

    print('At 10% false positive:')

    for mass_bin in [1,2,3,4]:
    # for mass_bin in [1,2,3,4,5,6]:
    # for mass_bin in [3, 4]:
        #data_sorted = np.sort(data_arrs[mass_bin])

        confidences = np.linspace(0, 1, 1000)

        true_hist = np.histogram(data_arrs[mass_bin], bins=confidences)[0]
        false_hist = np.histogram(data_arrs[0], bins=confidences)[0]

        true_raw = np.cumsum(true_hist[::-1])[::-1]
        false_raw = np.cumsum(false_hist[::-1])[::-1]

        tps = true_raw / np.sum(true_hist)
        fps = false_raw / np.sum(false_hist)

        try:
            print('{:.3f}'.format(tps[fps < 0.10][0]), end=', ')
        except IndexError:
            print('(Index error)', end=', ')

        plt.plot(fps, tps, label=classes[mass_bin+4]+' $M_\odot$')

    print()

    plt.legend(fontsize=14, loc='lower right')
    # plt.title('UNet')
        
    plt.plot([0,1], [0,1], 'grey', linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    plt.tight_layout()
    
    plt.savefig(f'tmp/roccurve{postfix}.pdf')
    plt.savefig(f'tmp/roccurve{postfix}.png')

    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('postfix', type=str)

    args = parser.parse_args()

    main(args.postfix)
