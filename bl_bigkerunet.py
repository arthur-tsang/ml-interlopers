import numpy as np
import os, sys
import datetime

import torch

# from SmaAtUNet.models.SmaAt_UNet_extralevel import SmaAt_UNet_extralevel
from bl_smaatunet import run_all

from SegmentationUtilities import UNet

import argparse

def model_generator_generator(fac, ker):
    def model_generator(n_channels, n_classes, pdrop):
        return UNet(n_channels, n_classes, kernel_size=ker, fac=fac)
    return model_generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('folder_postfix', type=str, help='example _subc_m8m11ninety_noise10_c60_hstelt3')
    parser.add_argument('maxlen', type=int)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('num_workers', type=int)
    parser.add_argument('pdrop', type=float)
    parser.add_argument('--pixnum', type=int, default=640)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--coordwide', action='store_true')
    parser.add_argument('--fac', type=int, default=64)
    parser.add_argument('--ker', type=int, default=3)
    
    args = parser.parse_args()
    
    folder_postfix = args.folder_postfix
    maxlen = args.maxlen
    batch_size = args.batch_size
    num_workers = args.num_workers
    pdrop = args.pdrop
    pixnum = args.pixnum
    lr = args.lr
    use_coordwide = args.coordwide
    fac = args.fac
    ker = args.ker
    
    print('args', args)
    
    # pin_memory = True
    pin_memory = False # trying to see if this affects the data loader
    separate_val_folders = True
    opt = 'adam'
    
    in_folders = [['in_cattrain' + folder_postfix], ['in_catval' + folder_postfix]]
    if use_coordwide:
        out_folders = [['coordwide_cattrain' + folder_postfix], ['coordwide_catval' + folder_postfix]]
    else:
        out_folders = [['coord_cattrain' + folder_postfix], ['coord_catval' + folder_postfix]]

    n_class = 2

    pdrop_string = str(int(pdrop * 100))
    weights = None
    # batch_size = 64
    quickmap = {20000: '2e4', 200000: '2e5', 2000000: '2e6', 6000000: '6e6',
                10000:'1e4', 100000:'1e5', 1000000:'1e6', 500000:'5e5'}

    if abs(lr - 1e-3) < 1e-6:
        lr_str = ''
    elif abs(lr - 2.5e-4) < 1e-6:
        lr_str = '_lquart'
    else:
        raise ValueError('Need name for corresponding learning rate')
    
    if use_coordwide:
        mname = 'bigkerUNet_blwide_{}_{}_drop{}_b{}{}_f{}_k{}'.format(
            'cat'+folder_postfix,
            quickmap[maxlen] if maxlen in quickmap else maxlen,
            pdrop_string,
            batch_size,
            lr_str,
            fac,
            ker)
    else:
        mname = 'bigkerUNet_bl_{}_{}_drop{}_b{}{}_f{}_k{}'.format(
            'cat'+folder_postfix,
            quickmap[maxlen] if maxlen in quickmap else maxlen,
            pdrop_string,
            batch_size,
            lr_str,
            fac,
            ker)

    unet_model = model_generator_generator(fac, ker)
    
    run_all(in_folders, out_folders, n_class, maxlen, pdrop, weights,
            batch_size, mname, pin_memory, num_workers, crop_outputs=False,
            normalization_scale=None, in_size=pixnum,
            separate_val_folders=separate_val_folders, val_maxlen=10000,
            unet_model=unet_model, val_data_offset=1000000,
            lr_init=lr)
