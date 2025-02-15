# save_nn_out.py

# code for saving the output of a neural net to files, so we can generate
# confusion matrices, ROC curves, etc.

import os
import argparse

import numpy as np

from scipy.special import softmax

import torch
from torch.utils.data import Dataset, DataLoader, Subset

# from SegmentationUtilities import UNet, CustomDatasetLayered2, get_paddings
from SegmentationUtilities import CustomDatasetLayered2
from SegmentationUtilities import translate_dict

#from MultiScaleAttention.src.models.my_stacked_danet import DAF_stack
# from SmaAtUNet.models.SmaAt_UNet import SmaAt_UNet
# from SmaAtUNetNoDrop.models.SmaAt_UNet import SmaAt_UNet as SmaAt_UNet_NoDrop
# from SmaAtUNet.models.SmaAt_UNet_doublechannel import SmaAt_UNet_doublechannel
# from SmaAtUNet.models.SmaAt_UNet_extralevel import SmaAt_UNet_extralevel

from data_gen_bright import safe_mkdir

# from align_pipe import MacroResnet
# from resnet import ResNetEstimator
# from baseline_resnet import FastDataset as FastDataset_resnet

# from bl_resnet import ResNetDataset, custom_resnet18
# from bl_unet import UNetDataset, custom_unet

from bl_bigkerunet import model_generator_generator as bigker_gg
# from bl_avgpool import model_generator_generator as avgpool_gg
# from bl_smel import model_generator_generator as smel_gg

# def custom_argmax(np_grid):
#     """Custom argmax-like function that will only return 0 (the background class) if
#     it has an absolute majority. Otherwise, it assigns the plurality mass bin."""

#     prob_grid = softmax(np_grid, axis=0)
#     bg_pixs = prob_grid[0] > 0.5
#     sub_pixs = np.argmax(prob_grid[1:], axis=0) + 1 # add back 1 since the [1:] messed with the indexing

#     return sub_pixs * (1 - bg_pixs)

def main(mname, in_folders, out_folder, coord_folders, root_dir, normalization_scale=None,
         datalen = 20000, data_offset = 180000, make_model=None, in_size=80,
         n_class = 9):

    ## Load model info
    loaded = torch.load('Models/{}.tar'.format(mname),
                        map_location=torch.device('cpu'))

    ### Make model

    # if make_model is custom_resnet18 or make_model is custom_unet:
    #     model = make_model(n_out=n_class)
    # elif make_model is SmaAt_UNet:
    #     model = make_model(len(in_folders), n_class)
    # elif 'avgpoolUNet' in mname or 'bigker' in mname or 'SmaAtel' in mname:
    #     model = make_model(1, 2, .1)
    # elif 'brain' in mname:
    #     model = make_model(1, 2, 0)
    # else:
    #     raise ValueError('make_model not recognized: {}'.format(make_model))

    ## Finally, we will assume all models follow the same format, regardless of name:
    assert make_model is not None
    model = make_model(1, 2, .1) # the dropout doesn't matter since we are evaluating
    
    ## Code to find the best i
    best_i = None
    for i in range(len(loaded['model_state_dict'])):
        if loaded['model_state_dict'][i] is not None:
            if best_i is None:
                best_i = i
    ## (Alternate code to fine last i)
    # best_i = -1

    model.load_state_dict(translate_dict(loaded['model_state_dict'][best_i]))
    model.eval()

    ## Data

    
    # if make_model is custom_resnet18:
    #     dataset = ResNetDataset(root_dir, in_folders[0], coord_folders[0])
    # elif make_model is custom_unet:
    #     dataset = UNetDataset(root_dir, in_folders[0], coord_folders[0])
    # elif make_model is SmaAt_UNet or 'avgpool' in mname or 'bigker' in mname or 'SmaAtel' in mname or 'brain' in mname:
    #     dataset = CustomDatasetLayered2(root_dir,
    #                                     in_folders,
    #                                     coord_folders,
    #                                     [torch.IntTensor],
    #                                     in_size=in_size)
    # else:
    #     raise ValueError

    dataset = CustomDatasetLayered2(root_dir,
                                    in_folders,
                                    coord_folders,
                                    [torch.IntTensor],
                                    in_size=in_size)
    
    batch_size = 1

    print('val dataset range:', data_offset, data_offset + datalen)
    val_dataset = Subset(dataset, range(data_offset, data_offset + datalen))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    ## make the output directory, if necessary
    safe_mkdir(os.path.join(root_dir, out_folder))

    model.eval()

    ## Loop through val_loader's images to visualize them and evaluate predictions

    with torch.no_grad():
        for i_batch, (x, label) in enumerate(val_loader):

            # coord = label[0][0].detach().numpy()

            # Note that we do not want to normalize if using a `custom_unet`
            # model, but the code already assumes we are not doing that
            # (`custom_unet` has to do with pre-training and fine tuning).

            # normalize input image
            for i in range(len(x)):
                normalization = torch.max(x[i]) if normalization_scale is None else normalization_scale
                x[i] = x[i] / normalization # normalize inputs

            ## calculate output
            out_tens = model(x)

            out_np = torch.nn.functional.softmax(out_tens[0], dim=0).detach().numpy() # not taking the argmax at all anymore
            np.save(os.path.join(root_dir, out_folder, '{}.npy'.format(i_batch + data_offset)), out_np)

# def main_macro_resnet(mname, in_folders, out_folder, coord_folders, root_dir,
#                 datalen, data_offset, use_last_epoch=True):

#     ## First, define the macro resnet object
#     use_shear = True
#     # mr = MacroResnet(mname, use_shear)
#     loaded = torch.load('Models/{}.tar'.format(mname),
#                         map_location=torch.device('cpu'))
#     n_out = 2
#     model = ResNetEstimator(n_out=n_out)

#     ## Code to find the best i
#     if use_last_epoch:
#         best_i = -1
#     else:
#         best_i = None
#         for i in range(len(loaded['model_state_dict'])):
#             if loaded['model_state_dict'][i] is not None:
#                 if best_i is None:
#                     best_i = i

#     model.load_state_dict(translate_dict(loaded['model_state_dict'][best_i]))
#     model.eval()

#     ## Then, load data

#     dataset_general = FastDataset_resnet(root_dir, in_folders[0], coord_folders[0], datalen)
#     dataset = Subset(dataset_general, range(data_offset, data_offset + datalen))
#     loader = DataLoader(dataset, batch_size=1)

#     ## make the output directory, if necessary
#     safe_mkdir(os.path.join(root_dir, out_folder))

#     with torch.no_grad():
#         for i_batch, (x, label) in enumerate(loader):

#             # normalize input:
#             assert x.shape[0] == 1 # namely that bach size is 1
#             norm_x = x / torch.max(x) # then the normalization step is simple

#             # we take x[0] because it's the only image in a batch size of 1
#             out_np = model(norm_x).detach().numpy()

#             # save
#             np.save(os.path.join(root_dir, out_folder, '{}.npy'.format(i_batch + data_offset)), out_np)

if __name__ == '__main__':
    # ## Model

    parser = argparse.ArgumentParser()
    parser.add_argument('mname', type=str)
    parser.add_argument('folder_postfix', type=str)
    parser.add_argument('in_size', type=int)
    parser.add_argument('out_prefix', type=str)
    parser.add_argument('startidx', type=int)
    parser.add_argument('endidx', type=int)
    parser.add_argument('--in_prefix', type=str, default='in')

    args = parser.parse_args()

    root_dir = '/n/holyscratch01/dvorkin_lab/Users/atsang/mif'

    ############################################################################
    # Each line below here needs to be customized depending on the particular model and data

    mname = args.mname
    folder_postfix = args.folder_postfix
    in_size = args.in_size
    out_prefix = args.out_prefix
    startidx = args.startidx
    endidx = args.endidx
    in_prefix = args.in_prefix

    in_folders = [in_prefix + folder_postfix]
    coord_folders = ['coord' + folder_postfix]
    out_folder = out_prefix + folder_postfix


    assert mname.split('_')[-2][0] == 'f'
    assert mname.split('_')[-1][0] == 'k'
    fac = int(mname.split('_')[-2][1:])
    ker = int(mname.split('_')[-1][1:])
    make_model = bigker_gg(fac, ker)

    main(mname, in_folders, out_folder, coord_folders, root_dir, normalization_scale=None,
         datalen = endidx-startidx, data_offset = startidx, make_model=make_model,
         in_size=in_size,
         n_class = 2)
