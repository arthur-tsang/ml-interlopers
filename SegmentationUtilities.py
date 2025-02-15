'''
Author: Bryan Ostdiek
Email: bostdiek@g.harvard.edu
This is the first pytorch UNet for strong graviational lensing.
Initializing takes the number of input chanels/colors (1) and the number of
output classes.

It is designed to work on images that are 80x80 pixels.
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Sampler
from torchvision import datasets, transforms
import numpy as np

import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.constants as const

from astropy.cosmology import default_cosmology
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel

from astropy.constants import G, c, M_sun

def get_paddings(in_size, out_size):
    """
    Helper function to calculate the amount of padding needed at the beginning
    and end of a UNet.
    """
    padding2_size = (out_size - 76)//4
    padding3_size = (80 - in_size)//2
    return padding2_size, padding3_size

def translate_dict(dico):
    """Helper function to interpret model weights"""
    newdico = {}
    length = len('module.')
    for k,v in dico.items():
        k_new = k[length:]
        newdico[k_new] = v
    return newdico

# ******************************************************************************
# Start of U-net
# ******************************************************************************
class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels,
                          kernel_size=3, padding=1, pdrop=0):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            padding=padding),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=pdrop),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            padding=padding
                            ),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=pdrop),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def expansive_block(self, in_channel, mid_channel, out_channels,
                        kernel_size=3, padding=1, stride=2, pdrop=0):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=in_channel,
                            out_channels=mid_channel,
                            padding=padding,
                            stride=stride
                            ),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=pdrop),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(in_channels=mid_channel,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride
                            ),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=pdrop),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels,
                    kernel_size=3, padding=1, pdrop=0,
                    final_relu=True):

        """
        This returns final block
        """
        if final_relu:
            block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size,
                                in_channels=in_channels,
                                out_channels=mid_channel,
                                padding=padding),
                torch.nn.ReLU(),
                torch.nn.Dropout2d(p=pdrop),
                torch.nn.BatchNorm2d(mid_channel),
                torch.nn.Conv2d(kernel_size=kernel_size,
                                in_channels=mid_channel,
                                out_channels=mid_channel,
                                padding=padding),
                torch.nn.ReLU(),
                torch.nn.Dropout2d(p=pdrop),
                torch.nn.BatchNorm2d(mid_channel),
                torch.nn.Conv2d(kernel_size=kernel_size,
                                in_channels=mid_channel,
                                out_channels=out_channels,
                                padding=padding),
                torch.nn.ReLU()
            )
        else:
            block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size,
                                in_channels=in_channels,
                                out_channels=mid_channel,
                                padding=padding),
                torch.nn.ReLU(),
                torch.nn.Dropout2d(p=pdrop),
                torch.nn.BatchNorm2d(mid_channel),
                torch.nn.Conv2d(kernel_size=kernel_size,
                                in_channels=mid_channel,
                                out_channels=mid_channel,
                                padding=padding),
                torch.nn.ReLU(),
                torch.nn.Dropout2d(p=pdrop),
                torch.nn.BatchNorm2d(mid_channel),
                torch.nn.Conv2d(kernel_size=kernel_size,
                                in_channels=mid_channel,
                                out_channels=out_channels,
                                padding=padding),
            )

        return block

    def __init__(self, in_channel, out_channel, kernel_size=3,
                 pdrop=0, final_relu=True, fac=64):
        """
        padding2_size=0 for 80x80 -> 76x76
        padding2_size=1 for 80x80 -> 80x80
        padding2_size=0; padding3_size=2 for 76x76 -> 76x76

        fac is the multiplicative factor on the number of channels in each block.
        In particular, it is equal to the number of channels output by the first block
        """
        
        super().__init__()
        padding = kernel_size // 2
        
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel,
                                                   out_channels=fac,
                                                   kernel_size=kernel_size,
                                                   padding=padding,
                                                   pdrop=pdrop
                                                  )
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(fac, 2*fac,
                                                   kernel_size=kernel_size,
                                                   padding=padding,
                                                   pdrop=pdrop
                                                  )
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(2*fac, 4*fac,
                                                   kernel_size=kernel_size,
                                                   padding=padding,
                                                   pdrop=pdrop
                                                  )
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=4*fac,
                            out_channels=8*fac,
                            padding=padding),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=pdrop),
            torch.nn.BatchNorm2d(8*fac),
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=8*fac,
                            out_channels=4*fac,
                            padding=padding
                            ),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=pdrop),
            torch.nn.BatchNorm2d(4*fac)
        )

        # Decode
        self.upsample3 = torch.nn.ConvTranspose2d(in_channels=4*fac,
                                                  out_channels=4*fac,
                                                  kernel_size=2,
                                                  stride=2
                                                  )
        self.conv_decode3 = self.expansive_block(8*fac, 4*fac, 2*fac,
                                                 kernel_size=kernel_size,
                                                 padding=padding,
                                                 stride=1,
                                                 pdrop=pdrop
                                                )
        self.upsample2 = torch.nn.ConvTranspose2d(in_channels=2*fac,
                                                  out_channels=2*fac,
                                                  kernel_size=2,
                                                  stride=2
                                                  )
        self.conv_decode2 = self.expansive_block(4*fac, 2*fac, fac,
                                                 kernel_size=kernel_size,
                                                 padding=padding,
                                                 stride=1,
                                                 pdrop=pdrop
                                                )
        self.upsample1 = torch.nn.ConvTranspose2d(in_channels=fac,
                                                  out_channels=fac,
                                                  kernel_size=2,
                                                  stride=2
                                                  )
        self.final_layer = self.final_block(2*fac, fac, out_channel,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            pdrop=pdrop,
                                            final_relu=final_relu
                                           )

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)

        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)

        # Decode
        upsampled3 = self.upsample3(bottleneck1)
        up_and_cat1 = torch.cat((upsampled3, encode_block3), 1)

        decode2 = self.conv_decode3(up_and_cat1)
        upsampled2 = self.upsample2(decode2)

        up_and_cat2 = torch.cat((upsampled2, encode_block2), 1)

        decode1 = self.conv_decode2(up_and_cat2)
        upsampled1 = self.upsample1(decode1)
        up_and_cat1 = torch.cat((upsampled1, encode_block1), 1)
        out = self.final_layer(up_and_cat1)

        return out

# ******************************************************************************
# End of U-net
# ******************************************************************************


# ******************************************************************************
# Start Datasets
# ******************************************************************************
class LensImages(Dataset):
    '''Lense image dataset'''

    def __init__(self, root_dir, transform=None, max_len=None,
                 startstring_image='val_image',
                 startstring_label='val_label',
                 start_num=0,
                 individualHalos=False,
                 RandomSample=None
                 ):
        '''
        args:
            root_dir (string): directory where the files are stored
                Assumes that they are .npy files
                Looks for all of the .npy files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        self.root_dir = root_dir
        self.images = sorted([x for x in os.listdir(self.root_dir) if (x.endswith('.npy') and
                                                                       x.startswith(startstring_image)
                                                                       )])
        self.labels = sorted([x for x in os.listdir(self.root_dir) if (x.endswith('.npy') and
                                                                       x.startswith(startstring_label)
                                                                       )])
        if max_len is None:
            max_len = len(self.images)
        if individualHalos:
            myimages = []
            mylabels = []
            for imname in self.images:
                imnum =  int(imname.split('_')[2][:4])
                if (imnum >= start_num) and (imnum < max_len):
                    myimages.append(imname)
            for imname in self.labels:
                imnum =  int(imname.split('_')[2][:4])
                if (imnum >= start_num) and (imnum < max_len):
                    mylabels.append(imname)
            self.images=myimages
            self.labels=mylabels
        elif RandomSample is not None:
            set_im = []
            set_label = []
            for i in range(len(self.images)):
                if i in RandomSample:
                    set_im.append(self.images[i])
                    set_label.append(self.labels[i])
            self.images = set_im
            self.labels = set_label
        else:
            self.images = self.images[start_num:max_len]
            self.labels = self.labels[start_num:max_len]
        # print(len(self.images))

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = os.path.join(self.root_dir, self.images[idx])
        label = os.path.join(self.root_dir, self.labels[idx])
        sample = np.load(image)
        label = np.load(label)

        if self.transform:
            sample = self.transform(sample)

        return(sample), label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = torch.from_numpy(sample).type(torch.float).unsqueeze(0)
        return image


class GenerateImages(Dataset):
    '''Lense image dataset'''

    def __init__(self, epoch_length=10000, transform=None, args=None):
        '''
        args:
            root_dir (string): directory where the files are stored
                Assumes that they are .npy files
                Looks for all of the .npy files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        self.epoch_length = epoch_length
        self.transform = transform
        self.args = args

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):

        if (self.args is not None) and ('devo' in self.args):
            image, subhalo_types_label, NumPixels, NumHalos = make_image_and_label(**self.args)
            NumHalos = np.array([NumHalos[x] for x in NumHalos])
            if self.transform:
                image = self.transform(image)
                NumHalos = torch.from_numpy(NumHalos).type(torch.float)

                return image, subhalo_types_label, NumPixels, NumHalos

        else:
            if (self.args is not None):
                image, label = make_image_and_label(**self.args)
            else:
                image, label = make_image_and_label()
            if self.transform:
                image = self.transform(image)

                return image, label

            
            
################################################################################

class CustomDataset(Dataset):
    """for images (or other data) stored in a folder as (number).npy files"""
    def __init__(self, root_dir, xfolder, yfolder, ytype=torch.IntTensor, maxlen=np.inf):
        self.root_dir = root_dir
        self.xfolder = xfolder
        self.yfolder = yfolder
        self.ytype = ytype
        self.maxlen = maxlen
    
    def __getitem__(self, index):
        x_filename = os.path.join(self.root_dir, self.xfolder, '{}.npy'.format(index))
        x_numpy = np.load(x_filename)[np.newaxis] # since the x input is only 1 color channel
        x_tensor = torch.from_numpy(x_numpy).float() # .to(device) # send to device now? (maybe specify float?)
        
        y_filename = os.path.join(self.root_dir, self.yfolder, '{}.npy'.format(index))
        y_numpy = np.load(y_filename)
        y_tensor = torch.from_numpy(y_numpy).type(self.ytype) # .to(device) # send to device now?

        return (x_tensor, y_tensor)

    def __len__(self):
        real_length = len(os.listdir(os.path.join(self.root_dir, self.yfolder)))
        return min(real_length, self.maxlen)

class CustomDatasetLayered(Dataset):
    """Like `CustomDataset` but takes in layers of xfolders"""
    def __init__(self, root_dir, xfolders, yfolder, ytype=torch.IntTensor, maxlen=np.inf, in_size=80):
        self.root_dir = root_dir
        self.xfolders = xfolders
        self.yfolder = yfolder
        self.ytype = ytype
        self.maxlen = maxlen
        self.in_size = in_size

    def __getitem__(self, index):
        x_arr = np.zeros((len(self.xfolders), self.in_size, self.in_size))
        for i, xfolder in enumerate(self.xfolders):
            x_filename = os.path.join(self.root_dir, xfolder, '{}.npy'.format(index))
            x_arr[i] = np.load(x_filename)
        x_tensor = torch.from_numpy(x_arr).float()


        y_filename = os.path.join(self.root_dir, self.yfolder, '{}.npy'.format(index))
        y_numpy = np.load(y_filename)
        y_tensor = torch.from_numpy(y_numpy).type(self.ytype)

        return (x_tensor, y_tensor)

    def __len__(self):
        real_length = len(os.listdir(os.path.join(self.root_dir, self.yfolder)))
        return min(real_length, self.maxlen)

class CustomDatasetLayered2(Dataset):
    """Like `CustomDatasetLayered` but takes in layers of xfolders and yfolders"""
    def __init__(self, root_dir, xfolders, yfolders, ytypes, maxlen=np.inf, in_size=80):
        self.root_dir = root_dir
        self.xfolders = xfolders
        self.yfolders = yfolders
        self.ytypes = ytypes
        self.maxlen = maxlen
        self.in_size = in_size

        ## `ytypes` can include torch.FloatTensor

    def __getitem__(self, index):
        x_arr = np.zeros((len(self.xfolders), self.in_size, self.in_size))
        for i, xfolder in enumerate(self.xfolders):
            x_filename = os.path.join(self.root_dir, xfolder, '{}.npy'.format(index))
            x_arr[i] = np.load(x_filename)
        x_tensor = torch.from_numpy(x_arr).float()

        y_tensor_list = []
        for yfolder, ytype in zip(self.yfolders, self.ytypes):
            y_filename = os.path.join(self.root_dir, yfolder, '{}.npy'.format(index))
            try:
                y_numpy = np.load(y_filename)
            except ValueError as e:
                print('Trying to load file {} and failing'.format(y_filename), flush=True)
                raise e
            y_tensor = torch.from_numpy(y_numpy).type(ytype)
            y_tensor_list.append(y_tensor)

        return (x_tensor, tuple(y_tensor_list))

    def __len__(self):
        real_length = len(os.listdir(os.path.join(self.root_dir, self.yfolders[0])))
        return min(real_length, self.maxlen)

class DatasetNoMassClass(CustomDatasetLayered2):
    """Like `CustomDatasetLayered2 but does not distinguish between different mass classes"""
    def __getitem__(self, index):
        x_tensor, y_tensor_list = super().__getitem__(index)
        new_y_tensor_list = tuple([torch.clamp(y_tensor, min=0, max=1) for y_tensor in y_tensor_list])
        return x_tensor, new_y_tensor_list
    
class CustomDatasetMult(Dataset):
    """for images (or other data) stored in a series of folders as (number).npy
    files. Unlike CustomDataset, CustomDatasetMult takes in a list of xfolders
    and yfolders (This is different from datasets with multiple input layers)"""
    def __init__(self, root_dir, xfolders, yfolders, ytype=torch.IntTensor, maxlen=np.inf):
        self.root_dir = root_dir
        self.xfolders = xfolders
        self.yfolders = yfolders
        self.ytype = ytype
        self.maxlen = maxlen

        self.calc_lengths() # calculate size of each folder
        print('folder lengths', self.lengths)

    def __getitem__(self, real_index):
        ## first calculate which folder the index belongs to
        index = real_index # `index` will be the index within the particular folder

        for i, folder_len in enumerate(self.lengths):
            if index >= folder_len:
                index -= folder_len
            else:
                folder_idx = i
                break

        x_filename = os.path.join(self.root_dir, self.xfolders[folder_idx], '{}.npy'.format(index))
        x_numpy = np.load(x_filename)[np.newaxis] # since the x input is only 1 color channel
        x_tensor = torch.from_numpy(x_numpy).float() # .to(device) # send to device now? (maybe specify float?)

        y_filename = os.path.join(self.root_dir, self.yfolders[folder_idx], '{}.npy'.format(index))
        y_numpy = np.load(y_filename)
        y_tensor = torch.from_numpy(y_numpy).type(self.ytype) # .to(device) # send to device now?

        return (x_tensor, y_tensor)

    def __len__(self):
        real_length = sum(self.lengths)
        return min(real_length, self.maxlen)

    def calc_lengths(self):
        self.lengths = [len(os.listdir(os.path.join(self.root_dir, yfolder)))
                          for yfolder in self.yfolders]
        lengths_alternate = [len(os.listdir(os.path.join(self.root_dir, xfolder)))
                          for xfolder in self.xfolders]
        assert(self.lengths == lengths_alternate)

class CheckpointSampler(Sampler):
    """Sampler that allows us to start anywhere in the middle of the dataset. Useful for training from a mid-epoch checkpoint."""
    
    def __init__(self, data_source, start_idx):
        self.data_source = data_source
        self.start_idx = start_idx
        assert(self.start_idx < len(self.data_source))
    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))
    def __len__(self):
        return len(self.data_source) - self.start_idx
