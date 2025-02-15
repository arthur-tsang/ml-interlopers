import numpy as np
import os, sys
import datetime

from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, TensorDataset, DataLoader
# from torch.utils.data.dataset import random_split
from torch.utils.data import Subset

from SegmentationUtilities import CustomDatasetLayered2
from SegmentationUtilities import DatasetNoMassClass
from SegmentationUtilities import CheckpointSampler
# from MoreNets import AttentionUNet
# from MoreNets import DeepAttention
# from MultiScaleAttention.src.models.my_stacked_danet import DAF_stack
from SmaAtUNet.models.SmaAt_UNet import SmaAt_UNet
from SmaAtUNet.models.SmaAt_UNet_doublechannel import SmaAt_UNet_doublechannel
from SegmentationUtilities import translate_dict

import pickle

def run_all(in_folders, out_folders, n_class, maxlen, pdrop, weights,
            batch_size, mname, pin_memory, num_workers, crop_outputs=True,
            data_offset=0,
            val_data_offset=None,
            val_maxlen=None,
            normalization_scale=None,
            root_folder='/n/holyscratch01/dvorkin_lab/Users/atsang/mif',
            unet_model=SmaAt_UNet,
            in_size=80, separate_val_folders=False,
            lr_init=1e-3):
    ################################################################################
    ### Things we vary on every run ###

    # folder_postfix = 'mediumvarylessql_int_m10'
    # in_folder = 'in_' + folder_postfix
    # curlbin_folder = 'curlbin_' + folder_postfix
    # n_class = 11
    # maxlen = int(2e5)
    # pdrop = .1
    # weights = np.ones(n_class)
    # batch_size = 64
    # mname = 'UNet_curlbin_{}_2e5_drop10_b64'.format(folder_postfix
    #                                                 if folder_postfix[-4:] != '_m10'
    #                                                 else folder_postfix[:-4])

    print('Variables')
    print('in_folders:', in_folders)
    print('out_folders:', out_folders)
    print('n_class', n_class)
    # assert(n_class == 9) # This line is not so relevant anymore.
    print('maxlen', maxlen)
    print('mname', mname)
    # print('weights', weights)
    print('forget about weights')
    assert(weights is None)
    print('pdrop', pdrop, 'which goes into DoubleConvDS for SmaAtUNet')
    print('batch size', batch_size)
    print('pin memory', pin_memory)
    print('num_workers', num_workers)
    print('data offset (index where dataset starts)', data_offset)

    print('normalization_scale', normalization_scale)
    print('root folder', root_folder)

    ################################################################################
    ## Now check that there really is as much data as we think there is ##

    if val_maxlen is None:
        val_maxlen = max(maxlen // 10, 10000)
    if val_data_offset is None:
        val_data_offset = data_offset + maxlen
    else:
        assert val_data_offset >= data_offset + maxlen

    if separate_val_folders:
        for in_folder in in_folders[0]:
            assert(len(os.listdir(os.path.join(root_folder, in_folder))) >= maxlen)
        for in_folder in in_folders[1]:
            assert(len(os.listdir(os.path.join(root_folder, in_folder))) >= val_maxlen)
        for out_folder in out_folders[0]:
            assert(len(os.listdir(os.path.join(root_folder, out_folder))) >= maxlen)
        for out_folder in out_folders[1]:
            assert(len(os.listdir(os.path.join(root_folder, out_folder))) >= val_maxlen)
        print('There is indeed (at least) as much data as we think.')
    else:
        for in_folder in in_folders:
            assert(len(os.listdir(os.path.join(root_folder, in_folder))) >= maxlen)
        for out_folder in out_folders:
            assert(len(os.listdir(os.path.join(root_folder, out_folder))) >= maxlen)
        print('There is indeed (at least) as much data as we think:', len(os.listdir(os.path.join(root_folder, in_folders[0]))))

    ################################################################################

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device, flush=True)

    KERNEL_SIZE = 3 # can also set to 5
    # mname = 'UNet_sub_m9_10000'
    # mname = 'UNet_curlbin_m10_1e4'
    # mname = 'UNet_5class_m10_20000'
    print('running mname', mname)

    ################################################################################
    ## Data loader
    ################################################################################

    if separate_val_folders:
        train_dataset_full = DatasetNoMassClass(root_folder,
                                                in_folders[0], out_folders[0],
                                                ytypes=[torch.IntTensor],
                                                maxlen=maxlen,
                                                in_size=in_size)
        train_datalen = len(train_dataset_full)
        train_dataset = Subset(train_dataset_full,
                               range(data_offset,
                                     data_offset + train_datalen))

        val_dataset_full = DatasetNoMassClass(root_folder,
                                              in_folders[1], out_folders[1],
                                              ytypes=[torch.IntTensor],
                                              maxlen=val_maxlen,
                                              in_size=in_size)
        val_datalen = len(val_dataset_full)
        val_dataset = Subset(val_dataset_full,
                             range(val_data_offset,
                                   val_data_offset + val_datalen))
        ## We put in this extra offset (train_datalen) above to ensure that
        ## there's no weird bias because all the analytical parameters
        ## (everything except the source's catalog index) are the same.

        print('train dataset length', train_datalen)
        print('val dataset length', val_datalen, flush=True)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
        
    else:
        dataset = DatasetNoMassClass(root_folder,
                                     in_folders, out_folders,
                                     ytypes=[torch.IntTensor], maxlen=maxlen,
                                     in_size=in_size)
        ############################################################################    

        datalen = len(dataset)
        print('dataset length', datalen, flush=True)
        n_train = int(0.9 * datalen)
        print('n_train', n_train)

        train_dataset = Subset(dataset, range(data_offset, data_offset + n_train))
        val_dataset = Subset(dataset, range(data_offset + n_train, data_offset + datalen))
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory)

    ################################################################################
    ## Train data
    ################################################################################

    ### Model ###

    # MyModel = SmaAt_UNet(len(in_folders), n_class, pdrop=pdrop)
    MyModel = unet_model(len(in_folders[0]) if separate_val_folders else len(in_folders),
                         n_class,
                         pdrop=pdrop)
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        MyModel = nn.DataParallel(MyModel).to(device)


    ### Optimizer ###
    # class_weights = torch.from_numpy(np.ones(n_class)).float().to(device)
    # class_weights = torch.from_numpy(weights).float().to(device)
    # assert(len(class_weights) == n_class)

    optimizer = optim.Adam(MyModel.parameters(), lr=lr_init)
    save_path = 'Models/{}.tar'.format(mname)
    chk_path = 'Models/chk_{}.tar'.format(mname)
    init_path = 'Models/init_{}.tar'.format(mname) # just a quick slightly hacky way of initializing on a nicer set of weights
    if os.path.isfile(save_path):
        # if a regular save file exists
        print('continuing partially-trained model')
        loaded = torch.load(save_path)


        model_dict_updt = {'model_state_dict': loaded['model_state_dict'],
                           'TrainingLoss': loaded['TrainingLoss'],
                           'ValidationLoss': loaded['ValidationLoss']}

        optimizer.load_state_dict(loaded['optimizer_state_dict'])
        start_epoch = loaded['epoch']

        loaded_chk = torch.load(chk_path) if os.path.isfile(chk_path) else None
        if loaded_chk and loaded_chk['epoch'] >= start_epoch:
            # if found checkpoint file (for saving in the middle of an epoch)
            # and the checkpoint is new enough to be useful

            assert(loaded_chk['epoch'] == start_epoch) # otherwise we did something wrong
            print('Using checkpoint file')

            MyModel.load_state_dict(loaded_chk['tmp_checkpoint'])
            checkpoint_batch_nr = loaded_chk['batch_nr']
            use_chk_file = True            
        else:
            # if not found checkpoint file, then we just use the regular save file
            print('Not using checkpoint file (may or may not exist)')

            MyModel.load_state_dict(loaded['model_state_dict'][-1])
            use_chk_file = False
    elif os.path.isfile(chk_path):
        # if there is a checkpoint file, but we haven't completed a full epoch yet
        loaded =  torch.load(chk_path)

        model_dict_updt = {'model_state_dict': [],
                           'TrainingLoss': [],
                           'ValidationLoss': []}

        # MyModel.load_state_dict(translate_dict(loaded['model_state_dict'][-1]))
        start_epoch = loaded['epoch']
        assert(start_epoch == 0)
        
        print('Using checkpoint file (no full-epoch save file found)')
        
        MyModel.load_state_dict(loaded['tmp_checkpoint'])
        checkpoint_batch_nr = loaded['batch_nr']
        use_chk_file = True
    else:
        # if no save file exists whatsoever
        model_dict_updt={'model_state_dict': [],
                         'TrainingLoss': [],
                         'ValidationLoss': []}
        start_epoch = 0
        use_chk_file = False

        if os.path.isfile(init_path):
            print('No save file found, but using initalization file {}'.format(init_path))
            
            loaded = torch.load(init_path)
            init_state_dict = loaded['model_state_dict'][-1] if 'model_state_dict' in loaded else loaded['tmp_checkpoint']
            MyModel.load_state_dict(init_state_dict)
            del loaded


    ############################################################################
    # In case we are loading from a checkpoint, we'll need a special DataLoader
    # that starts where the checkpoint starts.
    if use_chk_file:
        chk_sampler_start = (checkpoint_batch_nr + 1) * batch_size # off-by-one to not redo checkpoint batch again after saving
        chk_sampler = CheckpointSampler(train_dataset, chk_sampler_start)
        chk_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=chk_sampler,
                                num_workers=num_workers, pin_memory=pin_memory)

    ############################################################################


    Scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     patience=5,
                                                     verbose=True,
                                                     min_lr=1e-6)
    

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    ############################################################################
    ### Training loop ###
    ############################################################################
    model_checkpoints = model_dict_updt['model_state_dict']
    epoch_dict = {'TrainingLoss': model_dict_updt['TrainingLoss'],
                  'ValidationLoss': model_dict_updt['ValidationLoss'],
                  'Save': mname}
    min_loss = np.inf
    stopping_patience_START = 15 # used to be 15
    stopping_patience = stopping_patience_START


    # (Cropping output of model for compatibility with data)
    full_slice = slice(None, None)
    cropped_slice = slice(2, -2)
    output_slice = ((full_slice, full_slice, cropped_slice, cropped_slice)
                    if crop_outputs
                    else (full_slice, full_slice, full_slice, full_slice))

    # UPDATENUM = len(model_dict_updt['model_state_dict'])
    # UPDATENUM += 1
    for epoch in range(1000):
        if epoch < start_epoch:
            continue

        print('time stamp:', datetime.datetime.now(), flush=True)

        MyModel.train()
        train_loss = 0
        train_size = 0
        if use_chk_file and epoch == start_epoch:
            relevant_train_loader = chk_loader
            i_batch_offset = checkpoint_batch_nr + 1
        else:
            relevant_train_loader = train_loader
            i_batch_offset = 0

        for i_batch,(x, labels) in enumerate(relevant_train_loader):
            ## Check for nans
            if torch.sum(torch.isnan(x)) > 0:
                print('i_batch', i_batch, 'x contains nan')
                torch.save(x, 'tmp/x_with_nan.npy')
                raise ValueError
            elif torch.sum(torch.isnan(labels[0])) > 0:
                print('i_batch', i_batch, 'label contains nan')
                raise ValueError

            # if use_chk_file and epoch == start_epoch and i_batch <= checkpoint_batch_nr:
            #     continue
            
            # (add offset because enumerate will start from 0 even if the loader
            # starts from the middle of the dataset -- only relevant if using chk_loader)
            i_batch = i_batch + i_batch_offset 

            # used to save every 1000, but this can get a bit too slow
            if i_batch % 500 == 0 and i_batch != 0:
                torch.save({'tmp_checkpoint':MyModel.state_dict(),
                            'epoch':epoch,
                            'batch_nr':i_batch},
                           'Models/chk_{}.tar'.format(epoch_dict['Save']))
                print('saved checkpoint at epoch {} batch {}'.format(epoch, i_batch), flush=True)
                print('time stamp:', datetime.datetime.now(), flush=True)

            for i in range(len(x)):
                normalization = torch.max(x[i]) if normalization_scale is None else normalization_scale
                x[i] = x[i] / normalization # normalize inputs

            x = x.to(device) # .double()
            optimizer.zero_grad()
            torch.cuda.empty_cache() # hopefully this will solve the memory issues

            try:
                outputs = MyModel(x)
            except RuntimeError as e:
                print('x shape', x.shape)
                raise e

            ## The `labels` we get from the trainloader comes as a tuple of
            ## outputs, which we want to stack as different channels of a single
            ## output.
            # labels = torch.stack([labels[0], labels[1]], 1).to(device).float()
            labels = labels[0].to(device).long()
            ## Since we're using MSE, calculating the loss for the x and y
            ## components separately is equivalent to calculating the loss based
            ## distance between vectors.
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.shape[0]
            train_size += x.shape[0]
            if i_batch % 50 == 0: # used to be 100, but we're using such a small dataset this time
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i_batch * len(x), len(train_loader.dataset),
                    100. * i_batch / len(train_loader),
                    loss.item() ))

            if loss.item() != loss.item():
                print('Loss is nan')
                raise ValueError

        # end batch

        epoch_dict['TrainingLoss'].append(train_loss / train_size)

        val_loss = 0
        val_samples = 0
        MyModel.eval()

        with torch.no_grad(): # disable gradient calculation while validating
            for i_batch, (x, labels) in enumerate(val_loader):
                if i_batch % 10 == 0:
                    print('    Validation Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        epoch, i_batch * len(x), len(val_loader.dataset),
                        100. * i_batch / len(val_loader)
                        ))
                for i in range(len(x)):
                    normalization = torch.max(x[i]) if normalization_scale is None else normalization_scale
                    x[i] = x[i] / normalization # normalize inputs

                x = x.to(device) # .double()
                # print('validation x shape', x.shape)
                torch.cuda.empty_cache() # hopefully this will solve the memory issues
                outputs = MyModel(x) # seems the output gets all wrong when we get to eval mode

                ## The `labels` we get from the trainloader comes as a tuple of
                ## outputs, which we want to stack as different channels of a single
                ## output.
                # labels = torch.stack([labels[0], labels[1]], 1).to(device).float()
                labels = labels[0].to(device).long()
                ## Since we're using MSE, calculating the loss for the x and y
                ## components separately is equivalent to calculating the loss based
                ## distance between vectors.
                loss = criterion(outputs, labels)

                val_loss += loss.item() * x.shape[0]
                val_samples += x.shape[0]

        val_loss_per_sample = val_loss / val_samples
        epoch_dict['ValidationLoss'].append(val_loss_per_sample)
        print('Validation loss = {0:.4f}'.format(val_loss_per_sample))

        model_checkpoints.append(deepcopy(MyModel.state_dict()))
        if len(model_checkpoints) > 1:
            # delete non-optimal weights to save space!!
            # (note that we never delete the most recent weights)
            optimal_epoch = np.argmin(epoch_dict['ValidationLoss'])

            for i in range(len(model_checkpoints) - 1):
                if i != optimal_epoch:
                    model_checkpoints[i] = None

        # UPDATENUM += 1


        if val_loss_per_sample < min_loss:
            min_loss = val_loss_per_sample
            stopping_patience = stopping_patience_START
        else:
            stopping_patience -= 1

        torch.save({'epoch': epoch + 1,
                    'model_state_dict': model_checkpoints,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'TrainingLoss': epoch_dict['TrainingLoss'],
                    'ValidationLoss': epoch_dict['ValidationLoss']},
                   os.path.join('Models', epoch_dict['Save'] + '.tar'))

        Scheduler.step(val_loss_per_sample)

        if stopping_patience == 0:
            print('Model has not improved in {0} epochs...stopping early'.format(stopping_patience_START))
            break

    # end epoch

if __name__ == '__main__':
    """
    New example syntax:

    python -E bl_unet.py _subc_m9m11_noise10_c60_79pix 100000 8 $SLURM_NTASKS 0.00 0 1e-3 adam 79
    """
    
    assert len(sys.argv) == 11
    folder_postfix = sys.argv[1]
    maxlen = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    num_workers = int(sys.argv[4])
    pdrop = float(sys.argv[5])
    fine_tuning_code = int(sys.argv[6])
    lr = float(sys.argv[7])
    opt = sys.argv[8]
    assert opt in ['adam', 'sgd']
    pixnum = int(sys.argv[9])
    use_coordwide = bool(int(sys.argv[10]))
    
    ## (Interpreting the fine-tuning code number)
    fine_tuning_code_names = ['(none)', 'fine tuning (frozen)', 'fine tuning (unfrozen)']
    fine_tuning_names_short = ['noft', 'ftfreeze', 'ftfinal']
    assert fine_tuning_code == 0

    ## For tracking purposes, print all our inputs
    print('folder_postfix', folder_postfix)
    print('maxlen', maxlen)
    print('batch_size', batch_size)
    print('num_workers', num_workers)
    print('pdrop', pdrop)
    print('fine tuning code', fine_tuning_code, fine_tuning_code_names[fine_tuning_code])
    print('lr', lr)
    print('optimizer', opt)

    assert ('train' not in folder_postfix) and ('val' not in folder_postfix)

    pin_memory = True
    separate_val_folders = True

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
                10000:'1e4', 100000:'1e5', 1000000:'1e6'}

    if use_coordwide:
        mname = 'SmaAtUNet_blwide_{}_{}_drop{}_b{}_{}_{}'.format(
            'cat'+folder_postfix,
            quickmap[maxlen] if maxlen in quickmap else maxlen,
            pdrop_string,
            batch_size,
            opt,
            fine_tuning_names_short[fine_tuning_code])
    else:
        mname = 'SmaAtUNet_bl_{}_{}_drop{}_b{}_{}_{}'.format(
            'cat'+folder_postfix,
            quickmap[maxlen] if maxlen in quickmap else maxlen,
            pdrop_string,
            batch_size,
            opt,
            fine_tuning_names_short[fine_tuning_code])

    # pixnum = 320
    # pixnum = 1600
    # pixnum = 640
    ## pixnum is now an argument (see above)
    
    run_all(in_folders, out_folders, n_class, maxlen, pdrop, weights,
            batch_size, mname, pin_memory, num_workers, crop_outputs=False,
            normalization_scale=None, in_size=pixnum,
            separate_val_folders=separate_val_folders, val_maxlen=10000,
            unet_model=SmaAt_UNet,
            lr_init=lr)
