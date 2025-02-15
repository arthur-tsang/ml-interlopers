################################################################################
# gen_fordynesty.py
#
# Generates a set of images for dynesty, and saves macro parameters
################################################################################


import os
import sys
import argparse
# import pickle
import json

import torch
import numpy as np
from data_gen_bright import gen_image_general, safe_mkdir


def generate_set(seed, folder='dynestyproj/fordynesty/'):
    """
    Creates a set of 5 images:

    at c = 15, 80 mas
    at c = 60, 80 mas
    at c = 60, 20 mas

    no subhalo, 80 mas
    no subhalo, 20 mas

    and saves the macro parameters as well.

    We save everything in the folder `./fordynesty/`

    """

    concentration_map = {15: (1.075e-4, 0.7782), 60: (2.899e-5, 0.9758)}
    # in the format (rs, m200_ratio)

    keyword = 'catval' # catalog-based sources, validation set
    bright_ratio = .2
    
    ## First we generate all the params
    placeholder_mass = 10**10. # just needs to be a float
    all_params = gen_image_general(seed, True, placeholder_mass, keyword,
                                   constraints={'mass_multiplier':[1]},
                                   mode='all params',
                                   cat_source=True,
                                   pixnum=640,
                                   pixsize=0.01,
                                   bright_ratio=bright_ratio,
                                   n_subhalo=1)
    constraints = all_params.copy()
    for bad_key in ['main_theta', 'zl', 'zs', 'pixnum', 'pixsize', 'use_val_cat',
                    'observation_config', 'observation_band', 'observation_psf_type', 'rs']:
        constraints.pop(bad_key)

    # with open(os.path.join(folder, f'{seed:03}_params.p'), 'wb') as f:
    #     pickle.dump(constraints, f)

    ############################################################################
    # Now we store the params as a json file.

    ## Make sure the folder exists
    safe_mkdir(folder)
    
    json_data = constraints.copy()
    json_data['main_theta'] = all_params['main_theta']
    for key in json_data:
        if isinstance(json_data[key], np.ndarray):
            json_data[key] = list(json_data[key])

    with open(os.path.join(folder, f'{seed:03}_params.json'), 'w') as f:
        json.dump(json_data, f, indent=4)
    ############################################################################
        
        
    ## Now we generate each member of the quadruplet, following these params
    def generate_one(conc, pixnum):
        """Note: `conc` must be an integer, as it's used as a dictionary key."""
        
        rs, m200_ratio = concentration_map[conc]
        upper_mass = 10**9.5 / m200_ratio
        pixsize = 6.4/pixnum
        
        output = gen_image_general(
            seed, True, upper_mass, keyword,
            constraints={'num_exposures':10, 'rs':rs,
                         **constraints},
            mode='dynesty',
            cat_source=True,
            pixnum=pixnum,
            pixsize=pixsize,
            pixrad=2, # in number of pixels
            bright_ratio=bright_ratio,
            n_subhalo=1)

        """
        output is in the order:
        sub noisy, sub no noise, sub estimated noise,
        nosub noisy, nosub no noise, nosub estimated noise,
        source
        """

        assert len(output) == 7
        
        return output
    
    r80_c15, r80_c15_x, r80_c15_e, r80_none, r80_none_x, r80_none_e, _  = generate_one(15, 80)
    r80_c60, r80_c60_x, r80_c60_e, _, _, _, _ = generate_one(60, 80)
    r20_c60, r20_c60_x, r20_c60_e, r20_none, r20_none_x, r20_none_e, src = generate_one(60, 320)

    # Save the noisy images
    np.save(os.path.join(folder, f'{seed:03}_r80_c15.npy'), r80_c15)
    np.save(os.path.join(folder, f'{seed:03}_r80_none.npy'), r80_none)
    np.save(os.path.join(folder, f'{seed:03}_r80_c60.npy'), r80_c60)
    np.save(os.path.join(folder, f'{seed:03}_r20_c60.npy'), r20_c60)
    np.save(os.path.join(folder, f'{seed:03}_r20_none.npy'), r20_none)

    # Save the noiseless images
    np.save(os.path.join(folder, f'{seed:03}_r80_c15_x.npy'), r80_c15_x)
    np.save(os.path.join(folder, f'{seed:03}_r80_none_x.npy'), r80_none_x)
    np.save(os.path.join(folder, f'{seed:03}_r80_c60_x.npy'), r80_c60_x)
    np.save(os.path.join(folder, f'{seed:03}_r20_c60_x.npy'), r20_c60_x)
    np.save(os.path.join(folder, f'{seed:03}_r20_none_x.npy'), r20_none_x)

    # Save the noise estimates
    np.save(os.path.join(folder, f'{seed:03}_r80_c15_e.npy'), r80_c15_e)
    np.save(os.path.join(folder, f'{seed:03}_r80_none_e.npy'), r80_none_e)
    np.save(os.path.join(folder, f'{seed:03}_r80_c60_e.npy'), r80_c60_e)
    np.save(os.path.join(folder, f'{seed:03}_r20_c60_e.npy'), r20_c60_e)
    np.save(os.path.join(folder, f'{seed:03}_r20_none_e.npy'), r20_none_e)

    # Save the source
    np.save(os.path.join(folder, f'{seed:03}_src.npy'), src)



if __name__ == '__main__':
    for seed in range(0, 5):
        generate_set(seed, folder='dynestyproj/fordynesty_lownoise')
