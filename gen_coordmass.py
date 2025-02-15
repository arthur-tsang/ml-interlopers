################################################################################
# gen_coordmass.py

# Generates a map of the masses of the subhalos.
################################################################################

import numpy as np
import os
import sys
import torch
from data_gen_bright import gen_image_general, safe_mkdir
import argparse
import json

def clean_params(raw_params):
    """Turn params from a dict to an ordered list of numbers"""
    param_names = ['main_theta', 'gamma', 'phi_lens', 'q_lens', 'center_lens_x', 'center_lens_y',
                   'shear1', 'shear2',
                   'lens_a3', 'lens_phi3', 'lens_a4', 'lens_phi4']

    params_true = [raw_params[name] for name in param_names]

    return params_true

def main(startidx, endidx, keyword, num_exposures, concentration, bright_ratio,
         n_subhalo):
    """Generate an 80x80, 320x320, and 640x640 image together"""

    data_dir = '/n/holyscratch01/dvorkin_lab/Users/atsang/mif'
    # num_exposures = 10

    ## The numbers below can be calculated at the beginning of `concentration.ipynb`.
    
    if concentration == 15:
        rs = 1.075e-4 # Mpc ## concentration = 15
        m200_ratio = 0.7782 ## ratio of M200/Mtot at concentration = 15
    elif concentration == 20:
        rs = 8.296e-5
        m200_ratio = 0.8470
    elif concentration == 25:
        rs = 6.476e-5
        m200_ratio = 0.8896
    elif concentration == 30:
        rs = 5.680e-5
        m200_ratio = .9173
    elif concentration == 45:
        rs = 3.843e-5
        m200_ratio = .9589
    elif concentration == 60:
        rs = 2.899e-5 # Mpc ## concentration = 60
        m200_ratio = 0.9758 # ratio of M200/Mtot at concentration c = 60
    else:
        raise ValueError('concentration must either be 15, 20, 25, 30, 45, or 60')

    assert keyword in ['cattrain', 'catval', 'lecatval']
    cat_source = True
    
    postfix1 = keyword + '_exp{}_c{}_br{}_nsub{}_hstelt3'.format(num_exposures,
                                                                 concentration,
                                                                 int(100*bright_ratio),
                                                                 n_subhalo)
    postfix2 = keyword + '_exp{}_c{}_br{}_nsub{}_hstelt'.format(num_exposures,
                                                                concentration,
                                                                int(100*bright_ratio),
                                                                n_subhalo)
    postfix3 = keyword + '_exp{}_c{}_br{}_nsub{}_hst'.format(num_exposures,
                                                             concentration,
                                                             int(100*bright_ratio),
                                                             n_subhalo)
    postfixes = [postfix1, postfix2, postfix3]
    
    pixnums = [640, 320, 80]
    pixsizes = [.01, .02, .08]
    pixrads = [4, 2, 2]
    
    # prefixes = ['jparams', 'in', 'coord']
    prefixes = ['coordmass']
    for prefix in prefixes:
        for postfix in postfixes:
            safe_mkdir(os.path.join(data_dir, f'{prefix}_{postfix}'))

    constraints_all = {'num_exposures':num_exposures,
                        'rs':rs}

    # loop for generating images
    for i in range(startidx, endidx):
        nstr = f'{i}.npy'
        if os.path.exists(os.path.join(data_dir, 'coordmass_'+postfixes[-1], nstr)):
            continue

        np.random.seed(i+2)

        mass_bin = np.random.randint(6)
        upper_mass = 10**(8.5 + mass_bin/2) / m200_ratio
        lower_mass = 10**8. / m200_ratio # only relevant if auto-populating with n_subhalo=-1
        is_pert = np.random.binomial(1, .9) # 90-10 split (in favor of perturbers)

        all_params = gen_image_general(i, True, upper_mass, keyword,
                                       constraints=constraints_all,
                                       mode='all params',
                                       cat_source=True,
                                       pixnum=pixnums[0],
                                       pixsize=pixsizes[0],
                                       bright_ratio=bright_ratio,
                                       n_subhalo=n_subhalo,
                                       lower_mass=lower_mass)
        
        constraints = {**constraints_all,
                       **all_params}
        for bad_key in ['main_theta', 'zl', 'zs', 'pixnum', 'pixsize', 'use_val_cat',
                        'observation_config', 'observation_band', 'observation_psf_type']:
            constraints.pop(bad_key)

        ########################################################################
        ### Now we store params as json file
        json_data = constraints.copy()
        json_data['main_theta'] = all_params['main_theta']
        for key in json_data:
            if isinstance(json_data[key], np.ndarray):
                json_data[key] = list(json_data[key])

        with open(os.path.join(data_dir, 'jparams_'+postfix, nstr), 'w') as f:
            json.dump(json_data, f, indent=4)

        ########################################################################
            
        # Now we're left with constraints that will let us generate the same
        # image but in different resolutions
        coordmassl = []
        for pixnum, pixsize, pixrad in zip(pixnums, pixsizes, pixrads):
            coordmass = gen_image_general(
                i, True, upper_mass, keyword,
                constraints=constraints, mode='coordmass',
                cat_source=cat_source, pixnum=pixnum, pixsize=pixsize,
                pixrad=pixrad,
                bright_ratio=bright_ratio,
                n_subhalo=len(constraints['mass_multiplier']))
            # We give n_subhalo as a length in the line above, in case n_subhalo
            # was -1, meaning the true number of subhalos varies by random
            # draws.

            coordmassl.append(coordmass)

        # Note that the following lines that choose mass bin don't apply well to
        # the case of multiple subhalos, and should not be used for training
        # data. Future work should include reorganizing the "coord" code to
        # account for subhalo bins (which would involve changing
        # `gen_image_general`).
        if not is_pert:
            coordmassl = [np.zeros_like(c) for c in coordmassl]

        for postfix, coordmass in zip(postfixes, coordmassl):
            np.save(os.path.join(data_dir, 'coordmass_'+postfix, nstr),
                    coordmass)

        
            
    print('Saved all from {} to {}'.format(startidx, endidx))
            
    
if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('startidx', type=int)
    parser.add_argument('endidx', type=int)
    parser.add_argument('keyword', type=str,
                        help='cattrain or catval, unless using a higher minimum q (main lens axis ratio)')
    parser.add_argument('--numexposures', type=int, default=10)
    parser.add_argument('--concentration', type=int, default=60)
    parser.add_argument('--brightratio', type=float, default=0.20)
    parser.add_argument('--nsubhalo', type=int, default=1) # let nsubhalo be -1 for "realistic" distribution
    args = parser.parse_args()

    ## One more parameter, the minimum q (main lens axis ratio) will be modified by changing the `keyword`
    
    main(args.startidx, args.endidx, args.keyword,
         args.numexposures, args.concentration, args.brightratio, args.nsubhalo)
