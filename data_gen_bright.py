################################################################################
# data_gen_bright.py
#
# 25 Jan 2021
#
# Like `data_gen.py` but positions only on the bright parts of the image.
# This code is cleaner than the original "legacy" `data_gen.py`.
################################################################################

import numpy as np
# import matplotlib
# if __name__ == '__main__':
#     matplotlib.use('Agg') # only do this to run on the cluster
# import matplotlib.pyplot as plt

import sys
import os

from helpers import make_ctr_mask
from helpers import coord_image_multisub
from helpers import coord_image_multisub_bymass
from helpers import CustomImage, CustomWeak
from helpers import COSMOSImage, COSMOSWeak
from helpers import ADD, sigma_cr
from helpers import positive_poisson
from helpers import total_mass, total_nr, avg_mass
# from helpers import WeakImage
from data_gen_params import pick_params
from data_gen_params import pick_params_cat
# from data_gen import double_cone_angle
# from data_gen import coord_image_without_main_lens
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from telescopes import HSTELT, HSTELT3

### Constants ##################################################################
zl = 0.5 # main lens redshift
zs = 1.0
imgwidth = 6.4
pixnum = 80

pixsize = 1. * imgwidth / pixnum #  = .08
bin_scheme = [-.01, -.003, -.001, -.0003, -.0001, .0001, .0003, .001, .003, .01]    
################################################################################

################################################################################
### Double-cone ###

cosmo = FlatLambdaCDM(H0=67.5, Om0=0.316)

def double_cone_direct(com_z, com_l, com_s):
    # Input can be in any units
    if com_z < com_l:
        return com_z / com_l
    else:
        return (com_s - com_z) / (com_s - com_l)

def double_cone(z, zl=zl, zs=zs):
    com_z = cosmo.comoving_distance(z)
    com_l = cosmo.comoving_distance(zl)
    com_s = cosmo.comoving_distance(zs)
    return double_cone_direct(com_z, com_l, com_s).value

def double_cone_angle(z, zl=zl, zs=zs):
    # angle of how wide the interlopers can be dispersed so that they'd show up in the final image (according to the double-prism projection)
    
    # returns a ratio, 1 for z <= zl
    
    com_z = cosmo.comoving_distance(z)
    com_l = cosmo.comoving_distance(zl)
    return (double_cone(z, zl=zl, zs=zs) * com_l / com_z).value # unitless quantity anyways

################################################################################

def pix_to_pos(xpix, ypix, pixnum, pixsize):
    # go from a pixel index to a position
    imgwidth = pixnum * pixsize

    x_img_frac = (xpix / pixnum) - .5
    y_img_frac = (ypix / pixnum) - .5
    x_pos = x_img_frac * imgwidth
    y_pos = y_img_frac * imgwidth
    return (x_pos, y_pos)

def gen_image_general(seed, sub, mass, param_keyword, theta=2, mode='regular',
                      constraints={}, bright_ratio=0.2, cat_source=False, pixnum=pixnum,
                      pixsize=pixsize, pixrad=2, n_subhalo=1, lower_mass=None):
    """Generate image (or related tasks).
    `mode` specifies what kind of output we want.

    Options for `mode` include:
    - simple : (in, curlbin)
    - noiseless : (in_noiseless, curlbin)
    - macro params : z, `params`
    - all params : z, `params`, x, y
    - blank clean : macro-only image (clean)
    - blank noisy : macro-only image (noisy)
    - customimage objects : the full CustomImage objects of both the full image and the macro-only image (maybe useful for some notebooks)
    - noiseless coordinates : (in_noiseless, coordinates)
    - noisy coordinates : (in_noisy, coordinates)

    `mass` is really an upper mass for a given mass bin, which will then be
    multiplied by a randomly-chosen mass multiplier. These masses are understood
    to be total tNFW masses, not m200 masses.

    `bright_ratio` is the ratio of a given pixel's brightness to the brightest
    pixel's brightness for it to count as a potential perturber location
    
    `n_subhalo` gives the number of subhalos we will include (if we are
    including any subhalos at all). If `n_subhalo` is -1, that means we want the
    function to pick subhalos according to a mass function, conditioning on the
    fact that there must be at least one subhalo with mass within the interval
    [10**-.5 * mass, mass].

    `lower_mass` is the lowest mass of subhalos we consider (only matters if
    n_subhalo == -1 and we pick based on distribution)

    """

    assert isinstance(mass, float) or len(mass) == 1 or len(mass) == n_subhalo
    if n_subhalo < 0:
        assert isinstance(mass, float)

    assert cat_source
        
    # Take care of observation config at the beginning
    if pixnum == 320 and abs(pixsize - .02) < 1e-6:
        observation_config = HSTELT
        observation_band = 'my_filter'
        observation_psf_type = 'GAUSSIAN'
    elif pixnum == 640 and abs(pixsize - .01) < 1e-6:
        observation_config = HSTELT3
        observation_band = 'my_filter'
        observation_psf_type = 'GAUSSIAN'
    elif pixnum == 80 and abs(pixsize - .08) < 1e-6:
        observation_config = HST
        observation_band = 'WFC3_F160W'
        observation_psf_type = 'GAUSSIAN'
    else:
        raise ValueError('Pixnum {} and pixsize {} not compatible with telescope list'.format(pixnum, pixsize))

    
    np.random.seed(seed)

    if 'z' in constraints:
        z = constraints['z']
    elif sub:
        if n_subhalo >= 0:
            z = [zl] * n_subhalo
        else:
            z = None # later will define `z` once a real n_subhalo has been chosen
    else:
        if n_subhalo >= 0:
            z = list(np.random.uniform(.3, .7, size=n_subhalo))
        else:
            z = None
    

    if not cat_source:
        raise NotImplementedError # No longer maintaining this, now that we're
                                  # adding the option to simulate multiple
                                  # subhalos

    else: # (if cat_source)
        params_orig = pick_params_cat(param_keyword, n_subhalo=max(0,n_subhalo))
        lens_params_o, source_params_o, mass_multiplier_o, mag_lens_o, use_val_cat_o = params_orig
        lens_keys = ['theta_lens_multiplier', 'gamma', 'phi_lens', 'q_lens',
                     'center_lens_x', 'center_lens_y', 'shear1', 'shear2',
                     'lens_a3', 'lens_phi3', 'lens_a4', 'lens_phi4']
        source_keys = ['source_phi', 'source_center_x', 'source_center_y']
        other_keys = ['mass_multiplier', 'z', 'x', 'y', 'mass_sheets', 'num_exposures', 'source_cat_seed', 'rs']

        ## Check that there are no typos or unrecognized keys in `constraints`:
        for key in constraints:
            if key in lens_keys or key in source_keys or key in other_keys:
                continue
            else:
                raise KeyError('key {} in `constraints` not recognized'.format(key))

        if n_subhalo < 0:
            # If we don't even know how many subhalos will be in the image, the following don't make sense as constraints.
            assert 'mass_multiplier' not in constraints
            assert 'z' not in constraints
            assert 'x' not in constraints
            assert 'y' not in constraints
            
        ## Define lens kwargs (from either constraint value if exists, otherwise randomly picked lens param)
        lens_kwargs = {}
        for key, param_o in zip(lens_keys, lens_params_o):
            if key == 'theta_lens_multiplier': # (special case, since default theta is defined elsewhere)
                theta_lens_multiplier = constraints[key] if key in constraints else param_o
                lens_kwargs['main_theta'] = theta * theta_lens_multiplier
            else:
                lens_kwargs[key] = constraints[key] if key in constraints else param_o
        
        ## Define source kwargs (again, from constraint if exists else randomly picked param)
        source_kwargs = {}
        for key, param_o in zip(source_keys, source_params_o):
            source_kwargs[key] = constraints[key] if key in constraints else param_o

        ## Define other kwargs
        mass_sheets = constraints['mass_sheets'] if 'mass_sheets' in constraints else True
        rs = constraints['rs'] if 'rs' in constraints else 1e-4
        other_kwargs = {'mass_sheets': mass_sheets,
                        'zl':zl, 'zs':zs, 'pixnum':pixnum, 'pixsize':pixsize, 'rs':rs, 'use_val_cat':use_val_cat_o,
                        'observation_config':observation_config,
                        'observation_band':observation_band,
                        'observation_psf_type':observation_psf_type}

        mass_multiplier = constraints['mass_multiplier'] if 'mass_multiplier' in constraints else mass_multiplier_o ## Doesn't go directly in `other_kwargs`!
        
        assert len(mass_multiplier) == max(0, n_subhalo) or len(mass_multiplier) == 1
        


        source_cat_seed = seed+1 if 'source_cat_seed' not in constraints else constraints['source_cat_seed']

        # generate blank image
        blank_kwargs = {'xpos_list':[], 'ypos_list':[], 'redshift_list':[], 'source_cat_seed':source_cat_seed,
                        **lens_kwargs, **source_kwargs, **other_kwargs}

        blankimg = COSMOSImage(**blank_kwargs)

    num_exposures = constraints['num_exposures'] if 'num_exposures' in constraints else 1
    blankimg.calc_image_noisy(num_exposures=num_exposures)

    if mode=='blank clean':
        return blankimg.image
    elif mode=='blank noisy':
        return blankimg.image_noisy
    elif mode == 'macro alpha':
        blankimg.calc_div_curl()
        return blankimg.alphamat_y.T, blankimg.alphamat_x.T

    # pick interloper location (this will be its own function)
    xraws = []
    yraws = []
    xlist = []
    ylist = []
    
    bright_pixels = blankimg.image > bright_ratio * np.max(blankimg.image)
    bright_aw = np.argwhere(bright_pixels) # list of all bright pixels

    # Now we figure out how many subhalos we actually want (if applicable)
    if n_subhalo < 0:
        sigma_c = sigma_cr(zl, zs)
        area = pixsize**2 * len(bright_aw) * (np.pi/648000)**2 * ADD(0, zl)**2
        macro_mass = (sigma_c * area).to(u.Msun).value # total macro mass, assuming kappa = 1
        fsub = .01

        # calculate number of subhalos in top bin (must be at least 1)
        toprange = (np.log10(mass) - 0.5, np.log10(mass))
        fsub_topbin = fsub * total_mass(*toprange) / total_mass(8, 10)
        lam_topbin = fsub_topbin * macro_mass / avg_mass(*toprange) # expected number in topbin
        nsub_topbin = positive_poisson(lam_topbin)

        # calculate number of subhalos in bottom bins (can be 0)
        botrange = (np.log10(lower_mass), np.log10(mass) - 0.5)
        fsub_botbins = fsub * total_mass(*botrange) / total_mass(8, 10)
        lam_botbins = fsub_botbins * macro_mass / avg_mass(*botrange) # expected number in botbins
        nsub_botbins = np.random.poisson(lam_botbins)

        # And now we're ready to calculate z...
        nsub = nsub_topbin + nsub_botbins
        if sub:
            z = [zl] * nsub
        else:
            z = list(np.random.uniform(.3, .7, size=nsub))

        # ... and to calculate `mass_multiplier`.
        mass_multiplier = []
        for i in range(nsub_topbin):
            # sample masses using the inverse CDF
            rand = np.random.rand()
            mymass = ((10**(-0.9*toprange[1]) - 10**(-0.9*toprange[0]))*rand+10**(-0.9*toprange[0]))**(-1/0.9)
            mass_multiplier.append( mymass / mass )

        for i in range(nsub_botbins):
            # same idea here: sample masses using the inverse CDF
            rand = np.random.rand()
            mymass = ((10**(-0.9*botrange[1]) - 10**(-0.9*botrange[0]))*rand+10**(-0.9*botrange[0]))**(-1/0.9)
            mass_multiplier.append( mymass / mass )

        # Now that we're done with that, we can forget we did this whole process and set `n_subhalo` to match what we chose.
        n_subhalo = nsub_topbin + nsub_botbins
            
    ############################################################################
    # Now we can calculate `macro_params`
    macro_params = {'z':z, 'mass_multiplier': mass_multiplier,
                    **lens_kwargs, **source_kwargs, **other_kwargs}
    if mode == 'macro params':
        return macro_params
    ############################################################################
    
    for i in range(n_subhalo):
        xpix_raw, ypix_raw = bright_aw[np.random.randint(len(bright_aw))]
        x_raw_orig, y_raw_orig = pix_to_pos(xpix_raw, ypix_raw, pixnum, pixsize)

        x_raw = constraints['x'][i] if 'x' in constraints else x_raw_orig
        y_raw = constraints['y'][i] if 'y' in constraints else y_raw_orig
    
        # calculate interloper x, y from raw coordinates
        x = x_raw * double_cone_angle(z[i])
        y = y_raw * double_cone_angle(z[i])
        # (note that this is a little imprecise since it assumes the lens is centered at 0)

        xraws.append(x_raw)
        yraws.append(y_raw)
        xlist.append(x)
        ylist.append(y)

    if not cat_source:
        raise NotImplementedError
    else:
        if mode=='all params':
            return {'z':z, **lens_kwargs, **source_kwargs, **other_kwargs, 'x':xlist, 'y':ylist, 'mass_multiplier': mass_multiplier}

        myimg_kwargs = {'xpos_list':ylist, 'ypos_list':xlist, 'redshift_list':z,
                        'm':np.array(mass)*np.array(mass_multiplier),
                        'source_cat_seed':source_cat_seed,
                        **lens_kwargs, **source_kwargs, **other_kwargs}

        # generate perturbed image
        myimg = COSMOSImage(**myimg_kwargs)


    if mode == 'imageModel':
        myimg.calc_image_imageModel()
        return myimg.imageModel_image

    myimg.calc_image_noisy(num_exposures=num_exposures)

    # generate div and curl maps
    myimg.calc_div_curl_5pt()
    blankimg.calc_div_curl_5pt()
    
    curlbinmat = np.digitize(myimg.curlmat - blankimg.curlmat, bin_scheme, right=True)    

    def get_source():
        if not cat_source:
            weakimg = CustomWeak([], [], [], zs=zs,
                                 pixnum=pixnum, pixsize=pixsize,
                                 mass_sheets=mass_sheets,
                                 **source_kwargs) ## TODO test that this works

            
        else:
            pixfactor = 5.

            # print('pixnum', pixnum * pixfactor)
            weakimg = COSMOSWeak([], [], [], zs=zs,
                                 pixnum=int(pixnum * pixfactor), pixsize=pixsize / pixfactor,
                                 mass_sheets=mass_sheets,
                                 source_cat_seed=source_cat_seed,
                                 **source_kwargs,
                                 rs=rs,
                                 observation_config=observation_config,
                                 observation_band=observation_band,
                                 observation_psf_type=observation_psf_type
            )
            ## Don't want to plug in **other_kwargs directly, because don't want
            ## to plug in `zl`. We take care of the rest of the `other_kwargs`
            ## manually here.

        weakimg.calc_image_noisy(num_exposures=num_exposures)
        return weakimg.image # no noise!!

    if mode == 'regular':
        return (myimg.image_noisy,
                blankimg.image_noisy,
                coord_image_multisub(xraws, yraws, pixnum, pixsize, pixrad=pixrad),
                macro_params)
    elif mode == 'regular2':
        # like `regular` but includes noiseless versions
        return (myimg.image_noisy, myimg.image,
                blankimg.image_noisy, blankimg.image,
                coord_image_multisub(xraws, yraws, pixnum, pixsize, pixrad=pixrad),
                macro_params)
    elif mode == 'dynesty':
        source = get_source()
        return (myimg.image_noisy,
                myimg.image,
                myimg.estimated_noise,
                blankimg.image_noisy,
                blankimg.image,
                blankimg.estimated_noise,
                source)
    elif mode == 'coordmass':

        return coord_image_multisub_bymass(xraws, yraws,
                                           np.array(mass)*np.array(mass_multiplier),
                                           pixnum, pixsize, pixrad=pixrad)
        
    elif mode == 'noisy div curl':
        # using this for the multiple subhalos test
        return (myimg.image_noisy, blankimg.divmat.T, blankimg.curlmat.T)
    elif mode == 'source estnoise':
        source = get_source()
        return (source, myimg.estimated_noise)
    elif mode == 'simple':
        return (myimg.image_noisy, curlbinmat.T)
    elif mode == 'noiseless':
        return (myimg.image, curlbinmat.T)
    elif mode == 'customimage objects':
        return (myimg, blankimg)
    elif mode == 'noiseless coordinates':
        return (myimg.image, coord_image_multisub(xraws, yraws, pixnum, pixsize, pixrad=pixrad))
    elif mode == 'noisy coordinates':
        return (myimg.image_noisy, coord_image_multisub(xraws, yraws, pixnum, pixsize, pixrad=pixrad))
    elif mode == 'noisy noiseless coordinates':
        return (myimg.image_noisy, myimg.image, coord_image_multisub(xraws, yraws, pixnum, pixsize, pixrad=pixrad))
    elif mode == 'source alpha image noiseless':
        source = get_source()
        myimg.calc_div_curl() # just to calculate alphamat (doesn't have to be 5pt) # I think we did this above anyways so it's unnecessary
        return (source, myimg.alphamat_y.T, myimg.alphamat_x.T, myimg.image)
    elif mode == 'source alpha noisy noiseless coordinates':
        source = get_source()
        myimg.calc_div_curl() # just to calculate alphamat (doesn't have to be 5pt)
        return (source, myimg.alphamat_y.T, myimg.alphamat_x.T, myimg.image_noisy,
                myimg.image, coord_image_multisub(xraws, yraws, pixnum, pixsize, pixrad=pixrad))
    elif mode == 'noisy noiseless malpha coord':
        blankimg.calc_div_curl()
        return (myimg.image_noisy, myimg.image, blankimg.alphamat_y.T,
                blankimg.alphamat_x.T, coord_image_multisub(xraws, yraws, pixnum,
                                                   pixsize, pixrad=pixrad))
    elif mode == 'noisy noiseless malpha coord source':
        source = get_source()
        blankimg.calc_div_curl()
        return (myimg.image_noisy, myimg.image, blankimg.alphamat_y.T,
                blankimg.alphamat_x.T,
                coord_image_multisub(xraws, yraws, pixnum, pixsize, pixrad=pixrad),
                source)
    elif mode == 'noisy blank noisy malpha coord':
        blankimg.calc_div_curl()
        return (myimg.image_noisy, blankimg.image_noisy, blankimg.alphamat_y.T,
                blankimg.alphamat_x.T, coord_image_multisub(xraws, yraws, pixnum, pixsize, pixrad=pixrad))
    elif mode == 'noisy alpha malpha coord params':
        return (myimg.image_noisy, myimg.alphamat_y.T, myimg.alphamat_x.T,
                blankimg.alphamat_y.T, blankimg.alphamat_x.T,
                coord_image_multisub(xraws, yraws, pixnum, pixsize, pixrad=pixrad),
                macro_params)
    else:
        raise ValueError('Mode for gen_image_general "{}" not recognized'.format(mode))

def gen_image_simple(seed, sub, mass, param_keyword, theta=2):
    """Generate only the image and the curlbin map"""
    return gen_image_general(seed, sub, mass, param_keyword, theta=theta, mode='simple')

def recalc_params(seed, sub, mass, param_keyword, theta=2, macro_only=True):
    return gen_image_general(seed, sub, mass, param_keyword, theta=theta, mode='macro params' if macro_only else 'all params')

def safe_mkdir(mydir, verbose=True):
    try:
        os.mkdir(mydir)
        if verbose:
            print('created directory {}'.format(mydir))
    except FileExistsError:
        if verbose:
            print('folder {} already exists. Will proceed'.format(mydir))
