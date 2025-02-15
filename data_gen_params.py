### How to pick source and lens parameters for the data generation ###

# We'll stick this in another file so it doesn't clutter `data_gen.py`.

import numpy as np
from lenstronomy.Util.param_util import ellipticity2phi_q

def pick_params(keyword):
    if keyword == 'simpler':
        # theta_lens_multiplier = np.random.uniform(.4, .6)
        theta_lens_multiplier = 0.5
        # gamma = np.random.uniform(1.5, 2.5)
        gamma = 2
        # phi_lens = np.random.uniform(-np.pi/2., np.pi/2.)
        # q_lens = np.random.uniform(.5, 1)
        e1_lens = np.random.uniform(0, .1)
        e2_lens = np.random.uniform(0, .1)
        phi_lens, q_lens = ellipticity2phi_q(e1_lens, e2_lens)
        
        center_lens_x = np.random.uniform(-.2, .2)
        center_lens_y = np.random.uniform(-.2, .2)
        shear1 = 0
        shear2 = 0
        lens_a3 = 0
        lens_phi3 = 0
        lens_a4 = 0
        lens_phi4 = 0

        lens_params = (theta_lens_multiplier, gamma, phi_lens, q_lens, center_lens_x, center_lens_y, shear1, shear2, lens_a3, lens_phi3, lens_a4, lens_phi4)

        N_sersic = np.random.randint(1, 5) # high is exclusive,  so it's actually U[1, 4]
        N_clump = 0
        
        ## Set source params.
        source_params = []
        for i in range(N_sersic):
            r_sersic_source = np.random.uniform(.1, 1)
            ## We want to select e1 and e2 to be within a circle of radius 0.4.
            phi_source = np.random.uniform(-np.pi, np.pi)
            # The sqrt function transforms a flat pdf into a triangle pdf
            # (which is what we need for uniform area within a circle)
            eps = 0.4 * np.sqrt(np.random.uniform(0,1))
            # relationship between ellipticity and minor/major axis ratio:
            q_source = (1-eps)/(1+eps)
            
            beta_ra_source = np.random.uniform(-.2, .2)
            beta_dec_source = np.random.uniform(-.2, .2)
            # n_sersic_source = np.random.uniform(.5, 1)
            n_sersic_source = 1

            source_params.append((r_sersic_source, phi_source, q_source, beta_ra_source,
                         beta_dec_source, n_sersic_source))

        mag_source = 20

        ## Set source clump params.
        r_sersic_source_clumps = .1
        n_sersic_source_clumps = .5
        source_scatter = 0.75 * r_sersic_source
        mag_source_clumps = 22
        clump_params = (N_clump, r_sersic_source_clumps, n_sersic_source_clumps,
                        source_scatter, mag_source_clumps)

        
        ## Set perturber parameters.
        mass_multiplier = 10**np.random.uniform(-.5, 0)

        ## Set lens light (non-) parameters.
        mag_lens = None

        return (lens_params, source_params, mag_source, clump_params, mass_multiplier, mag_lens)
    
    else:
        raise ValueError(f'`keyword` {keyword} not recognized')
    

def pick_params_cat(keyword, n_subhalo=1):
    ## Old methods of generating catalogs:
    if keyword == 'narrowcat':
        # raise NotImplementedError # not implemented anymore ;)
        return narrowcatsrc(10**-.5)
    elif keyword == 'catshear':
        return catshear(10**-.5)

    ## The following two are the current ones.
    elif keyword == 'cattrain':
        # catalog galaxies, shear, multipole
        return cat_multipoles(False, n_subhalo=n_subhalo)
    elif keyword == 'catval':
        return cat_multipoles(True, n_subhalo=n_subhalo)

    elif keyword == 'lecatval':
        # low ellipticity catval
        return cat_multipoles(True, n_subhalo=n_subhalo, min_q_lens=0.8)
        
    else:
        raise Exception('keyword {} not recognized (assuming source from catalog)'.format(keyword))

def cat_multipoles(use_val_cat, n_subhalo=1, min_q_lens=0.5):
    ## Set lens params.
    theta_lens_multiplier = np.random.uniform(.4, .6)
    gamma = np.random.uniform(1.5, 2.5)
    phi_lens = np.random.uniform(-np.pi/2., np.pi/2.)
    q_lens = np.random.uniform(min_q_lens, 1)
    center_lens_x = np.random.uniform(-.2, .2)
    center_lens_y = np.random.uniform(-.2, .2)
    shear1 = np.random.uniform(-.1, .1)
    shear2 = np.random.uniform(-.1, .1)
    lens_a3 = np.random.uniform(-.02, .02)
    lens_phi3 = np.random.uniform(-np.pi/3., np.pi/3.)
    lens_a4 = np.random.uniform(-.02, .02)
    lens_phi4 = np.random.uniform(-np.pi/4., np.pi/4.)
    
    lens_params = (theta_lens_multiplier, gamma, phi_lens, q_lens, center_lens_x, center_lens_y, shear1, shear2, lens_a3, lens_phi3, lens_a4, lens_phi4)

    ## Set source params.
    source_center_x = np.random.uniform(-.2, .2)
    source_center_y = np.random.uniform(-.2, .2)
    source_phi = np.random.uniform(-np.pi, np.pi)
    source_params = (source_phi, source_center_x, source_center_y)

    ## Set perturber parameters.
    # old version: mass_multiplier = 10**np.random.uniform(-.5, 0)
    mass_multiplier = 10**np.random.uniform(-.5, 0, size=n_subhalo)

    ## Set lens light (non-) parameters.
    mag_lens = None

    return (lens_params, source_params, mass_multiplier, mag_lens, use_val_cat)


def catshear(low_mass_multiplier):
    ## Set lens params.
    theta_lens_multiplier = np.random.uniform(.4, .6)
    gamma = np.random.uniform(1.5, 2.5)
    phi_lens = np.random.uniform(-np.pi, np.pi)
    q_lens = np.random.uniform(.5, 1)
    center_lens_x = np.random.uniform(-.2, .2)
    center_lens_y = np.random.uniform(-.2, .2)
    shear1 = np.random.uniform(-.1, .1)
    shear2 = np.random.uniform(-.1, .1)
    lens_params = (theta_lens_multiplier, gamma, phi_lens, q_lens, center_lens_x, center_lens_y, shear1, shear2)

    ## Set source params.
    source_center_x = np.random.uniform(-.2, .2)
    source_center_y = np.random.uniform(-.2, .2)
    source_phi = np.random.uniform(-np.pi, np.pi)
    source_params = (source_phi, source_center_x, source_center_y)

    ## Set perturber parameters.
    mass_multiplier = np.random.uniform(low_mass_multiplier, 1) # so we vary from 1e9 to 1e10

    ## Set lens light (non-) parameters.
    mag_lens = None

    return (lens_params, source_params, mass_multiplier, mag_lens)


def narrowcatsrc(low_mass_multiplier):
    ## Set lens params.
    theta_lens_multiplier = np.random.uniform(.4, .6)
    gamma = np.random.uniform(1.5, 2.5)
    phi_lens = np.random.uniform(-np.pi, np.pi)
    q_lens = np.random.uniform(.5, 1)
    center_lens_x = np.random.uniform(-.2, .2)
    center_lens_y = np.random.uniform(-.2, .2)
    shear1 = 0
    shear2 = 0
    lens_params = (theta_lens_multiplier, gamma, phi_lens, q_lens, center_lens_x, center_lens_y,
                   shear1, shear2)

    ## Set source params.
    source_center_x = np.random.uniform(-.2, .2)
    source_center_y = np.random.uniform(-.2, .2)
    source_phi = np.random.uniform(-np.pi, np.pi)
    source_params = (source_phi, source_center_x, source_center_y)

    ## Set perturber parameters.
    mass_multiplier = np.random.uniform(low_mass_multiplier, 1) # so we vary from 1e9 to 1e10

    ## Set lens light (non-) parameters.
    mag_lens = None

    return (lens_params, source_params, mass_multiplier, mag_lens)

## See bups/bup_data_gen_params.py for old version
