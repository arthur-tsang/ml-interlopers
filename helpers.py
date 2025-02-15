"""
helpers.py
----------
Functionality:
`DefaultImage` : easy way to generate a truth-level image.
"""



# some standard python imports #
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cms
#%matplotlib inline
import lenstronomy
from lenstronomy.LensModel.lens_model import LensModel
# from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
# import lenstronomy.Plots.output_plots as lens_plot
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.param_util as param_util
# from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.psf import PSF
# import lenstronomy.Util.image_util as image_util
# from lenstronomy.Workflow.fitting_sequence import FittingSequence

## noise-related imports
# from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.SimulationAPI.sim_api import SimAPI
# from telescopes import HSTELT, SEVENTYNINE, HSTELT2, HSTELT3
from telescopes import HSTELT
# OBSERVATION_CONFIG = HST
# OBSERVATION_BAND = 'WFC3_F160W'
# OBSERVATION_PSF_TYPE = 'GAUSSIAN'
# OBSERVATION_CONFIG = SEVENTYNINE
# OBSERVATION_BAND = 'my_filter'
# OBSERVATION_PSF_TYPE = 'GAUSSIAN'
# OBSERVATION_CONFIG = HSTELT
# OBSERVATION_BAND = 'my_filter'
# OBSERVATION_PSF_TYPE = 'GAUSSIAN'
# OBSERVATION_CONFIG = HSTELT3
# OBSERVATION_BAND = 'my_filter'
# OBSERVATION_PSF_TYPE = 'GAUSSIAN'



## more imports

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from multiprocessing import Pool, cpu_count

from paltas.Sources.cosmos import COSMOSCatalog, COSMOSExcludeCatalog, COSMOSIncludeCatalog
import pandas as pd

from diff_shapelets import DiffShapelets, MultiShapelets

cosmo = FlatLambdaCDM(H0=67.5, Om0=0.316)

val_galaxies_global = pd.read_csv('paltas/Sources/val_galaxies.csv', names=['catalog_i'])['catalog_i'].to_numpy()
bad_galaxies_global = pd.read_csv('paltas/Sources/bad_galaxies.csv', names=['catalog_i'])['catalog_i'].to_numpy()

def ADD(z1,z2):
    ## This is a function that computes the angular diameter distance
    ## between two redshifts z1 and z2.
    return cosmo.angular_diameter_distance_z1z2(z1,z2)

def comdist(z):
    # Function computes the comoving distance to the given redshift
    return cosmo.comoving_distance(z)

def sigma_cr(zd,zs):
    ## This function calculates the critical surface mass density at
    ## redshift zd, relative to the source redshift zs.
    const = 1.663e18*u.M_sun / u.Mpc##c^2/(4 pi G)
    return const*(ADD(0,zs)/(ADD(zd,zs)*ADD(0,zd))) ##in units Msun/Mpc^2

def gfunc(c):
    ## This is the g(c) function that is defined
    ## commonly in NFW profiles.
    a = np.log(1.+c) - (c/(1.+c))
    return 1./a

def rs_angle(zd,rs): 
    ##takes in interloper redshift, gives you the scale redius in angular units
    Dd = ADD(0,zd)
    rs_mpc = rs*u.Mpc
    return ((1./4.848e-6)*rs_mpc)/Dd ##gives in arcsec

def alpha_s(m,rs,zd,zs):
    ##takes in subhalo mass, scale radius, interloper redshift, source redshift
    ##returns the angular deflection at scale radius
    m_msun = m*u.M_sun
    rs_mpc = rs*u.Mpc
    con = (1./np.pi)*gfunc(200.)*(1.-np.log(2))
    return (con*m_msun)/((rs_mpc**2.)*sigma_cr(zd,zs))

def alpha_s_nfw(m,rs,zd,zs):
    """Calculates deflection angle at scale radius where rs is in Mpc and m is the
    M200 mass. This is for untruncated M200"""
    rho_cr = cosmo.critical_density(zd).to(u.M_sun / u.Mpc**3)
    rs_unit = rs * u.Mpc
    # print('rs', rs_unit)
    conc = (1/rs_unit * (3 * m * u.M_sun / (800 * np.pi * rho_cr))**(1/3.)).si.value
    ans_rad = 16*rs_unit**2 * rho_cr * (1-np.log(2)) / (sigma_cr(zd,zs) * ADD(0, zd)) * 200 * conc**3 / 12 * gfunc(conc)
    return ans_rad * 648000/np.pi # deflection angle in arcsec
    
    

def alpha_s_tnfw(m,rs,zd,zs,tau):
    m_phys = m*u.M_sun # physical mass
    mass_factor = tau**2 / (tau**2+1)**2 * ((tau**2-1)*np.log(tau) + tau*np.pi - (tau**2+1))
    # print('not correct, but ignoring mass factor (closer to correct measured convergence)')
    # mass_factor = 1

    m_nfw = m_phys / mass_factor
    # the following code is supposed to match how lenstronomy works, but not necessarily our old `alpha_s`
    distl = ADD(0,zd).to(u.kpc)

    rs_phys = (rs * u.Mpc).to(u.kpc)
    rs_ang = (rs_phys / distl) * 648000/np.pi # arcsec

    rho0 = m_nfw / (2*rs_ang*rs_phys**2 * 2*np.pi) # this formula came from comparing formula 32 in Ana's paper with the function `density_2d` in lenstronomy
    
    alpha_rs = rho0 * 4 * rs_ang**2 * (1 - np.log(2)) # this formula came from Simon's code, where rs was in arcsec
    
    return (alpha_rs / sigma_cr(zd,zs)).si
    
def k_ext(N,m,A,zd,zs,pixsize):
    ## FOR NOW THIS IS SET TO ZERO BECAUSE I CAN'T GET IT TO WORK
    m_msun = m*u.M_sun
    A_mpc2 = (pixsize**4)*(ADD(0.,zd)**2.)*A*((4.848e-6)**2.)  ##arcsec conversion
    return 0.##-(N*m_msun)/(A_mpc2*sigma_cr(zd,zs))


def xi_to_x(xi,z):
    ##takes in physical coordinates (Mpc), turns it into angular coordinates (arcsec)
    x = np.array(((xi*u.Mpc)/ADD(0.,z))/4.848e-6)
    y = x.astype(np.float)
    return y

def x_to_xi(x,z):
    ##takes in angular coordinates (arcsec), turns it into physical coordinates (Mpc)
    return ((x*4.848e-6)*ADD(0,z))/u.Mpc

def xi_to_pix(xi,z,pixsize,pixnum):
    ## takes in physical coordinates (Mpc), identifies the appropriate pixel number
    return (xi_to_x(xi,z))/pixsize + pixnum/2.
def inv_rs_angle(zd,rs_angle):
    ## takes in the rs angle in arcsec gives rs in in MPC
    Dd = ADD(0,zd)
    return 4.848e-6*Dd*rs_angle


def inv_alpha_s(alpha_s,rs,zd,zs):
    ## takes in subhalo angular deflection at scale radius, scale radius,
    ## interloper redshift and source redshift and returns interloper mass
    rs_mpc = rs*u.Mpc
    con = (1./np.pi)*gfunc(200.)*(1.-np.log(2))
    return (alpha_s/con)*((rs_mpc**2.)*sigma_cr(zd,zs))

def autoshow(image, vmax=None, ext=None):
    # helper function like imshow but gets the colors centered at zero

    if vmax == None:
        vmin = np.min(image)
        vmax = np.max(image)
        vmin = min(vmin, -vmax)
        vmax = max(vmax, -vmin)
    else:
        vmin = -vmax

    extent = None if ext==None else [-ext,ext,ext,-ext]
    plt.imshow(image, vmin=vmin, vmax=vmax, cmap='seismic', extent=extent)
    plt.colorbar()

def measure_mass(convmat, zl, zs, ext):
    # measure mass from convergence matrix
    # ext is half the width of the image in arcsec
    phys_width = 2*ext * np.pi/648000 * ADD(0, zl).to(u.kpc)
    #print('phys width', phys_width)
    pixnum = len(convmat) # might be off by -4 depending on method but whatever
    pixsize_phys = phys_width / pixnum
    twod_integral_conv = np.sum(convmat) * pixsize_phys**2
    return twod_integral_conv * sigma_cr(zl, zs).to(u.Msun/u.kpc**2)    



def chi_to_z(chi, zmax=3.1):
    return z_at_value(cosmo.comoving_distance, chi*u.kpc, zmax=zmax)
def z_to_chi(z):
    return cosmo.comoving_distance(z).to(u.kpc).value

class CustomImage:

    def x_to_pix(self, x, z=None):
        # from x to pixel on the lens plane
        if z == None:
            z = self.zl
        return xi_to_pix(x_to_xi(x, z), self.zl, self.pixsize, self.pixnum)

    def double_cone_width(self, z):
        # Return comoving width in kpc

        # First, we calculate the angular extent of this image. Using the
        # comoving distance, we then calculate the comoving width at the widest
        # point of the double-cone (widest in comoving distance, at least).
        view_angle = self.pixsize*self.pixnum * np.pi/648000 # in radians
        lens_width_com = view_angle * comdist(self.zl) # flat-space trick

        com_z = comdist(z)
        com_l = comdist(self.zl)
        com_s = comdist(self.zs)
        
        if z < self.zl:
            width = (com_z / com_l) * lens_width_com
        else:
            width = (com_s - com_z)/(com_s - com_l) * lens_width_com

        return width.to(u.kpc).value
    
    def __init__(self, xpos_list, ypos_list, redshift_list, m=None, zl=0.2,
                 zs=1.0, pixsize=0.2, pixnum=200, mass_sheets=None, main_theta=0.3,
                 q_lens=1, phi_lens=-0.9,
                 center_lens_x=0, center_lens_y=0,
                 shear1=0, shear2=0,
                 lens_a3=0, lens_phi3=0, # 3rd and 4th multipoles
                 lens_a4=0, lens_phi4=0,
                 N_clump=0, clump_seed=0,
                 is_lens_light=False,
                 n_sersic_lens_light=2,
                 # The following source parameters are:
                 # r_sersic, phi, q, beta_ra, beta_dec, n_sersic
                 source_params = [(.5, .5, .3, .01, .02, 1.5)],
                 r_sersic_source_clumps=0.1,
                 n_sersic_source_clumps=1.5,
                 source_scatter=1,
                 gamma=2,
                 mag_source=20,
                 mag_source_clumps=22,
                 mag_lens=20,
                 rs=1e-4,
                 observation_config=HSTELT,
                 observation_band='my_filter',
                 observation_psf_type='GAUSSIAN'):
        # change: used to take in `m` as a single mass for all interlopers, but
        # now this can also be a list of masses
        
        # `mass_sheets` : default set to True, meaning we should add negative mass sheets to cancel out all the substructure

        assert(len(xpos_list) == len(ypos_list))
        assert(len(xpos_list) == len(redshift_list))

        self.xpos_list = xpos_list
        self.ypos_list = ypos_list
        self.redshift_list = redshift_list
        self.N = len(xpos_list) # number of interlopers+subhalos
        N = self.N
        self.zl = zl
        self.zs = zs
        if m is None:
            self.mass_list = [1e7] * self.N
        elif isinstance(m, float):
            self.mass_list = [m] * self.N
        else:
            self.mass_list = m

        self.pixsize = pixsize
        self.pixnum = pixnum

        if isinstance(mass_sheets, list) or isinstance(mass_sheets, np.ndarray):
            self.mass_sheets = mass_sheets
        elif mass_sheets is None or mass_sheets is True:
            self.mass_sheets = [True for _ in range(N)]
        elif mass_sheets is False:
            self.mass_sheets = [False for _ in range(N)]
        any_mass_sheets = np.any(self.mass_sheets)
            
        self.main_theta = main_theta
        self.q_lens = q_lens
        self.phi_lens = phi_lens
        self.center_lens_x = center_lens_x
        self.center_lens_y = center_lens_y
        self.shear1 = shear1
        self.shear2 = shear2
        self.lens_a3 = lens_a3
        self.lens_phi3 = lens_phi3
        self.lens_a4 = lens_a4
        self.lens_phi4 = lens_phi4
        
        self.N_clump = N_clump
        self.clump_seed = clump_seed

        self.n_sersic_lens_light = n_sersic_lens_light
        self.is_lens_light = is_lens_light

        self.source_params = source_params
        
        if isinstance(r_sersic_source_clumps, list) or isinstance(r_sersic_source_clumps, np.ndarray):
            self.r_sersic_source_clumps = r_sersic_source_clumps
        else:
            # (repeat the same value for every clump)
            self.r_sersic_source_clumps = [r_sersic_source_clumps] * self.N_clump
        if isinstance(n_sersic_source_clumps, list) or isinstance(n_sersic_source_clumps, np.ndarray):
            self.n_sersic_source_clumps = n_sersic_source_clumps
        else:
            self.n_sersic_source_clumps = [n_sersic_source_clumps] * self.N_clump
        self.source_scatter = source_scatter
        
        self.gamma = gamma # lens spep gamma

        if isinstance(mag_source, list) or isinstance(mag_source, np.ndarray):
            self.mag_source = mag_source
            assert(len(mag_source) == len(source_params))
        else:
            self.mag_source = [mag_source] * len(source_params)
        self.mag_source_clumps = mag_source_clumps
        self.mag_lens = mag_lens

        self.rs = rs # scale radius of subhalo in Mpc (at 1e6 Msun pivot mass)

        self.observation_config = observation_config
        self.observation_band = observation_band
        self.observation_psf_type = psf_type
        
        ## SOURCE-CLUMP PROPERTIES #########################################################################

        # note: 1210.4562 was using Gaussians rather than Sersic

        np.random.seed(self.clump_seed)

        #r_sersic_source_clumps = 0.1
        clumprandx = np.random.rand(self.N_clump)
        clumprandy = np.random.rand(self.N_clump)

        ####################################################################################################



        ## LENS PROPERTIES #################################################################################
        theta_lens = self.main_theta # used to be 10.
        r_theta_lens = x_to_xi(theta_lens,zl)
        e1, e2 = param_util.phi_q2_ellipticity(phi=self.phi_lens, q=self.q_lens) # used to be q=0.8
        # gamma = 2.

        # lens light properties
        # n_sersic_lens_light = 2.
        ####################################################################################################


        ## INTERLOPER PROPERTIES ########################################################################### 

        # for easier plotting only (current version only works when all the interlopers are at the lens redshift):
        # self.plot_xpixs = [self.x_to_pix(xpos) for xpos in xpos_list]
        # self.plot_ypixs = [self.y_to_pix(ypos), zl,pixsize,pixnum) for ypos in ypos_list]

        #beta_ra, beta_dec = beta_ras[0], beta_decs[0]

        # self.m = 1.0e7 # mass of interlopers (used to be 1e7, and then 1e9)
        # self.rs = 0.001  # interloper scale radius r_s
        # A = 80**2 ## in arcsec ## IGNORE THIS, THIS WAS FOR NEGATIVE CONVERGENCE
        # self.rs = 1e-4 # Mpc (pivot around m0=1e6)
        
        # kext = float(k_ext(N,m,A,zl,zs,pixsize))
        # note that there is no more self.rsang or self.alphars

        ## LENS model and redshifts
        # First we make a dictionary of convergence sheet masses
        convergence_sheet_masses = {z:0 for z in self.redshift_list}
        
        # In the unsorted list, we'll put the main lens first
        lens_model_unsorted = (['SPEP', 'SHEAR', 'MULTIPOLE', 'MULTIPOLE'] +
                               ['TNFW' for i in range(N)] +
                               (['CONVERGENCE' for _ in convergence_sheet_masses]
                                if any_mass_sheets else []) )
        redshifts_unsorted = ([self.zl, self.zl, self.zl, self.zl] +
                              list(self.redshift_list) +
                              (sorted(convergence_sheet_masses.keys())
                               if any_mass_sheets else []) )

        # Then we sort everything
        sort_idx = np.argsort(redshifts_unsorted)
        lens_model_sorted = [lens_model_unsorted[i] for i in sort_idx]
        redshifts_sorted = [redshifts_unsorted[i] for i in sort_idx]

        self.main_lens_idx = np.where(sort_idx == 0)[0][0] # which lens is the main lens?

        self.lens_model_mp = LensModel(lens_model_list=lens_model_sorted,
                                       z_source = self.zs,
                                       lens_redshift_list=redshifts_sorted,
                                       multi_plane=True)

        # LENS kwargs
        self.kwargs_spep = {'theta_E': theta_lens, 'e1': e1, 'e2': e2, 
                            'gamma': gamma, 'center_x': self.center_lens_x, 'center_y': self.center_lens_y}
        self.kwargs_shear = {'gamma1': self.shear1, 'gamma2': self.shear2}
        self.kwargs_mp3 = {'m':3, 'a_m': self.lens_a3, 'phi_m': self.lens_phi3,
                           'center_x':self.center_lens_x, 'center_y':self.center_lens_y}
        self.kwargs_mp4 = {'m':4, 'a_m': self.lens_a4, 'phi_m': self.lens_phi4,
                           'center_x':self.center_lens_x, 'center_y':self.center_lens_y}

        kwargs_unsorted = [self.kwargs_spep, self.kwargs_shear, self.kwargs_mp3, self.kwargs_mp4] # (+ will append more)
        #
        for i in range(N): # (append interlopers)
            center_nfw_x = xpos_list[i]
            center_nfw_y = ypos_list[i]

            tau = 20 # assume 20 as default

            rs_adjusted = self.rs * (self.mass_list[i]/1e6)**(1/3.) # adjusted according to physical mass
            
            rsang = float(rs_angle(self.redshift_list[i],rs_adjusted))
            alphars = float(alpha_s_tnfw(self.mass_list[i],rs_adjusted,self.redshift_list[i],zs,tau))
            # alphars = float(alpha_s(self.mass_list[i],self.rs,self.redshift_list[i],zs)) # old result

            kwargs_nfw = {'Rs':rsang, 'alpha_Rs':alphars,
                          'r_trunc':tau*rsang,
                          'center_x': center_nfw_x, 'center_y': center_nfw_y}
            kwargs_unsorted.append(kwargs_nfw)
        #
        if any_mass_sheets: # (append negative convergence sheets)
            for i in range(N):
                if self.mass_sheets[i]:
                    convergence_sheet_masses[self.redshift_list[i]] += self.mass_list[i]
            for z, m in sorted(convergence_sheet_masses.items()):
                area_com = self.double_cone_width(z)**2 # kpc**2 comoving
                area = area_com / (1+z)**2 # kpc**2 physical
                sig = m/area # Msun / kpc**2

                
                # our normalization is the formula from assuming that this redshift
                # is the only lens
                sig_cr = sigma_cr(z, self.zs).to(u.Msun/u.Mpc**2).value / 1000**2 # from Msun/Mpc**2 to Msun/kpc**2

                kwargs_convergence_sheet = {'kappa': -sig/sig_cr} # todo check this calculation
                kwargs_unsorted.append(kwargs_convergence_sheet)

        self.kwargs_lens = [kwargs_unsorted[i] for i in sort_idx]
        
        ########################################################################
        # set up the list of light models to be used #

        # SOURCE light
        source_light_model_list = []
        for i in range(len(self.source_params)):
            source_light_model_list.append('SERSIC_ELLIPSE')
        for i in range(N_clump):
            source_light_model_list.append('SERSIC')

        # self.light_model_source = LightModel(light_model_list = source_light_model_list)

        # LENS light
        lens_light_model_list = ['SERSIC_ELLIPSE'] if is_lens_light else []
        self.light_model_lens = LightModel(light_model_list = lens_light_model_list) if is_lens_light else None

        # SOURCE light kwargs

        self.kwargs_source_mag = []
        ## First we add all the main (elliptical) sources.
        for mag, (r,phi,q,ra,dec,n) in zip(self.mag_source, self.source_params):
            e1s, e2s = param_util.phi_q2_ellipticity(phi=phi, q=q)
            self.kwargs_source_mag.append({'magnitude': mag,
                                           'R_sersic': r, 'n_sersic': n,
                                           'center_x': ra, 'center_y': dec,
                                           'e1': e1s, 'e2': e2s})
        ## Then we auto-add any additional clumps.
        assert(len(self.source_params) > 0)
        beta_ra_source = self.source_params[0][3]
        beta_dec_source = self.source_params[0][4]
        for i in range(N_clump):
            self.kwargs_source_mag.append({'magnitude': self.mag_source_clumps, 'R_sersic': self.r_sersic_source_clumps[i],
                                           'n_sersic': self.n_sersic_source_clumps[i],
                                           'center_x': beta_ra_source+self.source_scatter*(clumprandx[i]-.5), 
                                           'center_y': beta_dec_source+self.source_scatter*(clumprandy[i]-.5)})

        # LENS light kwargs
        self.kwargs_lens_mag = ([{'magnitude': self.mag_lens, 'R_sersic': theta_lens, 'n_sersic': self.n_sersic_lens_light,
                                  'e1': e1, 'e2': e2, 'center_x': self.center_lens_x , 'center_y': self.center_lens_y}]
                                if is_lens_light else None)
        
        ################################################################################
        # Setup data_class, i.e. pixelgrid #
        ra_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        dec_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # ''
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self.pixsize  # linear translation matrix of a shift in pixel in a shift in coordinates
        kwargs_pixel = {'nx': self.pixnum, 'ny': self.pixnum,  # number of pixels per axis
                        'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                        'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                        'transform_pix2angle': transform_pix2angle} 
        self.pixel_grid = PixelGrid(**kwargs_pixel)

        # # Setup PSF #
        # # (should not affect alpha calculations) #
        # kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
        #               'fwhm': 0.01,  # full width at half maximum of the Gaussian PSF (in angular units)
        #               'pixel_size': self.pixsize  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
        #              }
        # self.psf = PSF(**kwargs_psf)
        # kernel = self.psf.kernel_point_source

        
        # define the numerics #
        self.kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
                                'supersampling_convolution': False}

        # # initialize the Image model class by combining the modules we created above #
        # self.imageModel = ImageModel(data_class=self.pixel_grid, psf_class=self.psf,
        #                              lens_model_class=self.lens_model_mp,
        #                              source_model_class=self.light_model_source,
        #                              lens_light_model_class=self.light_model_lens,
        #                              kwargs_numerics=self.kwargs_numerics)

        # alternatively define kwargs_model (in case we want to generate the noisy image)
        self.kwargs_model = {'lens_model_list':lens_model_sorted,
                             'z_source':self.zs,
                             'lens_redshift_list':redshifts_sorted,
                             'source_light_model_list':source_light_model_list,
                             'lens_light_model_list': lens_light_model_list,
                             'cosmo':None} # cosmo could also be set to something else
    

    ## Commenting this out for now to make sure I don't accidentally use it :)
    # def calc_image(self):
    #     # simulate image with the parameters we have defined above #
    #     self.image = self.imageModel.image(kwargs_lens=self.kwargs_lens,
    #                                        kwargs_source=self.kwargs_light_source,
    #                                        kwargs_lens_light=self.kwargs_light_lens)#, kwargs_ps=kwargs_ps)

    def calc_image_imageModel(self):
        ## Note that we're still invoking SimAPI, not to generate the image or
        ## calculate noise, but simply to convert the magnitude of the source to
        ## an amplitude.

        num_exposures = 1

        hst = self.observation_config(band=self.observation_band, psf_type=self.observation_psf_type)
        kwargs_single_band = hst.kwargs_single_band()
        kwargs_single_band['num_exposures'] = num_exposures

        Hub = SimAPI(numpix=self.pixnum,
                     kwargs_single_band=kwargs_single_band,
                     kwargs_model=self.kwargs_model)

        _, kwargs_source_amp, _ = Hub.magnitude2amplitude(kwargs_source_mag=self.kwargs_source_mag)

        self.imageModel_image = self.imageModel.image(kwargs_lens=self.kwargs_lens,
                                                      kwargs_source=kwargs_source_amp,
                                                      kwargs_lens_light=None)



    def calc_image_noisy(self, num_exposures=1):
        ## calculate the image with noise ##
        hst = self.observation_config(band=self.observation_band, psf_type=self.observation_psf_type)
        kwargs_single_band = hst.kwargs_single_band()
        kwargs_single_band['num_exposures'] = num_exposures

        Hub = SimAPI(numpix=self.pixnum,
                     kwargs_single_band=kwargs_single_band,
                     kwargs_model=self.kwargs_model)
        hb_im = Hub.image_model_class(kwargs_numerics=self.kwargs_numerics)

        ## first we convert from magnitudes to amplitudes
        kwargs_lens_amp, kwargs_source_amp, _ = Hub.magnitude2amplitude(
            kwargs_lens_light_mag=self.kwargs_lens_mag,
            kwargs_source_mag=self.kwargs_source_mag)


        # then we calculate the clean image (used to calculate curl and div)
        self.image = hb_im.image(kwargs_lens=self.kwargs_lens,
                                 kwargs_source=kwargs_source_amp,
                                 kwargs_lens_light=kwargs_lens_amp)

        self.noise = Hub.noise_for_model(model=self.image)
        self.image_noisy = self.image + self.noise
        self.estimated_noise = Hub.estimate_noise(self.image)

    def calc_div_curl(self):
        # Calculates divergence and curl of alpha (from ray shooting)
        
        self.alphamat_x = np.zeros((self.pixnum, self.pixnum))
        self.alphamat_y = np.zeros((self.pixnum, self.pixnum))
        for xpix in range(self.pixnum):
            for ypix in range(self.pixnum):
                image_xy = self.pixel_grid.map_pix2coord(xpix, ypix) # in angle units
                source_xy = self.lens_model_mp.ray_shooting(image_xy[0], image_xy[1], self.kwargs_lens)
                self.alphamat_x[xpix,ypix] = image_xy[0] - source_xy[0]
                self.alphamat_y[xpix,ypix] = image_xy[1] - source_xy[1]

        self.divmat = (np.gradient(self.alphamat_x, self.pixsize)[0]
                       + np.gradient(self.alphamat_y, self.pixsize)[1])
        self.curlmat = (np.gradient(self.alphamat_y, self.pixsize)[0]
                        - np.gradient(self.alphamat_x, self.pixsize)[1])
        return self.divmat, self.curlmat
    
    def calc_div_curl_5pt(self):
        ## Calculates divergence and curl using 5pt stencil.
        self.alphamat_x = np.zeros((self.pixnum, self.pixnum))
        self.alphamat_y = np.zeros((self.pixnum, self.pixnum))
        for xpix in range(self.pixnum):
            for ypix in range(self.pixnum):
                self.calc_alpha_pixel(xpix, ypix)
                
        return self.recalc_div_curl_5pt()

    def calc_alpha_pixel(self, xpix, ypix):
        # Helper for `calc_div_curl_5pt` which calculates alpha for an individual pixel
        image_xy = self.pixel_grid.map_pix2coord(xpix, ypix) # in angle units
        source_xy = self.lens_model_mp.ray_shooting(image_xy[0], image_xy[1], self.kwargs_lens)
        self.alphamat_x[xpix,ypix] = image_xy[0] - source_xy[0]
        self.alphamat_y[xpix,ypix] = image_xy[1] - source_xy[1]
    
    def recalc_div_curl_5pt(self):
        ## Calculates divergence and curl using 5pt stencil, assuming alphamat_x, alphamat_y were already calculated.
        self.divmat = np.zeros([self.pixnum-4,self.pixnum-4])
        self.curlmat = np.zeros([self.pixnum-4,self.pixnum-4])
        
        def divfunc(vec_x, vec_y,i,j):
            diffx = (-1./12.)*(vec_x[i][j+2] - vec_x[i][j-2])+(2./3.)*(vec_x[i][j+1] - vec_x[i][j-1])
            diffy = (-1./12.)*(vec_y[i+2][j] - vec_y[i-2][j])+(2./3.)*(vec_y[i+1][j] - vec_y[i-1][j])
            return (diffx + diffy)*(1./self.pixsize)

        def curlfunc(vec_x, vec_y,i,j):
            offy = (-1./12.)*(vec_y[i][j+2] - vec_y[i][j-2])+(2./3.)*(vec_y[i][j+1] - vec_y[i][j-1])
            offx = (-1./12.)*(vec_x[i+2][j] - vec_x[i-2][j])+(2./3.)*(vec_x[i+1][j] - vec_x[i-1][j])
            return (offy - offx)*(1./self.pixsize)
        
        for i in range(2,self.pixnum-2):
            for j in range(2,self.pixnum-2):
                self.divmat[i-2][j-2] = divfunc(self.alphamat_y,self.alphamat_x,i,j)
                self.curlmat[i-2][j-2] = curlfunc(self.alphamat_y,self.alphamat_x,i,j)
                
        return self.divmat, self.curlmat
    
    
    def div_curl_simple(self):
       # Calculates divergence and curl of alpha by subtracting neighboring pixels
        self.alphamat_x = np.zeros((self.pixnum, self.pixnum))
        self.alphamat_y = np.zeros((self.pixnum, self.pixnum))
        for xpix in range(self.pixnum):
            for ypix in range(self.pixnum):
                image_xy = self.pixel_grid.map_pix2coord(xpix, ypix) # in angle units
                source_xy = self.lens_model_mp.ray_shooting(image_xy[0], image_xy[1], self.kwargs_lens)
                self.alphamat_x[xpix,ypix] = image_xy[0] - source_xy[0]
                self.alphamat_y[xpix,ypix] = image_xy[1] - source_xy[1]
                
        self.divmat = np.zeros([self.pixnum-2,self.pixnum-2])
        self.curlmat = np.zeros([self.pixnum-2,self.pixnum-2])
        
        def divfunc(vec_x, vec_y,i,j):
            diffx = vec_x[i][j+1] - vec_x[i][j-1]
            diffy = vec_y[i+1][j] - vec_y[i-1][j]
            return (diffx + diffy)*(0.5/self.pixsize)

        def curlfunc(vec_x, vec_y,i,j):
            offy = vec_y[i][j+1] - vec_y[i][j-1]
            offx = vec_x[i+1][j] - vec_x[i-1][j]
            return (offy - offx)*(0.5/self.pixsize)
        
        for i in range(1,self.pixnum-1):
            for j in range(1,self.pixnum-1):
                self.divmat[i-1][j-1] = divfunc(self.alphamat_y,self.alphamat_x,i,j)
                self.curlmat[i-1][j-1] = curlfunc(self.alphamat_y,self.alphamat_x,i,j)
                
        return self.divmat, self.curlmat

class CustomWeak(CustomImage):
    def single_cone_width(self, z):
        """(note that this helper is also used in COSMOSWeak)"""

        # Return comoving width (at redshift z, of the prism that represents what we can see) in kpc

        # First, we calculate the angular extent of this image.
        view_angle = self.pixsize*self.pixnum * np.pi/648000 # in radians
        source_width_com = view_angle * comdist(self.zs) # flat-space trick

        com_z = comdist(z)
        com_s = comdist(self.zs)

        width = (com_z / com_s) * source_width_com

        return width.to(u.kpc).value

    def __init__(self, xpos_list, ypos_list, redshift_list, m=None,
                 zs=1.0, pixsize=0.2, pixnum=200, mass_sheets=None,
                 N_clump=0, clump_seed=0,
                 source_params = [(.5, .5, .3, .01, .02, 1.5)],
                 r_sersic_source_clumps=0.1,
                 n_sersic_source_clumps=1.5,
                 source_scatter=1,
                 mag_source=20,
                 mag_source_clumps=22,
                 rs=1e-4):
        # change: used to take in `m` as a single mass for all interlopers, but
        # now this can also be a list of masses
        
        # `mass_sheets` : default set to True, meaning we should add negative mass sheets to cancel out all the substructure

        assert(len(xpos_list) == len(ypos_list))
        assert(len(xpos_list) == len(redshift_list))

        self.xpos_list = xpos_list
        self.ypos_list = ypos_list
        self.redshift_list = redshift_list
        self.N = len(xpos_list) # number of interlopers+subhalos
        N = self.N
        self.zs = zs
        if m is None:
            self.mass_list = [1e7] * self.N
        elif isinstance(m, float):
            self.mass_list = [m] * self.N
        else:
            self.mass_list = m

        self.pixsize = pixsize
        self.pixnum = pixnum

        if isinstance(mass_sheets, list) or isinstance(mass_sheets, np.ndarray):
            self.mass_sheets = mass_sheets
        elif mass_sheets is None or mass_sheets is True:
            self.mass_sheets = [True for _ in range(N)]
        elif mass_sheets is False:
            self.mass_sheets = [False for _ in range(N)]
        any_mass_sheets = np.any(self.mass_sheets)
            
        self.N_clump = N_clump
        self.clump_seed = clump_seed

        self.source_params = source_params
        # self.r_sersic_source = r_sersic_source
        # self.phi_source = phi_source
        # self.q_source = q_source
        # self.beta_ra_source = beta_ra_source
        # self.beta_dec_source = beta_dec_source
        # self.n_sersic_source = n_sersic_source
        
        if isinstance(r_sersic_source_clumps, list) or isinstance(r_sersic_source_clumps, np.ndarray):
            self.r_sersic_source_clumps = r_sersic_source_clumps
        else:
            # (repeat the same value for every clump)
            self.r_sersic_source_clumps = [r_sersic_source_clumps] * self.N_clump
        if isinstance(n_sersic_source_clumps, list) or isinstance(n_sersic_source_clumps, np.ndarray):
            self.n_sersic_source_clumps = n_sersic_source_clumps
        else:
            self.n_sersic_source_clumps = [n_sersic_source_clumps] * self.N_clump
        self.source_scatter = source_scatter
        
        if isinstance(mag_source, list) or isinstance(mag_source, np.ndarray):
            self.mag_source = mag_source
            assert(len(mag_source) == len(source_params))
        else:
            self.mag_source = [mag_source] * len(source_params)
        self.mag_source_clumps = mag_source_clumps

        self.rs = rs # scale radius of subhalo in Mpc (at 1e6 Msun pivot mass)

        ## SOURCE PROPERTIES ###############################################################################
        # e1s, e2s = param_util.phi_q2_ellipticity(phi=self.phi_source, q=self.q_source)
        # e1se2s_list = [param_util.phi_q2_ellipticity(phi=phi,q=q)
        #              for (_, phi, q, _, _, _) in self.source_params]
        

        ## SOURCE-CLUMP PROPERTIES #########################################################################

        # note: 1210.4562 was using Gaussians rather than Sersic

        np.random.seed(self.clump_seed)

        #r_sersic_source_clumps = 0.1
        clumprandx = np.random.rand(self.N_clump)
        clumprandy = np.random.rand(self.N_clump)

        #source_scatter = 1. ## This is how wide the scatter of the clumps over the smooth source

        #n_sersic_source_clumps = 1.5

        ####################################################################################################


        ## IMAGE PROPERTIES ################################################################################
        # self.pixsize = 0.2
        # self.pixnum = 200
        ####################################################################################################



        ## INTERLOPER PROPERTIES ########################################################################### 

        # for easier plotting only (current version only works when all the interlopers are at the lens redshift):
        # self.plot_xpixs = [self.x_to_pix(xpos) for xpos in xpos_list]
        # self.plot_ypixs = [self.y_to_pix(ypos), zl,pixsize,pixnum) for ypos in ypos_list]

        #beta_ra, beta_dec = beta_ras[0], beta_decs[0]

        # self.m = 1.0e7 # mass of interlopers (used to be 1e7, and then 1e9)
        # self.rs = 0.001  # interloper scale radius r_s
        # A = 80**2 ## in arcsec ## IGNORE THIS, THIS WAS FOR NEGATIVE CONVERGENCE
        # self.rs = 1e-4 # Mpc (pivot around m0=1e6)
        
        # kext = float(k_ext(N,m,A,zl,zs,pixsize))
        # note that there is no more self.rsang or self.alphars

        ## LENS model and redshifts
        # First we make a dictionary of convergence sheet masses
        convergence_sheet_masses = {z:0 for z in self.redshift_list}
        
        # In the unsorted list, we'll put the main lens first
        lens_model_unsorted = ['TNFW' for i in range(N)] + (['CONVERGENCE' for _ in convergence_sheet_masses]
                                                                       if any_mass_sheets else [])
        redshifts_unsorted = list(self.redshift_list) + (sorted(convergence_sheet_masses.keys())
                                                                     if any_mass_sheets else [])

        # Then we sort everything
        sort_idx = np.argsort(redshifts_unsorted)
        lens_model_sorted = [lens_model_unsorted[i] for i in sort_idx]
        redshifts_sorted = [redshifts_unsorted[i] for i in sort_idx]

        self.lens_model_mp = LensModel(lens_model_list=lens_model_sorted,
                                       z_source = self.zs,
                                       lens_redshift_list=redshifts_sorted,
                                       multi_plane=True)

        # LENS kwargs
        kwargs_unsorted = [] # (+ will append more)
        #
        for i in range(N): # (append interlopers)
            center_nfw_x = xpos_list[i]
            center_nfw_y = ypos_list[i]

            tau = 20 # assume 20 as default

            rs_adjusted = self.rs * (self.mass_list[i]/1e6)**(1/3.) # adjusted according to physical mass
            
            rsang = float(rs_angle(self.redshift_list[i],rs_adjusted))
            alphars = float(alpha_s_tnfw(self.mass_list[i],rs_adjusted,self.redshift_list[i],zs,tau))
            # alphars = float(alpha_s(self.mass_list[i],self.rs,self.redshift_list[i],zs)) # old result

            kwargs_nfw = {'Rs':rsang, 'alpha_Rs':alphars,
                          'r_trunc':tau*rsang,
                          'center_x': center_nfw_x, 'center_y': center_nfw_y}
            kwargs_unsorted.append(kwargs_nfw)
        #
        if any_mass_sheets: # (append negative convergence sheets)
            for i in range(N):
                if self.mass_sheets[i]:
                    convergence_sheet_masses[self.redshift_list[i]] += self.mass_list[i]
            for z, m in sorted(convergence_sheet_masses.items()):
                area_com = self.double_cone_width(z)**2 # kpc**2 comoving
                area = area_com / (1+z)**2 # kpc**2 physical
                sig = m/area # Msun / kpc**2

                
                # our normalization is the formula from assuming that this redshift
                # is the only lens
                sig_cr = sigma_cr(z, self.zs).to(u.Msun/u.Mpc**2).value / 1000**2 # from Msun/Mpc**2 to Msun/kpc**2

                kwargs_convergence_sheet = {'kappa': -sig/sig_cr} # todo check this calculation
                kwargs_unsorted.append(kwargs_convergence_sheet)

        self.kwargs_lens = [kwargs_unsorted[i] for i in sort_idx]
        self.kwargs_lens_mag = []
        
        ########################################################################
        # set up the list of light models to be used #

        # SOURCE light
        source_light_model_list = []
        for i in range(len(self.source_params)):
            source_light_model_list.append('SERSIC_ELLIPSE')
        for i in range(N_clump):
            source_light_model_list.append('SERSIC')

        self.light_model_source = LightModel(light_model_list = source_light_model_list)

        # # LENS light
        # lens_light_model_list = ['SERSIC_ELLIPSE'] if is_lens_light else []
        # self.light_model_lens = LightModel(light_model_list = lens_light_model_list) if is_lens_light else None

        # SOURCE light kwargs

        self.kwargs_source_mag = []
        ## First we add all the main (elliptical) sources.
        for mag, (r,phi,q,ra,dec,n) in zip(self.mag_source, self.source_params):
            e1s, e2s = param_util.phi_q2_ellipticity(phi=phi, q=q)
            self.kwargs_source_mag.append({'magnitude': mag,
                                           'R_sersic': r, 'n_sersic': n,
                                           'center_x': ra, 'center_y': dec,
                                           'e1': e1s, 'e2': e2s})
        ## Then we auto-add any additional clumps.
        assert(len(self.source_params) > 0)
        beta_ra_source = self.source_params[0][3]
        beta_dec_source = self.source_params[0][4]
        for i in range(N_clump):
            self.kwargs_source_mag.append({'magnitude': self.mag_source_clumps, 'R_sersic': self.r_sersic_source_clumps[i],
                                           'n_sersic': self.n_sersic_source_clumps[i],
                                           'center_x': beta_ra_source+self.source_scatter*(clumprandx[i]-.5), 
                                           'center_y': beta_dec_source+self.source_scatter*(clumprandy[i]-.5)})

        # LENS light kwargs
        # self.kwargs_lens_mag = ([{'magnitude': self.mag_lens, 'R_sersic': theta_lens, 'n_sersic': self.n_sersic_lens_light,
        #                           'e1': e1, 'e2': e2, 'center_x': self.center_lens_x , 'center_y': self.center_lens_y}]
        #                         if is_lens_light else None)
        
        ################################################################################
        # Setup data_class, i.e. pixelgrid #
        ra_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        dec_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # ''
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self.pixsize  # linear translation matrix of a shift in pixel in a shift in coordinates
        kwargs_pixel = {'nx': self.pixnum, 'ny': self.pixnum,  # number of pixels per axis
                        'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                        'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                        'transform_pix2angle': transform_pix2angle} 
        self.pixel_grid = PixelGrid(**kwargs_pixel)

        # Setup PSF #
        # # (should not affect alpha calculations) #
        # kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
        #               'fwhm': 0.01,  # full width at half maximum of the Gaussian PSF (in angular units)
        #               'pixel_size': self.pixsize  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
        #              }
        # self.psf = PSF(**kwargs_psf)
        # kernel = self.psf.kernel_point_source

        
        # define the numerics #
        self.kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
                                'supersampling_convolution': False}

        # # initialize the Image model class by combining the modules we created above #
        # self.imageModel = ImageModel(data_class=self.pixel_grid, psf_class=self.psf,
        #                              lens_model_class=self.lens_model_mp,
        #                              source_model_class=self.light_model_source,
        #                              lens_light_model_class=self.light_model_lens,
        #                              kwargs_numerics=self.kwargs_numerics)

        # alternatively define kwargs_model (in case we want to generate the noisy image)
        self.kwargs_model = {'lens_model_list':lens_model_sorted,
                             'z_source':self.zs,
                             'lens_redshift_list':redshifts_sorted,
                             'source_light_model_list':source_light_model_list,
                             # 'lens_light_model_list': lens_light_model_list,
                             'cosmo':None} # cosmo could also be set to something else
    


class COSMOSImage(CustomImage):
    def __init__(self, xpos_list, ypos_list, redshift_list, m=None, zl=0.2,
                 zs=1.0, pixsize=0.2, pixnum=200, mass_sheets=None, main_theta=0.3,
                 q_lens=1, phi_lens=-0.9,
                 center_lens_x=0, center_lens_y=0,
                 shear1=0, shear2=0,
                 lens_a3=0, lens_phi3=0,
                 lens_a4=0, lens_phi4=0,
                 is_lens_light=False,
                 n_sersic_lens_light=2,
                 gamma=2,
                 mag_lens=20,
                 source_cat_seed=0,
                 source_phi=0,
                 source_center_x=0, source_center_y=0,
                 rs=1e-4,
                 use_val_cat=False,
                 val_galaxies=None,
                 bad_galaxies=None,
                 observation_config=HSTELT,
                 observation_band='my_filter',
                 observation_psf_type='GAUSSIAN'):

        # `mass_sheets` : default set to True, meaning we should add negative mass sheets to cancel out all the substructure

        assert(len(xpos_list) == len(ypos_list))
        assert(len(xpos_list) == len(redshift_list))

        self.xpos_list = xpos_list
        self.ypos_list = ypos_list
        self.redshift_list = redshift_list
        self.N = len(xpos_list) # number of interlopers+subhalos
        N = self.N
        self.zl = zl
        self.zs = zs
        if m is None:
            self.mass_list = [1e7] * self.N
        elif isinstance(m, float):
            self.mass_list = [m] * self.N
        else:
            self.mass_list = m

        self.pixsize = pixsize
        self.pixnum = pixnum

        if isinstance(mass_sheets, list) or isinstance(mass_sheets, np.ndarray):
            self.mass_sheets = mass_sheets
        elif mass_sheets is None or mass_sheets is True:
            self.mass_sheets = [True for _ in range(N)]
        elif mass_sheets is False:
            self.mass_sheets = [False for _ in range(N)]
        any_mass_sheets = np.any(self.mass_sheets)
            
        self.main_theta = main_theta
        self.q_lens = q_lens
        self.phi_lens = phi_lens
        self.center_lens_x = center_lens_x
        self.center_lens_y = center_lens_y
        self.shear1 = shear1
        self.shear2 = shear2
        self.lens_a3 = lens_a3
        self.lens_phi3 = lens_phi3
        self.lens_a4 = lens_a4
        self.lens_phi4 = lens_phi4

        self.n_sersic_lens_light = n_sersic_lens_light
        self.is_lens_light = is_lens_light

        self.gamma = gamma # lens spep gamma

        # if isinstance(mag_source, list) or isinstance(mag_source, np.ndarray):
        #     self.mag_source = mag_source
        #     assert(len(mag_source) == len(source_params))
        # else:
        #     self.mag_source = [mag_source] * len(source_params)
        # self.mag_source_clumps = mag_source_clumps
        self.mag_lens = mag_lens

        self.source_cat_seed = source_cat_seed
        self.source_phi = source_phi
        self.source_center_x = source_center_x
        self.source_center_y = source_center_y

        self.rs = rs # scale radius of subhalo in Mpc (at 1e6 Msun pivot mass)

        self.use_val_cat = use_val_cat # Whether to use validation or training set

        ## List of galaxies reserved for the validation set
        self.val_galaxies = val_galaxies_global if val_galaxies is None else val_galaxies
        
        ## List of bad galaxies
        self.bad_galaxies = bad_galaxies_global if bad_galaxies is None else bad_galaxies


        self.observation_config = observation_config
        self.observation_band = observation_band
        self.observation_psf_type = observation_psf_type
        
        ## SOURCE PROPERTIES ###############################################################################

        cosmos_folder = '/n/holyscratch01/dvorkin_lab/Users/atsang/great3/COSMOS_23.5_training_sample'
        output_ab_zeropoint = 25.9463 # (25.127 was also used in PALTAS code. The HST zeropoint is 25.96.)

        source_parameters = {
            'z_source':zs,
            'cosmos_folder':cosmos_folder,
            'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
            'smoothing_sigma':0.00,'random_rotation':True,
            'output_ab_zeropoint':output_ab_zeropoint,
            'min_flux_radius':10.0,
            'center_x':self.source_center_x,
            'center_y':self.source_center_y}

        if self.use_val_cat:
            MyCOSMOSCatalog = COSMOSIncludeCatalog
            source_parameters['source_inclusion_list'] = self.val_galaxies
        else:
            MyCOSMOSCatalog = COSMOSExcludeCatalog
            source_parameters['source_exclusion_list'] = np.append(self.val_galaxies, self.bad_galaxies)

        cc = MyCOSMOSCatalog('planck18', source_parameters)

        np.random.seed(self.source_cat_seed)
        self.source_cat_i, _ = cc.fill_catalog_i_phi_defaults()
        self.source_model_list, self.source_kwargs_list = cc.draw_source(catalog_i=self.source_cat_i, phi=self.source_phi)

        # ## TODO possibly remove/edit the following lines
        # print('default scale', self.source_kwargs_list[0]['scale'])
        # self.source_kwargs_list[0]['scale'] *= 0.02/self.pixsize
        ####################################################################################################



        ## LENS PROPERTIES #################################################################################
        theta_lens = self.main_theta # used to be 10.
        r_theta_lens = x_to_xi(theta_lens,zl)
        e1, e2 = param_util.phi_q2_ellipticity(phi=self.phi_lens, q=self.q_lens) # used to be q=0.8
        # gamma = 2.

        # lens light properties
        # n_sersic_lens_light = 2.
        ####################################################################################################



        ## IMAGE PROPERTIES ################################################################################
        # self.pixsize = 0.2
        # self.pixnum = 200
        ####################################################################################################



        ## INTERLOPER PROPERTIES ########################################################################### 

        # for easier plotting only (current version only works when all the interlopers are at the lens redshift):
        # self.plot_xpixs = [self.x_to_pix(xpos) for xpos in xpos_list]
        # self.plot_ypixs = [self.y_to_pix(ypos), zl,pixsize,pixnum) for ypos in ypos_list]

        #beta_ra, beta_dec = beta_ras[0], beta_decs[0]

        # self.m = 1.0e7 # mass of interlopers (used to be 1e7, and then 1e9)
        # self.rs = 0.001  # interloper scale radius r_s
        # A = 80**2 ## in arcsec ## IGNORE THIS, THIS WAS FOR NEGATIVE CONVERGENCE
        # self.rs = 1e-4 # Mpc (pivot around m0=1e6)
        
        # kext = float(k_ext(N,m,A,zl,zs,pixsize))
        # note that there is no more self.rsang or self.alphars

        ## LENS model and redshifts
        # First we make a dictionary of convergence sheet masses
        convergence_sheet_masses = {z:0 for z in self.redshift_list}
        
        # In the unsorted list, we'll put the main lens first
        lens_model_unsorted = (['SPEP', 'SHEAR', 'MULTIPOLE', 'MULTIPOLE'] +
                               ['TNFW' for i in range(N)] +
                               (['CONVERGENCE' for _ in convergence_sheet_masses]
                                if any_mass_sheets else []) )
        redshifts_unsorted = ([self.zl, self.zl, self.zl, self.zl] +
                              list(self.redshift_list) +
                              (sorted(convergence_sheet_masses.keys())
                               if any_mass_sheets else []) )

        # Then we sort everything
        sort_idx = np.argsort(redshifts_unsorted)
        lens_model_sorted = [lens_model_unsorted[i] for i in sort_idx]
        redshifts_sorted = [redshifts_unsorted[i] for i in sort_idx]

        self.main_lens_idx = np.where(sort_idx == 0)[0][0] # which lens is the main lens?

        self.lens_model_mp = LensModel(lens_model_list=lens_model_sorted,
                                       z_source = self.zs,
                                       lens_redshift_list=redshifts_sorted,
                                       multi_plane=True)

        # LENS kwargs
        self.kwargs_spep = {'theta_E': theta_lens, 'e1': e1, 'e2': e2, 
                            'gamma': gamma, 'center_x': self.center_lens_x, 'center_y': self.center_lens_y}
        self.kwargs_shear = {'gamma1': self.shear1, 'gamma2': self.shear2}
        self.kwargs_mp3 = {'m':3, 'a_m': self.lens_a3, 'phi_m': self.lens_phi3,
                           'center_x':self.center_lens_x, 'center_y':self.center_lens_y}
        self.kwargs_mp4 = {'m':4, 'a_m': self.lens_a4, 'phi_m': self.lens_phi4,
                           'center_x':self.center_lens_x, 'center_y':self.center_lens_y}


        kwargs_unsorted = [self.kwargs_spep, self.kwargs_shear, self.kwargs_mp3, self.kwargs_mp4] # (+ will append more)
        #
        for i in range(N): # (append interlopers)
            center_nfw_x = xpos_list[i]
            center_nfw_y = ypos_list[i]

            tau = 20 # assume 20 as default

            rs_adjusted = self.rs * (self.mass_list[i]/1e6)**(1/3.) # adjusted according to physical mass
            # print('subhalo mass', self.mass_list[i])
            # print('rs_adjusted', rs_adjusted)
            
            rsang = float(rs_angle(self.redshift_list[i],rs_adjusted))
            alphars_tnfw = float(alpha_s_tnfw(self.mass_list[i],rs_adjusted,self.redshift_list[i],zs,tau))
            # alphars_nfw = float(alpha_s_nfw(self.mass_list[i],rs_adjusted,self.redshift_list[i],zs))
            # alphars_old = float(alpha_s(self.mass_list[i],rs_adjusted,self.redshift_list[i],zs)) # old result
            # print('alphars comparison', alphars_tnfw, alphars_nfw, alphars_old)
            alphars = alphars_tnfw
            
            kwargs_nfw = {'Rs':rsang, 'alpha_Rs':alphars,
                          'r_trunc':tau*rsang,
                          'center_x': center_nfw_x, 'center_y': center_nfw_y}
            kwargs_unsorted.append(kwargs_nfw)
        #
        if any_mass_sheets: # (append negative convergence sheets)
            for i in range(N):
                if self.mass_sheets[i]:
                    convergence_sheet_masses[self.redshift_list[i]] += self.mass_list[i]
            for z, m in sorted(convergence_sheet_masses.items()):
                area_com = self.double_cone_width(z)**2 # kpc**2 comoving
                area = area_com / (1+z)**2 # kpc**2 physical
                sig = m/area # Msun / kpc**2

                
                # our normalization is the formula from assuming that this redshift
                # is the only lens
                sig_cr = sigma_cr(z, self.zs).to(u.Msun/u.Mpc**2).value / 1000**2 # from Msun/Mpc**2 to Msun/kpc**2

                kwargs_convergence_sheet = {'kappa': -sig/sig_cr} # todo check this calculation
                kwargs_unsorted.append(kwargs_convergence_sheet)

        self.kwargs_lens = [kwargs_unsorted[i] for i in sort_idx]
        
        ########################################################################
        # set up the list of light models to be used #

        # # SOURCE light
        # source_light_model_list = []
        # for i in range(len(self.source_params)):
        #     source_light_model_list.append('SERSIC_ELLIPSE')
        # for i in range(N_clump):
        #     source_light_model_list.append('SERSIC')

        # self.light_model_source = LightModel(light_model_list = source_light_model_list)

        # LENS light
        lens_light_model_list = ['SERSIC_ELLIPSE'] if is_lens_light else []
        self.light_model_lens = LightModel(light_model_list = lens_light_model_list) if is_lens_light else None

        # # SOURCE light kwargs

        # self.kwargs_source_mag = []
        # ## First we add all the main (elliptical) sources.
        # for mag, (r,phi,q,ra,dec,n) in zip(self.mag_source, self.source_params):
        #     e1s, e2s = param_util.phi_q2_ellipticity(phi=phi, q=q)
        #     self.kwargs_source_mag.append({'magnitude': mag,
        #                                    'R_sersic': r, 'n_sersic': n,
        #                                    'center_x': ra, 'center_y': dec,
        #                                    'e1': e1s, 'e2': e2s})
        # ## Then we auto-add any additional clumps.
        # assert(len(self.source_params) > 0)
        # beta_ra_source = self.source_params[0][3]
        # beta_dec_source = self.source_params[0][4]
        # for i in range(N_clump):
        #     self.kwargs_source_mag.append({'magnitude': self.mag_source_clumps, 'R_sersic': self.r_sersic_source_clumps[i],
        #                                    'n_sersic': self.n_sersic_source_clumps[i],
        #                                    'center_x': beta_ra_source+self.source_scatter*(clumprandx[i]-.5), 
        #                                    'center_y': beta_dec_source+self.source_scatter*(clumprandy[i]-.5)})

        # LENS light kwargs
        self.kwargs_lens_mag = ([{'magnitude': self.mag_lens, 'R_sersic': theta_lens, 'n_sersic': self.n_sersic_lens_light,
                                  'e1': e1, 'e2': e2, 'center_x': self.center_lens_x , 'center_y': self.center_lens_y}]
                                if is_lens_light else None)
        
        # ################################################################################
        # Setup data_class, i.e. pixelgrid #
        ra_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        dec_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # ''
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self.pixsize  # linear translation matrix of a shift in pixel in a shift in coordinates
        kwargs_pixel = {'nx': self.pixnum, 'ny': self.pixnum,  # number of pixels per axis
                        'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                        'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                        'transform_pix2angle': transform_pix2angle} 
        self.pixel_grid = PixelGrid(**kwargs_pixel)

        # # Setup PSF #
        # # (should not affect alpha calculations) #
        # kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
        #               'fwhm': 0.01,  # full width at half maximum of the Gaussian PSF (in angular units)
        #               'pixel_size': self.pixsize  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
        #              }
        # self.psf = PSF(**kwargs_psf)
        # kernel = self.psf.kernel_point_source

        
        # define the numerics #
        self.kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
                                'supersampling_convolution': False}

        # # initialize the Image model class by combining the modules we created above #
        # self.imageModel = ImageModel(data_class=self.pixel_grid, psf_class=self.psf,
        #                              lens_model_class=self.lens_model_mp,
        #                              source_model_class=self.source_model_list,
        #                              lens_light_model_class=self.light_model_lens,
        #                              kwargs_numerics=self.kwargs_numerics)

        # alternatively define kwargs_model (in case we want to generate the noisy image)
        self.kwargs_model = {'lens_model_list':lens_model_sorted,
                             'z_source':self.zs,
                             'lens_redshift_list':redshifts_sorted,
                             'source_light_model_list':self.source_model_list,
                             'lens_light_model_list': lens_light_model_list,
                             'cosmo':None} # cosmo could also be set to something else

    def calc_image_noisy(self, num_exposures=1):
        ## calculate the image with noise ##
        # hst = HST(band='WFC3_F160W', psf_type='PIXEL')
        hst = self.observation_config(band=self.observation_band, psf_type=self.observation_psf_type)
        kwargs_single_band = hst.kwargs_single_band()
        kwargs_single_band['num_exposures'] = num_exposures

        Hub = SimAPI(numpix=self.pixnum,
                     kwargs_single_band=kwargs_single_band,
                     kwargs_model=self.kwargs_model)
        hb_im = Hub.image_model_class(kwargs_numerics=self.kwargs_numerics)

        ## first we convert from magnitudes to amplitudes
        kwargs_lens_amp, _, _ = Hub.magnitude2amplitude(
            kwargs_lens_light_mag=self.kwargs_lens_mag,
            kwargs_source_mag=None)

        # then we calculate the clean image (used to calculate curl and div)
        self.image = hb_im.image(kwargs_lens=self.kwargs_lens,
                                 kwargs_source=self.source_kwargs_list,
                                 kwargs_lens_light=kwargs_lens_amp)

        self.noise = Hub.noise_for_model(model=self.image)
        self.image_noisy = self.image + self.noise
        self.estimated_noise = Hub.estimate_noise(self.image)
        

class COSMOSWeak(COSMOSImage):
    def single_cone_width(self, z):
        # Return comoving width (at redshift z, of the prism that represents what we can see) in kpc

        # First, we calculate the angular extent of this image.
        view_angle = self.pixsize*self.pixnum * np.pi/648000 # in radians
        source_width_com = view_angle * comdist(self.zs) # flat-space trick

        com_z = comdist(z)
        com_s = comdist(self.zs)

        width = (com_z / com_s) * source_width_com

        return width.to(u.kpc).value

    def __init__(self, xpos_list, ypos_list, redshift_list, m=None,
                 zs=1.0, pixsize=0.2, pixnum=200, mass_sheets=None,
                 source_cat_seed=0,
                 source_phi=0,
                 source_center_x=0, source_center_y=0,
                 rs=1e-4,
                 use_val_cat=False,
                 val_galaxies=None,
                 bad_galaxies=None,
                 observation_config=HSTELT,
                 observation_band='my_filter',
                 observation_psf_type='GAUSSIAN'):

        # `mass_sheets` : default set to True, meaning we should add negative mass sheets to cancel out all the substructure

        assert(len(xpos_list) == len(ypos_list))
        assert(len(xpos_list) == len(redshift_list))

        self.xpos_list = xpos_list
        self.ypos_list = ypos_list
        self.redshift_list = redshift_list
        self.N = len(xpos_list) # number of interlopers+subhalos
        N = self.N
        self.zs = zs
        if m is None:
            self.mass_list = [1e7] * self.N
        elif isinstance(m, float):
            self.mass_list = [m] * self.N
        else:
            self.mass_list = m

        self.pixsize = pixsize
        self.pixnum = pixnum

        if isinstance(mass_sheets, list) or isinstance(mass_sheets, np.ndarray):
            self.mass_sheets = mass_sheets
        elif mass_sheets is None or mass_sheets is True:
            self.mass_sheets = [True for _ in range(N)]
        elif mass_sheets is False:
            self.mass_sheets = [False for _ in range(N)]
        any_mass_sheets = np.any(self.mass_sheets)
            
        self.source_cat_seed = source_cat_seed
        self.source_phi = source_phi
        self.source_center_x = source_center_x
        self.source_center_y = source_center_y

        self.rs = rs # scale radius of subhalo in Mpc (at 1e6 Msun pivot mass)

        self.use_val_cat = use_val_cat # Whether to use validation or training set

        ## List of galaxies reserved for the validation set
        self.val_galaxies = val_galaxies_global if val_galaxies is None else val_galaxies
        
        ## List of bad galaxies
        self.bad_galaxies = bad_galaxies_global if bad_galaxies is None else bad_galaxies


        self.observation_config = observation_config
        self.observation_band = observation_band
        self.observation_psf_type = observation_psf_type
        
        ## SOURCE PROPERTIES ###############################################################################

        cosmos_folder = '/n/holyscratch01/dvorkin_lab/Users/atsang/great3/COSMOS_23.5_training_sample'
        output_ab_zeropoint = 25.9463 # (25.127 was also used in PALTAS code. The HST zeropoint is 25.96.)

        source_parameters = {
            'z_source':zs,
            'cosmos_folder':cosmos_folder,
            'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
            'smoothing_sigma':0.00,'random_rotation':True,
            'output_ab_zeropoint':output_ab_zeropoint,
            'min_flux_radius':10.0,
            'center_x':self.source_center_x,
            'center_y':self.source_center_y}

        if self.use_val_cat:
            MyCOSMOSCatalog = COSMOSIncludeCatalog
            source_parameters['source_inclusion_list'] = self.val_galaxies
        else:
            MyCOSMOSCatalog = COSMOSExcludeCatalog
            source_parameters['source_exclusion_list'] = np.append(self.val_galaxies, self.bad_galaxies)

        cc = MyCOSMOSCatalog('planck18', source_parameters)

        np.random.seed(self.source_cat_seed)
        self.source_cat_i, _ = cc.fill_catalog_i_phi_defaults()
        self.source_model_list, self.source_kwargs_list = cc.draw_source(catalog_i=self.source_cat_i, phi=self.source_phi)

        ####################################################################################################


        ## INTERLOPER PROPERTIES ########################################################################### 

        ## LENS model and redshifts
        # First we make a dictionary of convergence sheet masses
        convergence_sheet_masses = {z:0 for z in self.redshift_list}
        
        # In the unsorted list, we'll put the main lens first
        lens_model_unsorted = (['TNFW' for i in range(N)] +
                               (['CONVERGENCE' for _ in convergence_sheet_masses]
                                if any_mass_sheets else []) )
        redshifts_unsorted = (list(self.redshift_list) +
                              (sorted(convergence_sheet_masses.keys())
                               if any_mass_sheets else []) )

        # Then we sort everything
        sort_idx = np.argsort(redshifts_unsorted)
        lens_model_sorted = [lens_model_unsorted[i] for i in sort_idx]
        redshifts_sorted = [redshifts_unsorted[i] for i in sort_idx]

        self.lens_model_mp = LensModel(lens_model_list=lens_model_sorted,
                                       z_source = self.zs,
                                       lens_redshift_list=redshifts_sorted,
                                       multi_plane=True)

        # LENS kwargs
        kwargs_unsorted = [] # (+ will append more)
        #
        for i in range(N): # (append interlopers)
            center_nfw_x = xpos_list[i]
            center_nfw_y = ypos_list[i]

            tau = 20 # assume 20 as default

            rs_adjusted = self.rs * (self.mass_list[i]/1e6)**(1/3.) # adjusted according to physical mass
            rsang = float(rs_angle(self.redshift_list[i],rs_adjusted))
            alphars_tnfw = float(alpha_s_tnfw(self.mass_list[i],rs_adjusted,self.redshift_list[i],zs,tau))
            alphars = alphars_tnfw
            
            kwargs_nfw = {'Rs':rsang, 'alpha_Rs':alphars,
                          'r_trunc':tau*rsang,
                          'center_x': center_nfw_x, 'center_y': center_nfw_y}
            kwargs_unsorted.append(kwargs_nfw)
        #
        if any_mass_sheets: # (append negative convergence sheets)
            for i in range(N):
                if self.mass_sheets[i]:
                    convergence_sheet_masses[self.redshift_list[i]] += self.mass_list[i]
            for z, m in sorted(convergence_sheet_masses.items()):
                area_com = self.double_cone_width(z)**2 # kpc**2 comoving
                area = area_com / (1+z)**2 # kpc**2 physical
                sig = m/area # Msun / kpc**2
                
                # our normalization is the formula from assuming that this redshift
                # is the only lens
                sig_cr = sigma_cr(z, self.zs).to(u.Msun/u.Mpc**2).value / 1000**2 # from Msun/Mpc**2 to Msun/kpc**2

                kwargs_convergence_sheet = {'kappa': -sig/sig_cr} # todo check this calculation
                kwargs_unsorted.append(kwargs_convergence_sheet)

        self.kwargs_lens = [kwargs_unsorted[i] for i in sort_idx]

        ## kwargs_lens_mag
        self.kwargs_lens_mag = None
        
        # ################################################################################
        # Setup data_class, i.e. pixelgrid #
        ra_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        dec_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # ''
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self.pixsize  # linear translation matrix of a shift in pixel in a shift in coordinates
        kwargs_pixel = {'nx': self.pixnum, 'ny': self.pixnum,  # number of pixels per axis
                        'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                        'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                        'transform_pix2angle': transform_pix2angle} 
        self.pixel_grid = PixelGrid(**kwargs_pixel)
        
        # define the numerics #
        self.kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
                                'supersampling_convolution': False}

        # alternatively define kwargs_model (in case we want to generate the noisy image)
        self.kwargs_model = {'lens_model_list':lens_model_sorted,
                             'z_source':self.zs,
                             'lens_redshift_list':redshifts_sorted,
                             'source_light_model_list':self.source_model_list,
                             'lens_light_model_list': [],
                             'cosmo':None} # cosmo could also be set to something else

    
    # def __init__(self, xpos_list, ypos_list, redshift_list, m=None,
    #              zs=1.0, pixsize=0.2, pixnum=200, mass_sheets=None,
    #              source_cat_seed=0,
    #              source_phi=0,
    #              source_center_x=0, source_center_y=0,
    #              rs=1e-4):

    #     # `mass_sheets` : default set to True, meaning we should add negative mass sheets to cancel out all the substructure

    #     assert(len(xpos_list) == len(ypos_list))
    #     assert(len(xpos_list) == len(redshift_list))

    #     self.xpos_list = xpos_list
    #     self.ypos_list = ypos_list
    #     self.redshift_list = redshift_list
    #     self.N = len(xpos_list) # number of interlopers+subhalos
    #     N = self.N
    #     self.zs = zs
    #     if m is None:
    #         self.mass_list = [1e7] * self.N
    #     elif isinstance(m, float):
    #         self.mass_list = [m] * self.N
    #     else:
    #         self.mass_list = m

    #     self.pixsize = pixsize
    #     self.pixnum = pixnum

    #     if isinstance(mass_sheets, list) or isinstance(mass_sheets, np.ndarray):
    #         self.mass_sheets = mass_sheets
    #     elif mass_sheets is None or mass_sheets is True:
    #         self.mass_sheets = [True for _ in range(N)]
    #     elif mass_sheets is False:
    #         self.mass_sheets = [False for _ in range(N)]
    #     any_mass_sheets = np.any(self.mass_sheets)

    #     self.source_cat_seed = source_cat_seed
    #     self.source_phi = source_phi
    #     self.source_center_x = source_center_x
    #     self.source_center_y = source_center_y

    #     self.rs = rs # scale radius of subhalo in Mpc (at 1e6 Msun pivot mass)

    #     ## SOURCE PROPERTIES ###############################################################################

    #     cosmos_folder = '/n/holyscratch01/dvorkin_lab/Users/atsang/great3/COSMOS_23.5_training_sample'
    #     output_ab_zeropoint = 25.9463 # (25.127 was also used in PALTAS code. The HST zeropoint is 25.96.)

    #     source_parameters = {
    #         'z_source':zs,
    #         'cosmos_folder':cosmos_folder,
    #         'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
    #         'smoothing_sigma':0.00,'random_rotation':True,
    #         'output_ab_zeropoint':output_ab_zeropoint,
    #         'min_flux_radius':10.0,
    #         'center_x':self.source_center_x,
    #         'center_y':self.source_center_y}

    #     cc = COSMOSCatalog('planck18', source_parameters)

    #     np.random.seed(self.source_cat_seed)
    #     self.source_cat_i, _ = cc.fill_catalog_i_phi_defaults()
    #     self.source_model_list, self.source_kwargs_list = cc.draw_source(catalog_i=self.source_cat_i, phi=self.source_phi)

    #     self.source_kwargs_list[0]['scale'] *= 0.08/self.pixsize # hack to make the scale bigger
    #     # print('scale', self.source_kwargs_list[0]['scale']) ## TODO remove


    #     ####################################################################################################


    #     ## IMAGE PROPERTIES ################################################################################
    #     # self.pixsize = 0.2
    #     # self.pixnum = 200
    #     ####################################################################################################



    #     ## INTERLOPER PROPERTIES ########################################################################### 

    #     # for easier plotting only (current version only works when all the interlopers are at the lens redshift):
    #     # self.plot_xpixs = [self.x_to_pix(xpos) for xpos in xpos_list]
    #     # self.plot_ypixs = [self.y_to_pix(ypos), zl,pixsize,pixnum) for ypos in ypos_list]

    #     #beta_ra, beta_dec = beta_ras[0], beta_decs[0]

    #     # self.m = 1.0e7 # mass of interlopers (used to be 1e7, and then 1e9)
    #     # self.rs = 0.001  # interloper scale radius r_s
    #     # A = 80**2 ## in arcsec ## IGNORE THIS, THIS WAS FOR NEGATIVE CONVERGENCE
    #     # self.rs = 1e-4 # Mpc (pivot around m0=1e6)
        
    #     # kext = float(k_ext(N,m,A,zl,zs,pixsize))
    #     # note that there is no more self.rsang or self.alphars

    #     ## LENS model and redshifts
    #     # First we make a dictionary of convergence sheet masses
    #     convergence_sheet_masses = {z:0 for z in self.redshift_list}
        
    #     # In the unsorted list, we'll put the main lens first
    #     lens_model_unsorted = ['TNFW' for i in range(N)] + (['CONVERGENCE' for _ in convergence_sheet_masses]
    #                                                         if any_mass_sheets else [])
    #     redshifts_unsorted = list(self.redshift_list) + (sorted(convergence_sheet_masses.keys())
    #                                                      if any_mass_sheets else [])

    #     # Then we sort everything
    #     sort_idx = np.argsort(redshifts_unsorted)
    #     lens_model_sorted = [lens_model_unsorted[i] for i in sort_idx]
    #     redshifts_sorted = [redshifts_unsorted[i] for i in sort_idx]

    #     self.lens_model_mp = LensModel(lens_model_list=lens_model_sorted,
    #                                    z_source = self.zs,
    #                                    lens_redshift_list=redshifts_sorted,
    #                                    multi_plane=True)

    #     # LENS kwargs
    #     kwargs_unsorted = [] # (+ will append more)
    #     #
    #     for i in range(N): # (append interlopers)
    #         center_nfw_x = xpos_list[i]
    #         center_nfw_y = ypos_list[i]

    #         tau = 20 # assume 20 as default

    #         rs_adjusted = self.rs * (self.mass_list[i]/1e6)**(1/3.) # adjusted according to physical mass
            
    #         rsang = float(rs_angle(self.redshift_list[i],rs_adjusted))
    #         alphars = float(alpha_s_tnfw(self.mass_list[i],rs_adjusted,self.redshift_list[i],zs,tau))
    #         # alphars = float(alpha_s(self.mass_list[i],self.rs,self.redshift_list[i],zs)) # old result

    #         kwargs_nfw = {'Rs':rsang, 'alpha_Rs':alphars,
    #                       'r_trunc':tau*rsang,
    #                       'center_x': center_nfw_x, 'center_y': center_nfw_y}
    #         kwargs_unsorted.append(kwargs_nfw)
    #     #
    #     if any_mass_sheets: # (append negative convergence sheets)
    #         for i in range(N):
    #             if self.mass_sheets[i]:
    #                 convergence_sheet_masses[self.redshift_list[i]] += self.mass_list[i]
    #         for z, m in sorted(convergence_sheet_masses.items()):
    #             area_com = self.single_cone_width(z)**2 # kpc**2 comoving
    #             area = area_com / (1+z)**2 # kpc**2 physical
    #             sig = m/area # Msun / kpc**2

                
    #             # our normalization is the formula from assuming that this redshift
    #             # is the only lens
    #             sig_cr = sigma_cr(z, self.zs).to(u.Msun/u.Mpc**2).value / 1000**2 # from Msun/Mpc**2 to Msun/kpc**2

    #             kwargs_convergence_sheet = {'kappa': -sig/sig_cr} # todo check this calculation
    #             kwargs_unsorted.append(kwargs_convergence_sheet)

    #     self.kwargs_lens = [kwargs_unsorted[i] for i in sort_idx]
    #     self.kwargs_lens_mag = [] # trying this instead of None
        
    #     # ################################################################################
    #     # Setup data_class, i.e. pixelgrid #
    #     ra_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
    #     dec_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # ''
    #     transform_pix2angle = np.array([[1, 0], [0, 1]]) * self.pixsize  # linear translation matrix of a shift in pixel in a shift in coordinates
    #     kwargs_pixel = {'nx': self.pixnum, 'ny': self.pixnum,  # number of pixels per axis
    #                     'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
    #                     'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
    #                     'transform_pix2angle': transform_pix2angle} 
    #     self.pixel_grid = PixelGrid(**kwargs_pixel)

    #     # # Setup PSF #
    #     # # (should not affect alpha calculations) #
    #     # kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
    #     #               'fwhm': 0.01,  # full width at half maximum of the Gaussian PSF (in angular units)
    #     #               'pixel_size': self.pixsize  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
    #     #              }
    #     # self.psf = PSF(**kwargs_psf)
    #     # kernel = self.psf.kernel_point_source

        
    #     # define the numerics #
    #     self.kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
    #                             'supersampling_convolution': False}

    #     # # initialize the Image model class by combining the modules we created above #
    #     # self.imageModel = ImageModel(data_class=self.pixel_grid, psf_class=self.psf,
    #     #                              lens_model_class=self.lens_model_mp,
    #     #                              source_model_class=self.source_model_list,
    #     #                              lens_light_model_class=self.light_model_lens,
    #     #                              kwargs_numerics=self.kwargs_numerics)

    #     # alternatively define kwargs_model (in case we want to generate the noisy image)
    #     self.kwargs_model = {'lens_model_list':lens_model_sorted,
    #                          'z_source':self.zs,
    #                          'lens_redshift_list':redshifts_sorted,
    #                          'source_light_model_list':self.source_model_list,
    #                          # 'lens_light_model_list': lens_light_model_list,
    #                          'cosmo':None} # cosmo could also be set to something else


class ShapeletImage(COSMOSImage):
    def __init__(self, xpos_list, ypos_list, redshift_list, m=None, zl=0.2,
                 zs=1.0, pixsize=0.2, pixnum=200, mass_sheets=None, main_theta=0.3,
                 q_lens=1, phi_lens=-0.9,
                 center_lens_x=0, center_lens_y=0,
                 is_lens_light=False,
                 n_sersic_lens_light=2,
                 gamma=2,
                 mag_lens=20,
                 source_nmax=25,
                 source_beta=.1,
                 source_coeffs=None,
                 rs=1e-4):

        # `mass_sheets` : default set to True, meaning we should add negative mass sheets to cancel out all the substructure

        assert(len(xpos_list) == len(ypos_list))
        assert(len(xpos_list) == len(redshift_list))

        self.xpos_list = xpos_list
        self.ypos_list = ypos_list
        self.redshift_list = redshift_list
        self.N = len(xpos_list) # number of interlopers+subhalos
        N = self.N
        self.zl = zl
        self.zs = zs
        if m is None:
            self.mass_list = [1e7] * self.N
        elif isinstance(m, float):
            self.mass_list = [m] * self.N
        else:
            self.mass_list = m

        self.pixsize = pixsize
        self.pixnum = pixnum

        if isinstance(mass_sheets, list) or isinstance(mass_sheets, np.ndarray):
            self.mass_sheets = mass_sheets
        elif mass_sheets is None or mass_sheets is True:
            self.mass_sheets = [True for _ in range(N)]
        elif mass_sheets is False:
            self.mass_sheets = [False for _ in range(N)]
        any_mass_sheets = np.any(self.mass_sheets)
            
        self.main_theta = main_theta
        self.q_lens = q_lens
        self.phi_lens = phi_lens
        self.center_lens_x = center_lens_x
        self.center_lens_y = center_lens_y

        self.n_sersic_lens_light = n_sersic_lens_light
        self.is_lens_light = is_lens_light

        self.gamma = gamma # lens spep gamma

        self.mag_lens = mag_lens

        assert(source_coeffs is not None)
        assert(len(source_coeffs) == (source_nmax + 1) * (source_nmax + 2) // 2)
        self.source_nmax = source_nmax
        self.source_beta = source_beta
        self.source_coeffs = source_coeffs

        self.rs = rs # scale radius of subhalo in Mpc (at 1e6 Msun pivot mass)

        ## SOURCE PROPERTIES ###############################################################################

        self.source_model_list = ['INTERPOL']

        source_pix_factor = 5
        source_pixsize = self.pixsize / source_pix_factor
        source_pixnum = round(self.pixnum * source_pix_factor)
        ds = DiffShapelets(self.source_nmax, self.source_beta,
                           pixnum=source_pixnum, pixsize=source_pixsize)
        source_image = ds.reco_source(self.source_coeffs)

        self.source_kwargs_list = [{'image':source_image,
                                    'center_x':0, 'center_y':0,
                                    'phi_G':0,
                                    'scale':source_pixsize,
                                    'amp': 1/self.pixsize**2}]

        # cosmos_folder = '/n/holyscratch01/dvorkin_lab/Users/atsang/great3/COSMOS_23.5_training_sample'
        # output_ab_zeropoint = 25.9463 # (25.127 was also used in PALTAS code. The HST zeropoint is 25.96.)

        # source_parameters = {
        #     'z_source':zs,
        #     'cosmos_folder':cosmos_folder,
        #     'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
        #     'smoothing_sigma':0.00,'random_rotation':True,
        #     'output_ab_zeropoint':output_ab_zeropoint,
        #     'min_flux_radius':10.0,
        #     'center_x':self.source_center_x,
        #     'center_y':self.source_center_y}

        # cc = COSMOSCatalog('planck18', source_parameters)

        # np.random.seed(self.source_cat_seed)
        # self.source_cat_i, _ = cc.fill_catalog_i_phi_defaults()
        # self.source_model_list, self.source_kwargs_list = cc.draw_source(catalog_i=self.source_cat_i, phi=self.source_phi)

        ####################################################################################################



        ## LENS PROPERTIES #################################################################################
        theta_lens = self.main_theta # used to be 10.
        r_theta_lens = x_to_xi(theta_lens,zl)
        e1, e2 = param_util.phi_q2_ellipticity(phi=self.phi_lens, q=self.q_lens) # used to be q=0.8
        # gamma = 2.

        # lens light properties
        # n_sersic_lens_light = 2.
        ####################################################################################################



        ## IMAGE PROPERTIES ################################################################################
        # self.pixsize = 0.2
        # self.pixnum = 200
        ####################################################################################################



        ## INTERLOPER PROPERTIES ########################################################################### 

        # for easier plotting only (current version only works when all the interlopers are at the lens redshift):
        # self.plot_xpixs = [self.x_to_pix(xpos) for xpos in xpos_list]
        # self.plot_ypixs = [self.y_to_pix(ypos), zl,pixsize,pixnum) for ypos in ypos_list]

        #beta_ra, beta_dec = beta_ras[0], beta_decs[0]

        # self.m = 1.0e7 # mass of interlopers (used to be 1e7, and then 1e9)
        # self.rs = 0.001  # interloper scale radius r_s
        # A = 80**2 ## in arcsec ## IGNORE THIS, THIS WAS FOR NEGATIVE CONVERGENCE
        # self.rs = 1e-4 # Mpc (pivot around m0=1e6)
        
        # kext = float(k_ext(N,m,A,zl,zs,pixsize))
        # note that there is no more self.rsang or self.alphars

        ## LENS model and redshifts
        # First we make a dictionary of convergence sheet masses
        convergence_sheet_masses = {z:0 for z in self.redshift_list}
        
        # In the unsorted list, we'll put the main lens first
        lens_model_unsorted = ['SPEP'] + ['TNFW' for i in range(N)] + (['CONVERGENCE' for _ in convergence_sheet_masses]
                                                                       if any_mass_sheets else [])
        redshifts_unsorted = [self.zl] + list(self.redshift_list) + (sorted(convergence_sheet_masses.keys())
                                                                     if any_mass_sheets else [])

        # Then we sort everything
        sort_idx = np.argsort(redshifts_unsorted)
        lens_model_sorted = [lens_model_unsorted[i] for i in sort_idx]
        redshifts_sorted = [redshifts_unsorted[i] for i in sort_idx]

        self.main_lens_idx = np.where(sort_idx == 0)[0][0] # which lens is the main lens?

        self.lens_model_mp = LensModel(lens_model_list=lens_model_sorted,
                                       z_source = self.zs,
                                       lens_redshift_list=redshifts_sorted,
                                       multi_plane=True)

        # LENS kwargs
        self.kwargs_spep = {'theta_E': theta_lens, 'e1': e1, 'e2': e2, 
                            'gamma': gamma, 'center_x': self.center_lens_x, 'center_y': self.center_lens_y}

        kwargs_unsorted = [self.kwargs_spep] # (+ will append more)
        #
        for i in range(N): # (append interlopers)
            center_nfw_x = xpos_list[i]
            center_nfw_y = ypos_list[i]

            tau = 20 # assume 20 as default

            rs_adjusted = self.rs * (self.mass_list[i]/1e6)**(1/3.) # adjusted according to physical mass
            
            rsang = float(rs_angle(self.redshift_list[i],rs_adjusted))
            alphars = float(alpha_s_tnfw(self.mass_list[i],rs_adjusted,self.redshift_list[i],zs,tau))
            # alphars = float(alpha_s(self.mass_list[i],self.rs,self.redshift_list[i],zs)) # old result

            kwargs_nfw = {'Rs':rsang, 'alpha_Rs':alphars,
                          'r_trunc':tau*rsang,
                          'center_x': center_nfw_x, 'center_y': center_nfw_y}
            kwargs_unsorted.append(kwargs_nfw)
        #
        if any_mass_sheets: # (append negative convergence sheets)
            for i in range(N):
                if self.mass_sheets[i]:
                    convergence_sheet_masses[self.redshift_list[i]] += self.mass_list[i]
            for z, m in sorted(convergence_sheet_masses.items()):
                area_com = self.double_cone_width(z)**2 # kpc**2 comoving
                area = area_com / (1+z)**2 # kpc**2 physical
                sig = m/area # Msun / kpc**2

                
                # our normalization is the formula from assuming that this redshift
                # is the only lens
                sig_cr = sigma_cr(z, self.zs).to(u.Msun/u.Mpc**2).value / 1000**2 # from Msun/Mpc**2 to Msun/kpc**2

                kwargs_convergence_sheet = {'kappa': -sig/sig_cr} # todo check this calculation
                kwargs_unsorted.append(kwargs_convergence_sheet)

        self.kwargs_lens = [kwargs_unsorted[i] for i in sort_idx]
        
        ########################################################################
        # set up the list of light models to be used #

        # # SOURCE light
        # source_light_model_list = []
        # for i in range(len(self.source_params)):
        #     source_light_model_list.append('SERSIC_ELLIPSE')
        # for i in range(N_clump):
        #     source_light_model_list.append('SERSIC')

        # self.light_model_source = LightModel(light_model_list = source_light_model_list)

        # LENS light
        lens_light_model_list = ['SERSIC_ELLIPSE'] if is_lens_light else []
        self.light_model_lens = LightModel(light_model_list = lens_light_model_list) if is_lens_light else None

        # # SOURCE light kwargs

        # self.kwargs_source_mag = []
        # ## First we add all the main (elliptical) sources.
        # for mag, (r,phi,q,ra,dec,n) in zip(self.mag_source, self.source_params):
        #     e1s, e2s = param_util.phi_q2_ellipticity(phi=phi, q=q)
        #     self.kwargs_source_mag.append({'magnitude': mag,
        #                                    'R_sersic': r, 'n_sersic': n,
        #                                    'center_x': ra, 'center_y': dec,
        #                                    'e1': e1s, 'e2': e2s})
        # ## Then we auto-add any additional clumps.
        # assert(len(self.source_params) > 0)
        # beta_ra_source = self.source_params[0][3]
        # beta_dec_source = self.source_params[0][4]
        # for i in range(N_clump):
        #     self.kwargs_source_mag.append({'magnitude': self.mag_source_clumps, 'R_sersic': self.r_sersic_source_clumps[i],
        #                                    'n_sersic': self.n_sersic_source_clumps[i],
        #                                    'center_x': beta_ra_source+self.source_scatter*(clumprandx[i]-.5), 
        #                                    'center_y': beta_dec_source+self.source_scatter*(clumprandy[i]-.5)})

        # LENS light kwargs
        self.kwargs_lens_mag = ([{'magnitude': self.mag_lens, 'R_sersic': theta_lens, 'n_sersic': self.n_sersic_lens_light,
                                  'e1': e1, 'e2': e2, 'center_x': self.center_lens_x , 'center_y': self.center_lens_y}]
                                if is_lens_light else None)
        
        # ################################################################################
        # Setup data_class, i.e. pixelgrid #
        ra_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        dec_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # ''
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self.pixsize  # linear translation matrix of a shift in pixel in a shift in coordinates
        kwargs_pixel = {'nx': self.pixnum, 'ny': self.pixnum,  # number of pixels per axis
                        'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                        'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                        'transform_pix2angle': transform_pix2angle} 
        self.pixel_grid = PixelGrid(**kwargs_pixel)

        # # Setup PSF #
        # # (should not affect alpha calculations) #
        # kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
        #               'fwhm': 0.01,  # full width at half maximum of the Gaussian PSF (in angular units)
        #               'pixel_size': self.pixsize  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
        #              }
        # self.psf = PSF(**kwargs_psf)
        # kernel = self.psf.kernel_point_source

        
        # define the numerics #
        self.kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
                                'supersampling_convolution': False}

        # # initialize the Image model class by combining the modules we created above #
        # self.imageModel = ImageModel(data_class=self.pixel_grid, psf_class=self.psf,
        #                              lens_model_class=self.lens_model_mp,
        #                              source_model_class=self.source_model_list,
        #                              lens_light_model_class=self.light_model_lens,
        #                              kwargs_numerics=self.kwargs_numerics)

        # alternatively define kwargs_model (in case we want to generate the noisy image)
        self.kwargs_model = {'lens_model_list':lens_model_sorted,
                             'z_source':self.zs,
                             'lens_redshift_list':redshifts_sorted,
                             'source_light_model_list':self.source_model_list,
                             'lens_light_model_list': lens_light_model_list,
                             'cosmo':None} # cosmo could also be set to something else

class MultiShapeletImage(COSMOSImage):
    def __init__(self, xpos_list, ypos_list, redshift_list, m=None, zl=0.2,
                 zs=1.0, pixsize=0.2, pixnum=200, mass_sheets=None, main_theta=0.3,
                 q_lens=1, phi_lens=-0.9,
                 center_lens_x=0, center_lens_y=0,
                 is_lens_light=False,
                 n_sersic_lens_light=2,
                 gamma=2,
                 mag_lens=20,
                 source_nmaxes=[15, 15],
                 source_betas=[.11, .6],
                 source_coeffs=None,
                 source_centers=[(0,0), (0,0)],
                 rs=1e-4):

        # `mass_sheets` : default set to True, meaning we should add negative mass sheets to cancel out all the substructure

        assert(len(xpos_list) == len(ypos_list))
        assert(len(xpos_list) == len(redshift_list))

        self.xpos_list = xpos_list
        self.ypos_list = ypos_list
        self.redshift_list = redshift_list
        self.N = len(xpos_list) # number of interlopers+subhalos
        N = self.N
        self.zl = zl
        self.zs = zs
        if m is None:
            self.mass_list = [1e7] * self.N
        elif isinstance(m, float):
            self.mass_list = [m] * self.N
        else:
            self.mass_list = m

        self.pixsize = pixsize
        self.pixnum = pixnum

        if isinstance(mass_sheets, list) or isinstance(mass_sheets, np.ndarray):
            self.mass_sheets = mass_sheets
        elif mass_sheets is None or mass_sheets is True:
            self.mass_sheets = [True for _ in range(N)]
        elif mass_sheets is False:
            self.mass_sheets = [False for _ in range(N)]
        any_mass_sheets = np.any(self.mass_sheets)
            
        self.main_theta = main_theta
        self.q_lens = q_lens
        self.phi_lens = phi_lens
        self.center_lens_x = center_lens_x
        self.center_lens_y = center_lens_y

        self.n_sersic_lens_light = n_sersic_lens_light
        self.is_lens_light = is_lens_light

        self.gamma = gamma # lens spep gamma

        self.mag_lens = mag_lens

        assert(source_coeffs is not None)
        assert(len(source_coeffs) == sum((nmax + 1) * (nmax + 2) // 2 for nmax in source_nmaxes))
        self.source_nmaxes = source_nmaxes
        self.source_betas = source_betas
        self.source_coeffs = source_coeffs
        self.source_centers = source_centers

        self.rs = rs # scale radius of subhalo in Mpc (at 1e6 Msun pivot mass)

        ## SOURCE PROPERTIES ###############################################################################

        self.source_model_list = ['INTERPOL']

        source_pix_factor = 5
        source_pixsize = self.pixsize / source_pix_factor
        source_pixnum = round(self.pixnum * source_pix_factor)
        # ds = DiffShapelets(self.source_nmax, self.source_beta,
        #                    pixnum=source_pixnum, pixsize=source_pixsize)
        ms = MultiShapelets(self.source_nmaxes, self.source_betas,
                            pixnum=source_pixnum, pixsize=source_pixsize,
                            centers=self.source_centers)

        source_image = ms.reco_source(self.source_coeffs)

        self.source_kwargs_list = [{'image':source_image,
                                    'center_x':0, 'center_y':0,
                                    'phi_G':0,
                                    'scale':source_pixsize,
                                    'amp': 1/self.pixsize**2}]
        ####################################################################################################



        ## LENS PROPERTIES #################################################################################
        theta_lens = self.main_theta # used to be 10.
        r_theta_lens = x_to_xi(theta_lens,zl)
        e1, e2 = param_util.phi_q2_ellipticity(phi=self.phi_lens, q=self.q_lens) # used to be q=0.8
        # gamma = 2.

        # lens light properties
        # n_sersic_lens_light = 2.
        ####################################################################################################

        ## INTERLOPER PROPERTIES ########################################################################### 

        # for easier plotting only (current version only works when all the interlopers are at the lens redshift):
        # self.plot_xpixs = [self.x_to_pix(xpos) for xpos in xpos_list]
        # self.plot_ypixs = [self.y_to_pix(ypos), zl,pixsize,pixnum) for ypos in ypos_list]

        #beta_ra, beta_dec = beta_ras[0], beta_decs[0]

        # self.m = 1.0e7 # mass of interlopers (used to be 1e7, and then 1e9)
        # self.rs = 0.001  # interloper scale radius r_s
        # A = 80**2 ## in arcsec ## IGNORE THIS, THIS WAS FOR NEGATIVE CONVERGENCE
        # self.rs = 1e-4 # Mpc (pivot around m0=1e6)
        
        # kext = float(k_ext(N,m,A,zl,zs,pixsize))
        # note that there is no more self.rsang or self.alphars

        ## LENS model and redshifts
        # First we make a dictionary of convergence sheet masses
        convergence_sheet_masses = {z:0 for z in self.redshift_list}
        
        # In the unsorted list, we'll put the main lens first
        lens_model_unsorted = ['SPEP'] + ['TNFW' for i in range(N)] + (['CONVERGENCE' for _ in convergence_sheet_masses]
                                                                       if any_mass_sheets else [])
        redshifts_unsorted = [self.zl] + list(self.redshift_list) + (sorted(convergence_sheet_masses.keys())
                                                                     if any_mass_sheets else [])

        # Then we sort everything
        sort_idx = np.argsort(redshifts_unsorted)
        lens_model_sorted = [lens_model_unsorted[i] for i in sort_idx]
        redshifts_sorted = [redshifts_unsorted[i] for i in sort_idx]

        self.main_lens_idx = np.where(sort_idx == 0)[0][0] # which lens is the main lens?

        self.lens_model_mp = LensModel(lens_model_list=lens_model_sorted,
                                       z_source = self.zs,
                                       lens_redshift_list=redshifts_sorted,
                                       multi_plane=True)

        # LENS kwargs
        self.kwargs_spep = {'theta_E': theta_lens, 'e1': e1, 'e2': e2, 
                            'gamma': gamma, 'center_x': self.center_lens_x, 'center_y': self.center_lens_y}

        kwargs_unsorted = [self.kwargs_spep] # (+ will append more)
        #
        for i in range(N): # (append interlopers)
            center_nfw_x = xpos_list[i]
            center_nfw_y = ypos_list[i]

            tau = 20 # assume 20 as default

            rs_adjusted = self.rs * (self.mass_list[i]/1e6)**(1/3.) # adjusted according to physical mass
            
            rsang = float(rs_angle(self.redshift_list[i],rs_adjusted))
            alphars = float(alpha_s_tnfw(self.mass_list[i],rs_adjusted,self.redshift_list[i],zs,tau))
            # alphars = float(alpha_s(self.mass_list[i],self.rs,self.redshift_list[i],zs)) # old result

            kwargs_nfw = {'Rs':rsang, 'alpha_Rs':alphars,
                          'r_trunc':tau*rsang,
                          'center_x': center_nfw_x, 'center_y': center_nfw_y}
            kwargs_unsorted.append(kwargs_nfw)
        #
        if any_mass_sheets: # (append negative convergence sheets)
            for i in range(N):
                if self.mass_sheets[i]:
                    convergence_sheet_masses[self.redshift_list[i]] += self.mass_list[i]
            for z, m in sorted(convergence_sheet_masses.items()):
                area_com = self.double_cone_width(z)**2 # kpc**2 comoving
                area = area_com / (1+z)**2 # kpc**2 physical
                sig = m/area # Msun / kpc**2

                
                # our normalization is the formula from assuming that this redshift
                # is the only lens
                sig_cr = sigma_cr(z, self.zs).to(u.Msun/u.Mpc**2).value / 1000**2 # from Msun/Mpc**2 to Msun/kpc**2

                kwargs_convergence_sheet = {'kappa': -sig/sig_cr} # todo check this calculation
                kwargs_unsorted.append(kwargs_convergence_sheet)

        self.kwargs_lens = [kwargs_unsorted[i] for i in sort_idx]
        
        ########################################################################
        # set up the list of light models to be used #

        # # SOURCE light
        # source_light_model_list = []
        # for i in range(len(self.source_params)):
        #     source_light_model_list.append('SERSIC_ELLIPSE')
        # for i in range(N_clump):
        #     source_light_model_list.append('SERSIC')

        # self.light_model_source = LightModel(light_model_list = source_light_model_list)

        # LENS light
        lens_light_model_list = ['SERSIC_ELLIPSE'] if is_lens_light else []
        self.light_model_lens = LightModel(light_model_list = lens_light_model_list) if is_lens_light else None

        # # SOURCE light kwargs

        # self.kwargs_source_mag = []
        # ## First we add all the main (elliptical) sources.
        # for mag, (r,phi,q,ra,dec,n) in zip(self.mag_source, self.source_params):
        #     e1s, e2s = param_util.phi_q2_ellipticity(phi=phi, q=q)
        #     self.kwargs_source_mag.append({'magnitude': mag,
        #                                    'R_sersic': r, 'n_sersic': n,
        #                                    'center_x': ra, 'center_y': dec,
        #                                    'e1': e1s, 'e2': e2s})
        # ## Then we auto-add any additional clumps.
        # assert(len(self.source_params) > 0)
        # beta_ra_source = self.source_params[0][3]
        # beta_dec_source = self.source_params[0][4]
        # for i in range(N_clump):
        #     self.kwargs_source_mag.append({'magnitude': self.mag_source_clumps, 'R_sersic': self.r_sersic_source_clumps[i],
        #                                    'n_sersic': self.n_sersic_source_clumps[i],
        #                                    'center_x': beta_ra_source+self.source_scatter*(clumprandx[i]-.5), 
        #                                    'center_y': beta_dec_source+self.source_scatter*(clumprandy[i]-.5)})

        # LENS light kwargs
        self.kwargs_lens_mag = ([{'magnitude': self.mag_lens, 'R_sersic': theta_lens, 'n_sersic': self.n_sersic_lens_light,
                                  'e1': e1, 'e2': e2, 'center_x': self.center_lens_x , 'center_y': self.center_lens_y}]
                                if is_lens_light else None)
        
        # ################################################################################
        # Setup data_class, i.e. pixelgrid #
        ra_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        dec_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # ''
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self.pixsize  # linear translation matrix of a shift in pixel in a shift in coordinates
        kwargs_pixel = {'nx': self.pixnum, 'ny': self.pixnum,  # number of pixels per axis
                        'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                        'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                        'transform_pix2angle': transform_pix2angle} 
        self.pixel_grid = PixelGrid(**kwargs_pixel)

        # # Setup PSF #
        # # (should not affect alpha calculations) #
        # kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
        #               'fwhm': 0.01,  # full width at half maximum of the Gaussian PSF (in angular units)
        #               'pixel_size': self.pixsize  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
        #              }
        # self.psf = PSF(**kwargs_psf)
        # kernel = self.psf.kernel_point_source

        
        # define the numerics #
        self.kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
                                'supersampling_convolution': False}

        # # initialize the Image model class by combining the modules we created above #
        # self.imageModel = ImageModel(data_class=self.pixel_grid, psf_class=self.psf,
        #                              lens_model_class=self.lens_model_mp,
        #                              source_model_class=self.source_model_list,
        #                              lens_light_model_class=self.light_model_lens,
        #                              kwargs_numerics=self.kwargs_numerics)

        # alternatively define kwargs_model (in case we want to generate the noisy image)
        self.kwargs_model = {'lens_model_list':lens_model_sorted,
                             'z_source':self.zs,
                             'lens_redshift_list':redshifts_sorted,
                             'source_light_model_list':self.source_model_list,
                             'lens_light_model_list': lens_light_model_list,
                             'cosmo':None} # cosmo could also be set to something else


class PixSourceImage(COSMOSImage):
    def __init__(self, xpos_list, ypos_list, redshift_list, m=None, zl=0.2,
                 zs=1.0, pixsize=0.2, pixnum=200, mass_sheets=None, main_theta=0.3,
                 q_lens=1, phi_lens=-0.9,
                 center_lens_x=0, center_lens_y=0,
                 is_lens_light=False,
                 n_sersic_lens_light=2,
                 gamma=2,
                 mag_lens=20,
                 # source_nmaxes=[15, 15],
                 # source_betas=[.11, .6],
                 # source_coeffs=None,
                 # source_centers=[(0,0), (0,0)],
                 source_image=None,
                 source_pixsize=None,
                 rs=1e-4):

        # `mass_sheets` : default set to True, meaning we should add negative mass sheets to cancel out all the substructure

        assert(len(xpos_list) == len(ypos_list))
        assert(len(xpos_list) == len(redshift_list))

        self.xpos_list = xpos_list
        self.ypos_list = ypos_list
        self.redshift_list = redshift_list
        self.N = len(xpos_list) # number of interlopers+subhalos
        N = self.N
        self.zl = zl
        self.zs = zs
        if m is None:
            self.mass_list = [1e7] * self.N
        elif isinstance(m, float):
            self.mass_list = [m] * self.N
        else:
            self.mass_list = m

        self.pixsize = pixsize
        self.pixnum = pixnum

        if isinstance(mass_sheets, list) or isinstance(mass_sheets, np.ndarray):
            self.mass_sheets = mass_sheets
        elif mass_sheets is None or mass_sheets is True:
            self.mass_sheets = [True for _ in range(N)]
        elif mass_sheets is False:
            self.mass_sheets = [False for _ in range(N)]
        any_mass_sheets = np.any(self.mass_sheets)
            
        self.main_theta = main_theta
        self.q_lens = q_lens
        self.phi_lens = phi_lens
        self.center_lens_x = center_lens_x
        self.center_lens_y = center_lens_y

        self.n_sersic_lens_light = n_sersic_lens_light
        self.is_lens_light = is_lens_light

        self.gamma = gamma # lens spep gamma

        self.mag_lens = mag_lens

        assert(source_image is not None)
        self.source_image = source_image
        self.source_pixsize = (source_pixsize if source_pixsize is not None else
                               pixsize * pixnum / len(source_image))
        # That is, the default value is to assume the extent of the image and source is the same

        self.rs = rs # scale radius of subhalo in Mpc (at 1e6 Msun pivot mass)

        ## SOURCE PROPERTIES ###############################################################################

        self.source_model_list = ['INTERPOL']

        self.source_kwargs_list = [{'image':self.source_image,
                                    'center_x':0, 'center_y':0,
                                    'phi_G':0,
                                    'scale':self.source_pixsize,
                                    'amp': 1/self.pixsize**2}]
        ####################################################################################################



        ## LENS PROPERTIES #################################################################################
        theta_lens = self.main_theta # used to be 10.
        r_theta_lens = x_to_xi(theta_lens,zl)
        e1, e2 = param_util.phi_q2_ellipticity(phi=self.phi_lens, q=self.q_lens) # used to be q=0.8
        # gamma = 2.

        # lens light properties
        # n_sersic_lens_light = 2.
        ####################################################################################################

        ## INTERLOPER PROPERTIES ########################################################################### 

        # for easier plotting only (current version only works when all the interlopers are at the lens redshift):
        # self.plot_xpixs = [self.x_to_pix(xpos) for xpos in xpos_list]
        # self.plot_ypixs = [self.y_to_pix(ypos), zl,pixsize,pixnum) for ypos in ypos_list]

        #beta_ra, beta_dec = beta_ras[0], beta_decs[0]

        # self.m = 1.0e7 # mass of interlopers (used to be 1e7, and then 1e9)
        # self.rs = 0.001  # interloper scale radius r_s
        # A = 80**2 ## in arcsec ## IGNORE THIS, THIS WAS FOR NEGATIVE CONVERGENCE
        # self.rs = 1e-4 # Mpc (pivot around m0=1e6)
        
        # kext = float(k_ext(N,m,A,zl,zs,pixsize))
        # note that there is no more self.rsang or self.alphars

        ## LENS model and redshifts
        # First we make a dictionary of convergence sheet masses
        convergence_sheet_masses = {z:0 for z in self.redshift_list}
        
        # In the unsorted list, we'll put the main lens first
        lens_model_unsorted = ['SPEP'] + ['TNFW' for i in range(N)] + (['CONVERGENCE' for _ in convergence_sheet_masses]
                                                                       if any_mass_sheets else [])
        redshifts_unsorted = [self.zl] + list(self.redshift_list) + (sorted(convergence_sheet_masses.keys())
                                                                     if any_mass_sheets else [])

        # Then we sort everything
        sort_idx = np.argsort(redshifts_unsorted)
        lens_model_sorted = [lens_model_unsorted[i] for i in sort_idx]
        redshifts_sorted = [redshifts_unsorted[i] for i in sort_idx]

        self.main_lens_idx = np.where(sort_idx == 0)[0][0] # which lens is the main lens?

        self.lens_model_mp = LensModel(lens_model_list=lens_model_sorted,
                                       z_source = self.zs,
                                       lens_redshift_list=redshifts_sorted,
                                       multi_plane=True)

        # LENS kwargs
        self.kwargs_spep = {'theta_E': theta_lens, 'e1': e1, 'e2': e2, 
                            'gamma': gamma, 'center_x': self.center_lens_x, 'center_y': self.center_lens_y}

        kwargs_unsorted = [self.kwargs_spep] # (+ will append more)
        #
        for i in range(N): # (append interlopers)
            center_nfw_x = xpos_list[i]
            center_nfw_y = ypos_list[i]

            tau = 20 # assume 20 as default

            rs_adjusted = self.rs * (self.mass_list[i]/1e6)**(1/3.) # adjusted according to physical mass
            
            rsang = float(rs_angle(self.redshift_list[i],rs_adjusted))
            alphars = float(alpha_s_tnfw(self.mass_list[i],rs_adjusted,self.redshift_list[i],zs,tau))
            # alphars = float(alpha_s(self.mass_list[i],self.rs,self.redshift_list[i],zs)) # old result

            kwargs_nfw = {'Rs':rsang, 'alpha_Rs':alphars,
                          'r_trunc':tau*rsang,
                          'center_x': center_nfw_x, 'center_y': center_nfw_y}
            kwargs_unsorted.append(kwargs_nfw)
        #
        if any_mass_sheets: # (append negative convergence sheets)
            for i in range(N):
                if self.mass_sheets[i]:
                    convergence_sheet_masses[self.redshift_list[i]] += self.mass_list[i]
            for z, m in sorted(convergence_sheet_masses.items()):
                area_com = self.double_cone_width(z)**2 # kpc**2 comoving
                area = area_com / (1+z)**2 # kpc**2 physical
                sig = m/area # Msun / kpc**2

                
                # our normalization is the formula from assuming that this redshift
                # is the only lens
                sig_cr = sigma_cr(z, self.zs).to(u.Msun/u.Mpc**2).value / 1000**2 # from Msun/Mpc**2 to Msun/kpc**2

                kwargs_convergence_sheet = {'kappa': -sig/sig_cr} # todo check this calculation
                kwargs_unsorted.append(kwargs_convergence_sheet)

        self.kwargs_lens = [kwargs_unsorted[i] for i in sort_idx]
        
        ########################################################################
        # set up the list of light models to be used #

        # # SOURCE light
        # source_light_model_list = []
        # for i in range(len(self.source_params)):
        #     source_light_model_list.append('SERSIC_ELLIPSE')
        # for i in range(N_clump):
        #     source_light_model_list.append('SERSIC')

        # self.light_model_source = LightModel(light_model_list = source_light_model_list)

        # LENS light
        lens_light_model_list = ['SERSIC_ELLIPSE'] if is_lens_light else []
        self.light_model_lens = LightModel(light_model_list = lens_light_model_list) if is_lens_light else None

        # # SOURCE light kwargs

        # self.kwargs_source_mag = []
        # ## First we add all the main (elliptical) sources.
        # for mag, (r,phi,q,ra,dec,n) in zip(self.mag_source, self.source_params):
        #     e1s, e2s = param_util.phi_q2_ellipticity(phi=phi, q=q)
        #     self.kwargs_source_mag.append({'magnitude': mag,
        #                                    'R_sersic': r, 'n_sersic': n,
        #                                    'center_x': ra, 'center_y': dec,
        #                                    'e1': e1s, 'e2': e2s})
        # ## Then we auto-add any additional clumps.
        # assert(len(self.source_params) > 0)
        # beta_ra_source = self.source_params[0][3]
        # beta_dec_source = self.source_params[0][4]
        # for i in range(N_clump):
        #     self.kwargs_source_mag.append({'magnitude': self.mag_source_clumps, 'R_sersic': self.r_sersic_source_clumps[i],
        #                                    'n_sersic': self.n_sersic_source_clumps[i],
        #                                    'center_x': beta_ra_source+self.source_scatter*(clumprandx[i]-.5), 
        #                                    'center_y': beta_dec_source+self.source_scatter*(clumprandy[i]-.5)})

        # LENS light kwargs
        self.kwargs_lens_mag = ([{'magnitude': self.mag_lens, 'R_sersic': theta_lens, 'n_sersic': self.n_sersic_lens_light,
                                  'e1': e1, 'e2': e2, 'center_x': self.center_lens_x , 'center_y': self.center_lens_y}]
                                if is_lens_light else None)
        
        # ################################################################################
        # Setup data_class, i.e. pixelgrid #
        ra_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        dec_at_xy_0 = -0.5*self.pixnum*self.pixsize + 0.5*self.pixsize # ''
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self.pixsize  # linear translation matrix of a shift in pixel in a shift in coordinates
        kwargs_pixel = {'nx': self.pixnum, 'ny': self.pixnum,  # number of pixels per axis
                        'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                        'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                        'transform_pix2angle': transform_pix2angle} 
        self.pixel_grid = PixelGrid(**kwargs_pixel)

        # # Setup PSF #
        # # (should not affect alpha calculations) #
        # kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
        #               'fwhm': 0.01,  # full width at half maximum of the Gaussian PSF (in angular units)
        #               'pixel_size': self.pixsize  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
        #              }
        # self.psf = PSF(**kwargs_psf)
        # kernel = self.psf.kernel_point_source

        
        # define the numerics #
        self.kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction) 
                                'supersampling_convolution': False}

        # # initialize the Image model class by combining the modules we created above #
        # self.imageModel = ImageModel(data_class=self.pixel_grid, psf_class=self.psf,
        #                              lens_model_class=self.lens_model_mp,
        #                              source_model_class=self.source_model_list,
        #                              lens_light_model_class=self.light_model_lens,
        #                              kwargs_numerics=self.kwargs_numerics)

        # alternatively define kwargs_model (in case we want to generate the noisy image)
        self.kwargs_model = {'lens_model_list':lens_model_sorted,
                             'z_source':self.zs,
                             'lens_redshift_list':redshifts_sorted,
                             'source_light_model_list':self.source_model_list,
                             'lens_light_model_list': lens_light_model_list,
                             'cosmo':None} # cosmo could also be set to something else


    
class PoolResults:
    """
    Runs and stores results for pool-based function calls
    """
    def __init__(self, func, init_args_list):
        """
        args_list = [xpos_args, ypos_args, zds] (for example)
        """
        # function we'll be running and args we'll be running it on
        self.func = func
        self.args_list = [] # we will change this after running on init_args_list
        self.results = {}

        self.run(init_args_list)

        # # number of different kinds of arguments the function takes in (not counting id number as first arg)
        # self.n_args = len(self.all_args)

        # # number of fits we want
        # self.n_fits = len(self.all_args[0])
        # for i in range(self.all_args[1:]):
        #     assert(len(self.all_args[i]) == self.n_fits)


    
    def __repr__(self):
        return 'PoolResults'+self.results.__repr__()
    
    def callback(self, result):
        assert(len(result) == 2)
        # result[0] is an id number that was also the first argument of func
        # result[1] is the PSOFit object (or whatever we're actually interested in)
        self.results[result[0]] = result[1]
        
    def run(self, new_args_list):
        # TODO: preprocessing step to remove anything redundant from the args_list
        print('cpu count', cpu_count())
        
        if len(new_args_list) == 0: return
        
        nargs = len(new_args_list[0])
        for i in range(1,len(new_args_list)):
            assert(len(new_args_list[i]) == nargs) # check that all `args` are the same length
        
        with Pool() as pool:
            p_list = []
            for i, func_args in enumerate(new_args_list, start=len(self.args_list[0]) if len(self.args_list) > 0 else 0):
                p = pool.apply_async(self.func, args=func_args, callback=self.callback)
                p_list.append(p)
                
            for p in p_list: # I think this is redundant
                p.wait()

            for p in p_list:
                p.get()


        if len(self.args_list) == 0:
            self.args_list = [list(args) for args in new_args_list]
        else:
            assert(len(self.args_list) == len(new_args_list)) # should be true, because otherwise the function call would have failed
            for i in range(len(self.args_list)):
                self.args_list[i] += list(new_args_list[i])

    def get_results_list(self):
        N = max(self.results.keys()) + 1
        results_list = [None] * N
        for k,v in self.results.items():
            results_list[k] = v
        return results_list




################################################################################
# Mask functions

def isinmask(xpix, ypix, r, dr, pixsize, pixnum):
    # r is the einstein radius, and we take pixels within r +- dr
    # (sharp cutoff)
    npix = np.sqrt((xpix-pixnum/2)**2 + (ypix-pixnum/2)**2)
    pixdist = npix * pixsize
    return (r - dr < pixdist < r + dr)

def isinmask_smooth(xpix, ypix, r, dr, pixsize, pixnum):
    # r is the einstein radius, and we take pixels within r +- dr
    # gaussian smoothing
    npix = np.sqrt((xpix-pixnum/2)**2 + (ypix-pixnum/2)**2)
    pixdist = npix * pixsize
    return np.exp(-(pixdist-r)**2/(2*dr**2))

def make_mask(theta, width, ext, pixnum_plus_four):
    # makes a sharp mask
    pixnum = pixnum_plus_four - 4
    
    mymask = np.zeros((pixnum,pixnum))
    for xpix in range(pixnum):
        for ypix in range(pixnum):
            mymask[xpix, ypix] = 1 if isinmask(xpix, ypix, theta, width, 2*ext/pixnum, pixnum) else 0
            
    return mymask

def make_ctr_mask(radius, ext, pixnum_plus_four):
    # makes a sharp mask to take out the center

    pixnum = pixnum_plus_four - 4

    ctr_pix = pixnum / 2.

    pixsize = 2.*ext/pixnum

    mymask = np.ones((pixnum, pixnum))
    for xpix in range(pixnum):
        for ypix in range(pixnum):
            dist = pixsize * np.sqrt((xpix - ctr_pix)**2 + (ypix - ctr_pix)**2)
            if dist <= radius:
                mymask[xpix, ypix] = 0

    return mymask

################################################################################

def coord_image(x_raw, y_raw, pixnum, pixsize, pixrad=2):
    # Draws an otherwise blank image with a 2-pixel circle around the x,y location
    #
    # (Commented out code also marks the main lens as a separate class (as a generic circle in the center))

    image = np.zeros((pixnum, pixnum)) # 0 is Background class

    for xpix in range(pixnum):
        for ypix in range(pixnum):
            x = (xpix - pixnum/2) * pixsize
            y = (ypix - pixnum/2) * pixsize

            dist = np.sqrt((x-x_raw)**2 + (y-y_raw)**2)
            if dist <= pixrad * pixsize:
                image[xpix, ypix] = 1 # Perturber class
            # elif np.sqrt(x**2 + y**2) <= 7 * pixsize:
            #     image[xpix, ypix] = 2 # Main class

    return image.astype(int)

def coord_image_multisub(xraws, yraws, pixnum, pixsize, pixrad=2):
    """ Same as coord_image but takes in a list of xs and ys """

    image = np.zeros((pixnum, pixnum), dtype=int)

    for xpix in range(pixnum):
        for ypix in range(pixnum):
            x = (xpix - pixnum/2) * pixsize
            y = (ypix - pixnum/2) * pixsize

            dists = [np.sqrt((x-x_raw)**2 + (y-y_raw)**2)
                     for (x_raw, y_raw) in zip(xraws, yraws)]

            if min(dists) <= pixrad * pixsize:
                image[xpix, ypix] = 1 # Perturber class

    return image

def coord_image_multisub_bymass(xraws, yraws, masses, pixnum, pixsize, pixrad=2):
    """Instead of returning a map of ints, we will return a map of mass multipliers

    (in case of collision, sum the masses)"""

    image = np.zeros((pixnum, pixnum))

    for xraw, yraw, mass in zip(xraws, yraws, masses):
        x_discretized = int(xraw / pixsize + pixnum/2)
        y_discretized = int(yraw / pixsize + pixnum/2)

        ## make a slightly wider than necessary window just to be safe
        for xpix in range(x_discretized - pixrad - 2, x_discretized + pixrad + 2):
            if xpix < 0 or xpix >= pixnum:
                continue
            for ypix in range(y_discretized - pixrad - 2, y_discretized + pixrad + 2):
                if ypix < 0 or ypix >= pixnum:
                    continue

                x = (xpix - pixnum/2) * pixsize
                y = (ypix - pixnum/2) * pixsize
                mydist = np.sqrt((xraw - x)**2 + (yraw - y)**2)

                if mydist <= pixrad * pixsize:
                    # add this subhalo to our image
                    image[xpix, ypix] += mass
                    
    return image

################################################################################
# simple_div and simple_curl

def simple_div(x, y, pixsize=.08):
    #return (x[2:,1:-1] - x[:-2,1:-1])/(2*pixsize) + (y[1:-1,2:] - y[1:-1,:-2])/(2*pixsize)
    offx = (-1./12.)*(x[4:,2:-2] - x[:-4,2:-2])+(2./3.)*(x[3:-1,2:-2] - x[1:-3,2:-2])
    offy = (-1./12.)*(y[2:-2,4:] - y[2:-2,:-4])+(2./3.)*(y[2:-2,3:-1] - y[2:-2,1:-3])
    return (offx+offy)/pixsize

def simple_curl(x, y, pixsize=.08):
    #return (y[2:,1:-1] - y[:-2,1:-1])/(2*pixsize) - (x[1:-1,2:] - x[1:-1,:-2])/(2*pixsize)
    offx = (-1./12.)*(x[2:-2,4:] - x[2:-2,:-4])+(2./3.)*(x[2:-2,3:-1] - x[2:-2,1:-3])
    offy = (-1./12.)*(y[4:,2:-2] - y[:-4,2:-2])+(2./3.)*(y[3:-1,2:-2] - y[1:-3,2:-2])
    return (offy-offx)/pixsize

################################################################################

## Some helpers for the "realistic" multiple subhalo generation

def positive_poisson(lam):
    """Draws from Poisson, but have to try again if we draw a 0

    This is useful for the multisubhalo case, when we want to produce an example
    that falls within a particular mass bin.

    """
    if lam < 10:
        def probs(k):
            """Probability of any given k if 0 is not allowed"""
            if k == 0:
                return 0
            # Otherwise, we take the regular Poisson eq, divided by its value at k=0, namely e^-lambda
            return lam**k / np.math.factorial(k) / (np.exp(lam) - 1)

        rand = np.random.rand()
        totprob = 0
        k = 0
        while True:
            k += 1
            totprob += probs(k)
            if rand < totprob:
                return k
    else: # for large lambda
        while True:
            val = np.random.poisson(lam)
            if val != 0:
                return val

def total_mass(low_exp, high_exp):
    """Unnormalized total mass of subhalos within a certain mass range.

    The mass function here is simply n(m) = m^-1.9.
    
    The input should be given as the base-10 exponent, so for example, if we
    consider the mass range 10^8 - 10^10 solar masses, low_exp is 8 and high_exp
    is 10.

    """
    if low_exp - 1e-6 < high_exp <= low_exp:
        return 0
    assert high_exp > low_exp

    return (10**(0.1*high_exp) - 10**(0.1*low_exp)) / 0.1

def total_nr(low_exp, high_exp):
    """Unnormalized total number of subhalos within a certain mass range.

    Note this uses the same unnormalized mass function as `total_mass` above.

    """
    if low_exp - 1e-6 < high_exp <= low_exp:
        return 0
    assert high_exp > low_exp
    
    return (10**(-0.9*high_exp) - 10**(-0.9*low_exp)) / (-0.9)

def avg_mass(low_exp, high_exp):
    """Returns the average mass within a certain mass range.

    This quantity is agnostic of the normalization.

    """
    if np.abs(low_exp - high_exp) < 1e-6:
        # don't want to get numerical errors if these are practically identical
        return 10.**low_exp
    
    return total_mass(low_exp, high_exp) / total_nr(low_exp, high_exp)
