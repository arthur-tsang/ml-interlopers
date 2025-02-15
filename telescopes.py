""" More telescope specs, mostly based on HST specs. """

import lenstronomy.Util.util as util

# __all__ = ['HSTELT', 'SEVENTYNINE']

"""
See https://www.aanda.org/articles/aa/pdf/2011/07/aa16603-11.pdf for some parameters.
"""

my_filter = {'exposure_time': 5400,
             'sky_brightness': 22.3,
             'magnitude_zero_point': 25.96,
             'num_exposures':1,
             'seeing':.010, # from https://www.mpe.mpg.de/391311/Davies10_SPIE7735.pdf
             'psf_type': 'GAUSSIAN'
}

class HSTELT(object):
    """
    based off of HST, but with ELT-like resolution
    """

    def __init__(self, band='my_filter', psf_type='GAUSSIAN', coadd_years=None):
        if band == 'my_filter':
            self.obs = my_filter
        else:
            raise ValueError("band %s is not supported. Choose 'my_filter'." % band)

        if psf_type == 'GAUSSIAN':
            self.obs['psf_type'] = 'GAUSSIAN'
        elif psf_type != 'PIXEL':
            raise ValueError("psf_type %s not supported!" % psf_type)

        if coadd_years is not None:
            raise ValueError(" %s coadd_years not supported!")

        # (imagined) camera settings
        self.camera = {'read_noise': 5, # from https://www.aanda.org/articles/aa/pdf/2011/07/aa16603-11.pdf Table 1
                       'pixel_scale': .020, # .004 or .0015 (.003 according to https://www.aanda.org/articles/aa/pdf/2011/07/aa16603-11.pdf Table 1) -- I'm "chunking" now just to see if we can perceive a difference.
                       'ccd_gain': 2.5,
        }

    def kwargs_single_band(self):
        """

        :return: merged kwargs from camera and obs dicts
        """
        kwargs = util.merge_dicts(self.camera, self.obs)
        return kwargs

class HSTELT2(HSTELT):
    def __init__(self, band='my_filter', psf_type='GAUSSIAN', coadd_years=None):

        super().__init__()

        self.camera = {'read_noise': 5, # from https://www.aanda.org/articles/aa/pdf/2011/07/aa16603-11.pdf Table 1
                       'pixel_scale': .004, # medium pixel size https://www.mpe.mpg.de/391311/Davies10_SPIE7735.pdf
                       'ccd_gain': 2.5,
        }

class HSTELT3(HSTELT):
    def __init__(self, band='my_filter', psf_type='GAUSSIAN', coadd_years=None):

        super().__init__()

        self.camera = {'read_noise': 5, # from https://www.aanda.org/articles/aa/pdf/2011/07/aa16603-11.pdf Table 1
                       'pixel_scale': .01, # set to FWHM size (real pixel size is smaller, but would take a week just to generate data...)
                       'ccd_gain': 2.5,
        }


        
class SEVENTYNINE(HSTELT):
    def __init__(self, band='my_filter', psf_type='GAUSSIAN', coadd_years=None):

        super().__init__()

        self.camera = {'read_noise': 5, # from https://www.aanda.org/articles/aa/pdf/2011/07/aa16603-11.pdf Table 1
                       'pixel_scale': .0633,
                       'ccd_gain': 2.5,
        }
