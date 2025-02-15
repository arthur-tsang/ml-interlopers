################################################################################
# diff_shapelets.py
# *6 Dec 2021*
#
# This file takes care of shapelet operations, but lets us do it fully within
# pytorch (so it should be fully differentiable).
#
################################################################################


import numpy as np
#import matplotlib.pyplot as plt
import torch

#from numpy.polynomial.hermite import hermval
from numpy.polynomial.hermite import herm2poly

# from astropy.modeling.functional_models import Sersic2D

def eval_poly(x, c, cutoff=None):
    """Evaluate a polynomial with coefficients c in from lowest to highest order at point (or tensor) x
    (for example: eval_poly(2, torch.tensor([1,2,1])) )"""

    if cutoff is not None:
        x[x > cutoff] = 0
        x[x < -cutoff] = 0
    poly_terms = torch.stack([c[i] * x**i for i in range(len(c))])
    result = torch.sum(poly_terms, axis=0)
    return torch.nan_to_num(result, posinf=0, neginf=0)

class DiffShapelets():
    def __init__(self, nmax, beta, alphas=None, center_x=0, center_y=0, pixnum=80, pixsize=.08, mask=None, cutoff=None):
        self.nmax = nmax
        self.beta = beta
        if alphas is None:
            self.alphas = torch.zeros((2, pixnum, pixnum))
        else:
            self.alphas = alphas
        self.center_x = center_x
        self.center_y = center_y
        self.pixnum = pixnum
        self.pixsize = pixsize
        if mask is None:
            self.mask = torch.ones((pixnum, pixnum), dtype=bool)
            self.maskpixnum = pixnum**2 # number of pixels covered by the mask
            self.maskbool = False
        else:
            self.mask = mask.bool()
            self.maskpixnum = torch.sum(mask)
            self.maskbool = True
        self.cutoff = cutoff

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._precalc()

    def _precalc(self):
        self.xy_grid = torch.zeros((self.maskpixnum, 2)).to(self.device)
        self.hermite_precalc = torch.zeros((self.maskpixnum, 2, self.nmax+1)).to(self.device)

        herm2poly_cache = []
        for n in range(self.nmax+1):
            herm = np.zeros(n+1)
            herm[n] = 1
            herm2poly_cache.append(herm2poly(herm))

        # list of x and y to do precalc on
        pix_idx = 0
        for i in range(self.pixnum):
            for j in range(self.pixnum):
                if self.maskbool and not self.mask[i,j]:
                    continue
                self.xy_grid[pix_idx, 0] = (i-self.pixnum/2+1/2)*self.pixsize - self.center_x - self.alphas[0,i,j]
                self.xy_grid[pix_idx, 1] = (j-self.pixnum/2+1/2)*self.pixsize - self.center_y - self.alphas[1,i,j]
                pix_idx += 1

        # precalculate hermite at each order at each point
        for herm_order in range(self.nmax+1):
            # prefactor = (self.beta * 2**herm_order * np.sqrt(np.pi) * np.math.factorial(herm_order))**-.5
            ## (using `log_prefactor` seems a bit more robust against numerical issues)

            ## Testing the alternate "old" version instead ##

            # beta = self.beta if isinstance(self.beta, torch.Tensor) else torch.tensor(self.beta).float()
            # prefactor = (self.beta * 2**herm_order * np.sqrt(np.pi) * np.math.factorial(herm_order))**-.5
            
            # self.hermite_precalc[:, 0, herm_order] = (eval_poly(self.xy_grid[:, 0]/beta, herm2poly_cache[herm_order], cutoff=self.cutoff)
            #                                           * torch.exp(-(self.xy_grid[:, 0]/beta)**2/2)
            #                                           * prefactor)
            # self.hermite_precalc[:, 1, herm_order] = (eval_poly(self.xy_grid[:, 1]/beta, herm2poly_cache[herm_order], cutoff=self.cutoff)
            #                                           * torch.exp(-(self.xy_grid[:, 1]/beta)**2/2)
            #                                           * prefactor)

            ## end code snippet ##


            ## The following code works and is the new version ##

            beta = self.beta if isinstance(self.beta, torch.Tensor) else torch.tensor(self.beta).float()
            log_prefactor = -1/2 * torch.log(beta) - herm_order/2 * np.log(2) - 1/4 * np.log(np.pi) - 1/2 * np.log(1. * np.math.factorial(herm_order))

            self.hermite_precalc[:, 0, herm_order] = (eval_poly(self.xy_grid[:, 0]/beta, herm2poly_cache[herm_order], cutoff=self.cutoff)
                                                      * torch.exp(-(self.xy_grid[:, 0]/beta)**2/2
                                                                  + log_prefactor))
            self.hermite_precalc[:, 1, herm_order] = (eval_poly(self.xy_grid[:, 1]/beta, herm2poly_cache[herm_order], cutoff=self.cutoff)
                                                      * torch.exp(-(self.xy_grid[:, 1]/beta)**2/2
                                                                  + log_prefactor))

            ## end code snippet ##


    def get_coeffs(self, image):
        image_flattened = torch.masked_select(image, self.mask) if self.maskbool else image.flatten()

        # A tensor where the rows are the shapelet order (in 2d) and the columns are the datapoints
        coeff_tens = torch.zeros(((self.nmax+1)*(self.nmax+2)//2, len(image_flattened))).to(self.device)

        # Slightly confusing way of looping through all the possible values of n1,n2 such that n1+n2 <= nmax.
        # We do it like this to match the lenstronomy code.
        for nsum in range(self.nmax + 1):
            for n2 in range(nsum + 1):
                n1 = nsum - n2
                n_index = nsum * (nsum + 1) // 2 + n2

                coeff_tens[n_index, :] = self.hermite_precalc[:,0,n1] * self.hermite_precalc[:,1,n2]

        return torch.linalg.inv(coeff_tens @ coeff_tens.T) @ coeff_tens @ image_flattened

    def reco_image_flat(self, coeffs):
        brightnesses = torch.zeros(self.maskpixnum).to(self.device)

        # loop through the shapelet orders
        for nsum in range(self.nmax + 1):
            for n2 in range(nsum + 1):
                n1 = nsum - n2
                n_index = nsum * (nsum + 1) // 2 + n2

                brightnesses += coeffs[n_index] * self.hermite_precalc[:,0,n1] * self.hermite_precalc[:,1,n2]

        return brightnesses

    def reco_image(self, coeffs):

        brightnesses_flat = self.reco_image_flat(coeffs)

        if self.maskbool:
            brightnesses_2d = torch.zeros((self.pixnum, self.pixnum))

            pix_idx = 0
            for i in range(self.pixnum):
                for j in range(self.pixnum):
                    if not self.mask[i,j]:
                        continue

                    brightnesses_2d[i,j] = brightnesses_flat[pix_idx]
                    pix_idx += 1

            return brightnesses_2d
        else:
            return brightnesses_flat.reshape((self.pixnum, self.pixnum))

    def reco_source(self, coeffs):
        # Reconstruct the source according to the coeffs, disregarding alpha
        # (Since this is for visualization purposes only, I will code this up as quickly as I can, even if this method is bad practice.)
        true_alphas = self.alphas
        true_mask_params = self.mask, self.maskpixnum, self.maskbool
        self.alphas = torch.zeros((2, self.pixnum, self.pixnum))
        self.mask = torch.ones((self.pixnum, self.pixnum), dtype=bool)
        self.maskpixnum = self.pixnum**2
        self.maskbool = False
        self._precalc()
        answer = self.reco_image(coeffs)
        self.alphas = true_alphas
        self.mask, self.maskpixnum, self.maskbool = true_mask_params
        self._precalc()
        return answer

class MultiShapelets:
    """This class allows us to define the source in terms of multiple sets of
    shapelets, with different betas (and nmaxes and centers)"""

    def __init__(self, nmaxes, betas, alphas=None, centers=None, pixnum=80, pixsize=.08, cutoff=None):
        self.num_shape_sets = len(nmaxes)
        assert(self.num_shape_sets == len(betas))
        assert(centers is None or self.num_shape_sets == len(centers))

        def torchify(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x).float()
        
        self.nmaxes = nmaxes.detach().numpy() if isinstance(nmaxes, torch.Tensor) else nmaxes
        self.betas = torchify(betas)
        
        if alphas is None:
            self.alphas = torch.zeros((2, pixnum, pixnum))
        else:
            self.alphas = (alphas if isinstance(alphas, torch.Tensor)
                           else torch.stack(alphas) if isinstance(alphas[0], torch.Tensor)
                           else torch.from_numpy(np.array(alphas)))
        
        if centers is None:
            self.centers = [(0, 0) for _ in range(self.num_shape_sets)]
        else:
            self.centers = centers

        self.pixnum = pixnum
        self.pixsize = pixsize
        self.cutoff = cutoff

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._precalc()

    def _precalc(self):
        self.xy_grid = torch.zeros((self.pixnum**2, 2)).to(self.device)

        ## Set up xy_grid ##
        # See the code in `DiffShapelets` to implement masks here. For now,
        # we'll do something simpler. Also note that the shapelet center offsets are no
        # longer included in `self.xy_grid` itself.
        xvals = torch.arange((-self.pixnum/2 + .5)*self.pixsize, (self.pixnum/2 + .5)*self.pixsize, self.pixsize)
        xgrid, ygrid = torch.meshgrid(xvals, xvals)
        self.xy_grid[:,0] = xgrid.flatten() - self.alphas[0].flatten()
        self.xy_grid[:,1] = ygrid.flatten() - self.alphas[1].flatten()

        #####

        maxnmax = np.max(self.nmaxes)
        self.hermite_precalc = torch.zeros((self.num_shape_sets, self.pixnum**2, 2, maxnmax+1)).to(self.device)

        herm2poly_cache = []
        for n in range(maxnmax+1):
            herm = np.zeros(n+1)
            herm[n] = 1
            herm2poly_cache.append(herm2poly(herm))

        # precalculate hermite at each order at each point
        for i, (nmax, beta, center) in enumerate(zip(self.nmaxes, self.betas, self.centers)):
            for herm_order in range(nmax+1):

                log_prefactor = (-1/2 * torch.log(beta) - herm_order/2 * np.log(2)
                                 - 1/4 * np.log(np.pi) - 1/2 * np.log(1. * np.math.factorial(herm_order)))

                x_scaled = (self.xy_grid[:,0] - center[0])/beta
                y_scaled = (self.xy_grid[:,1] - center[1])/beta

                self.hermite_precalc[i, :, 0, herm_order] = (eval_poly(x_scaled, herm2poly_cache[herm_order], cutoff=self.cutoff)
                                                             * torch.exp(-x_scaled**2/2
                                                                         + log_prefactor))
                self.hermite_precalc[i, :, 1, herm_order] = (eval_poly(y_scaled, herm2poly_cache[herm_order], cutoff=self.cutoff)
                                                             * torch.exp(-y_scaled**2/2
                                                                         + log_prefactor))

    def get_coeffs(self, image):
        image_flattened = image.flatten()

        n_param_list = [(nmax+1)*(nmax+2)//2 for nmax in self.nmaxes]
        n_param_tot = np.sum(n_param_list)

        # A tensor where the rows are the shapelet order (in 2d) and the columns are the datapoints
        coeff_tens = torch.zeros((n_param_tot, len(image_flattened))).to(self.device)

        n_offset = 0
        for i, nmax in enumerate(self.nmaxes):
            # Slightly confusing way of looping through all the possible values of n1,n2 such that n1+n2 <= nmax.
            # We do it like this to match the lenstronomy code.
            for nsum in range(nmax + 1):
                for n2 in range(nsum + 1):
                    n1 = nsum - n2
                    n_index = nsum * (nsum + 1) // 2 + n2

                    coeff_tens[n_offset+n_index, :] = self.hermite_precalc[i,:,0,n1] * self.hermite_precalc[i,:,1,n2]

            n_offset += (nmax+1)*(nmax+2)//2

        return torch.linalg.inv(coeff_tens @ coeff_tens.T) @ coeff_tens @ image_flattened        

    def reco_image_flat(self, coeffs):
        brightnesses = torch.zeros(self.pixnum**2).to(self.device)

        n_offset = 0
        for i, nmax in enumerate(self.nmaxes):
            # loop through the shapelet orders
            for nsum in range(nmax + 1):
                for n2 in range(nsum + 1):
                    n1 = nsum - n2
                    n_index = nsum * (nsum + 1) // 2 + n2

                    brightnesses += coeffs[n_offset+n_index] * self.hermite_precalc[i,:,0,n1] * self.hermite_precalc[i,:,1,n2]

            n_offset += (nmax+1)*(nmax+2)//2

        return brightnesses
        
    def reco_image(self, coeffs):
        return self.reco_image_flat(coeffs).view((self.pixnum, self.pixnum))


    def reco_source(self, coeffs):
        # Reconstruct the source according to the coeffs, disregarding alpha
        # (Since this is for visualization purposes only, I will code this up as quickly as I can, even if this method is bad practice.)
        true_alphas = self.alphas
        self.alphas = torch.zeros((2, self.pixnum, self.pixnum))
        self._precalc()
        answer = self.reco_image(coeffs)
        self.alphas = true_alphas
        self._precalc()
        return answer

class MultiShapeletsMasked:
    """This class allows us to define the source in terms of multiple sets of
    shapelets, with different betas (and nmaxes and centers).

    Unlike the previous `MultiShapelets`, this class will skip calculations for the dim pixels. """

    def __init__(self, nmaxes, betas, alphas=None, centers=None, pixnum=80, pixsize=.08, cutoff=None,
                 mask=None):
        self.num_shape_sets = len(nmaxes)
        assert(self.num_shape_sets == len(betas))
        assert(centers is None or self.num_shape_sets == len(centers))

        def torchify(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x).float()
        
        self.nmaxes = nmaxes.detach().numpy() if isinstance(nmaxes, torch.Tensor) else nmaxes
        self.betas = torchify(betas)
        
        if alphas is None:
            self.alphas = torch.zeros((2, pixnum, pixnum))
        else:
            self.alphas = (alphas if isinstance(alphas, torch.Tensor)
                           else torch.stack(alphas) if isinstance(alphas[0], torch.Tensor)
                           else torch.from_numpy(np.array(alphas)))
        
        if centers is None:
            self.centers = [(0, 0) for _ in range(self.num_shape_sets)]
        else:
            self.centers = centers

        self.pixnum = pixnum
        self.pixsize = pixsize
        ## Introducing a mask to save on calculation.
        if mask is None:
            self.mask = torch.ones((pixnum, pixnum), dtype=bool)
            self.maskpixnum = pixnum**2 # number of pixels covered by mask
            self.maskbool = False
        else:
            self.mask = mask.bool()
            self.maskpixnum = torch.sum(mask)
            self.maskbool = True
        ## Cutoff for evalpoly.
        self.cutoff = cutoff

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._precalc()

    def _precalc(self):
        self.xy_grid = torch.zeros((self.maskpixnum, 2)).to(self.device)

        maxnmax = np.max(self.nmaxes)
        self.hermite_precalc = torch.zeros((self.num_shape_sets, self.maskpixnum, 2, maxnmax+1)).to(self.device)

        herm2poly_cache = []
        for n in range(maxnmax+1):
            herm = np.zeros(n+1)
            herm[n] = 1
            herm2poly_cache.append(herm2poly(herm))

        # list of x and y to do precalc on
        pix_idx = 0
        for i in range(self.pixnum):
            for j in range(self.pixnum):
                if self.maskbool and not self.mask[i,j]:
                    continue
                self.xy_grid[pix_idx, 0] = (i-self.pixnum/2+1/2)*self.pixsize - self.alphas[0, i, j]
                self.xy_grid[pix_idx, 1] = (j-self.pixnum/2+1/2)*self.pixsize - self.alphas[1, i, j]
                pix_idx += 1

        # precalculate hermite at each order at each point
        for i, (nmax, beta, center) in enumerate(zip(self.nmaxes, self.betas, self.centers)):
            for herm_order in range(nmax+1):
                log_prefactor = (-1/2 * torch.log(beta) - herm_order/2 * np.log(2)
                                 - 1/4 * np.log(np.pi) - 1/2 * np.log(1. * np.math.factorial(herm_order)))
                x_scaled = (self.xy_grid[:,0] - center[0])/beta
                y_scaled = (self.xy_grid[:,1] - center[1])/beta

                self.hermite_precalc[i, :, 0, herm_order] = (eval_poly(x_scaled, herm2poly_cache[herm_order], cutoff=self.cutoff)
                                                             * torch.exp(-x_scaled**2/2
                                                                         + log_prefactor))

                self.hermite_precalc[i, :, 1, herm_order] = (eval_poly(y_scaled, herm2poly_cache[herm_order], cutoff=self.cutoff)
                                                             * torch.exp(-y_scaled**2/2
                                                                         + log_prefactor))

    def get_coeffs(self, image):
        image_flattened = torch.masked_select(image, self.mask) if self.maskbool else image.flatten()
        
        n_param_list = [(nmax+1)*(nmax+2)//2 for nmax in self.nmaxes]
        n_param_tot = np.sum(n_param_list)

        # A tensor where the rows are the shapelet order (in 2d) and the columns are the datapoints
        coeff_tens = torch.zeros((n_param_tot, len(image_flattened))).to(self.device)

        n_offset = 0
        for i, nmax in enumerate(self.nmaxes):
            # Slightly confusing way of looping through all the possible values of n1,n2 such that n1+n2 <= nmax.
            # We do it like this to match the lenstronomy code.
            for nsum in range(nmax + 1):
                for n2 in range(nsum + 1):
                    n1 = nsum - n2
                    n_index = nsum * (nsum + 1) // 2 + n2

                    coeff_tens[n_offset+n_index, :] = self.hermite_precalc[i,:,0,n1] * self.hermite_precalc[i,:,1,n2]

            n_offset += (nmax+1)*(nmax+2)//2

        return torch.linalg.inv(coeff_tens @ coeff_tens.T) @ coeff_tens @ image_flattened

    def reco_image_flat(self, coeffs):
        brightnesses = torch.zeros(self.maskpixnum).to(self.device)

        n_offset = 0
        for i, nmax in enumerate(self.nmaxes):
            # loop through the shapelet orders
            for nsum in range(nmax + 1):
                for n2 in range(nsum + 1):
                    n1 = nsum - n2
                    n_index = nsum * (nsum + 1) // 2 + n2

                    brightnesses += coeffs[n_offset+n_index] * self.hermite_precalc[i,:,0,n1] * self.hermite_precalc[i,:,1,n2]

            n_offset += (nmax+1)*(nmax+2)//2

        return brightnesses
        
    def reco_image(self, coeffs):
        brightnesses_flat = self.reco_image_flat(coeffs)

        if self.maskbool:
            brightnesses_2d = torch.zeros((self.pixnum, self.pixnum))

            pix_idx = 0
            for i in range(self.pixnum):
                for j in range(self.pixnum):
                    if not self.mask[i,j]:
                        continue

                    brightnesses_2d[i,j] = brightnesses_flat[pix_idx]
                    pix_idx += 1

            return brightnesses_2d
        else:
            return brightnesses_flat.reshape((self.pixnum, self.pixnum))

    def reco_source(self, coeffs):
        # Reconstruct the source according to the coeffs, disregarding alpha
        # (Since this is for visualization purposes only, I will code this up as quickly as I can, even if this method is bad practice.)
        true_alphas = self.alphas
        true_mask_params = self.mask, self.maskpixnum, self.maskbool
        self.alphas = torch.zeros((2, self.pixnum, self.pixnum))
        self.mask = torch.ones((self.pixnum, self.pixnum), dtype=bool)
        self.maskpixnum = self.pixnum**2
        self.maskbool = False
        self._precalc()
        answer = self.reco_image(coeffs)
        self.alphas = true_alphas
        self.mask, self.maskpixnum, self.maskbool = true_mask_params
        self._precalc()
        return answer


class DiffShapeletsBatched:
    def __init__(self, nmax, beta, alphas=None, center_x=0, center_y=0, pixnum=80, pixsize=.08, mask=None, cutoff=None, batch_size=1):
        """Note that alphas should be indexed first by batch, then by (x,y), then by row and column"""

        self.nmax = nmax
        self.beta = beta
        if alphas is None:
            self.alphas = torch.zeros((batch_size, 2, pixnum, pixnum))
        else:
            assert(len(alphas) == batch_size)
            self.alphas = alphas
        self.center_x = center_x
        self.center_y = center_y
        self.pixnum = pixnum
        self.pixsize = pixsize
        if mask is None:
            self.mask = torch.ones((batch_size, pixnum, pixnum), dtype=bool)
            self.maskpixnum = pixnum**2 # number of pixels covered by the mask
            self.maskbool = False
        else:
            assert(len(mask) == batch_size)
            self.mask = mask.bool()
            self.maskpixnum = torch.sum(mask)
            self.maskbool = True
        self.cutoff = cutoff
        self.batch_size = batch_size

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._precalc()

    def _precalc(self):
        self.xy_grid = torch.zeros((self.batch_size, self.maskpixnum, 2)).to(self.device)
        self.hermite_precalc = torch.zeros((self.batch_size, self.maskpixnum, 2, self.nmax+1)).to(self.device)

        herm2poly_cache = []
        for n in range(self.nmax+1):
            herm = np.zeros(n+1)
            herm[n] = 1
            herm2poly_cache.append(herm2poly(herm))

        # list of x and y to do precalc on
        pix_idx = 0
        for i in range(self.pixnum):
            for j in range(self.pixnum):
                if self.maskbool and not self.mask[i,j]:
                    continue

                self.xy_grid[:, pix_idx, 0] = (i-self.pixnum/2+1/2)*self.pixsize + self.center_x - self.alphas[:,0,i,j]
                self.xy_grid[:, pix_idx, 1] = (j-self.pixnum/2+1/2)*self.pixsize + self.center_y - self.alphas[:,1,i,j]
                pix_idx += 1

        # precalculate hermite at each order at each point
        for herm_order in range(self.nmax+1):
            # todo: make sure this prefactor is computed as a double so we don't
            # get overflow errors (although torch likes floats, so I don't think
            # this is a good idea either)
            prefactor = (self.beta * 2**herm_order * np.sqrt(np.pi) * np.math.factorial(herm_order))**-.5
            self.hermite_precalc[:, :, 0, herm_order] = (eval_poly(self.xy_grid[:, :, 0]/self.beta, herm2poly_cache[herm_order], cutoff=self.cutoff)
                                                         * torch.exp(-(self.xy_grid[:, :, 0]/self.beta)**2/2)
                                                         * prefactor) # todo add back!!!
            self.hermite_precalc[:, :, 1, herm_order] = (eval_poly(self.xy_grid[:, :, 1]/self.beta, herm2poly_cache[herm_order], cutoff=self.cutoff)
                                                         * torch.exp(-(self.xy_grid[:, :, 1]/self.beta)**2/2)
                                                         * prefactor)


    def get_coeffs(self, image):
        assert(len(image) == self.batch_size)

        image_flattened = torch.masked_select(image, self.mask) if self.maskbool else image.flatten(start_dim=1)
        image_flattened = image_flattened[:,:,None] # add extra dimension to not confuse matrix multiplication at the end
        # TODO: check it the shape is [batch_size, 6400, 1] even if using a mask

        # `coeff_tens` : a tensor where the rows are the shapelet order (in 2d) and the columns are the datapoints
        ntot = (self.nmax+1)*(self.nmax+2)//2 # total number of shapelet coefficients required for self.nmax
        coeff_tens = torch.zeros((self.batch_size, ntot, len(image_flattened[0]))).to(self.device)

        # Slightly confusing way of looping through all the possible values of n1,n2 such that n1+n2 <= nmax.
        # We do it like this to match the lenstronomy code.
        for nsum in range(self.nmax + 1):
            for n2 in range(nsum + 1):
                n1 = nsum - n2
                n_index = nsum * (nsum + 1) // 2 + n2

                coeff_tens[:, n_index, :] = self.hermite_precalc[:,:,0,n1] * self.hermite_precalc[:,:,1,n2]


        coeffs = torch.linalg.inv(coeff_tens @ torch.transpose(coeff_tens, -2, -1)) @ coeff_tens @ image_flattened
        return coeffs.view(self.batch_size, ntot) # remove the extra dimension we added to not confuse the matrix multiplication

    def reco_image_flat(self, coeffs):
        brightnesses = torch.zeros(self.batch_size, self.maskpixnum).to(self.device)

        # loop through the shapelet orders
        for nsum in range(self.nmax + 1):
            for n2 in range(nsum + 1):
                n1 = nsum - n2
                n_index = nsum * (nsum + 1) // 2 + n2

                brightnesses += coeffs[:, n_index, None] * self.hermite_precalc[:,:,0,n1] * self.hermite_precalc[:,:,1,n2]

        return brightnesses

    def reco_image(self, coeffs):
        brightnesses_flat = self.reco_image_flat(coeffs)

        if self.maskbool:
            # TODO test that this code (for using the mask) works

            brightnesses_2d = torch.zeros((self.batch_size, self.pixnum, self.pixnum))

            pix_idx = 0
            for i in range(self.pixnum):
                for j in range(self.pixnum):
                    brightnesses_2d[:, i,j] = brightnesses_flat[:, pix_idx] * self.mask[:,i,j]
                    pix_idx += 1

            return brightnesses_2d
        else:
            return brightnesses_flat.reshape((self.batch_size, 1, self.pixnum, self.pixnum))

    def reco_source(self, coeffs):
        # Reconstruct the source according to the coeffs, disregarding alpha
        # (Since this is for visualization purposes only, I will code this up as quickly as I can, even if this method is bad practice.)
        true_alphas = self.alphas
        true_mask_params = self.mask, self.maskpixnum, self.maskbool
        self.alphas = torch.zeros((self.batch_size, 2, self.pixnum, self.pixnum))
        self.mask = torch.ones((self.batch_size, self.pixnum, self.pixnum), dtype=bool)
        self.maskpixnum = self.pixnum**2
        self.maskbool = False
        self._precalc()
        answer = self.reco_image(coeffs)
        self.alphas = true_alphas
        self.mask, self.maskpixnum, self.maskbool = true_mask_params
        self._precalc()
        return answer

def sersic(x, y, amplitude, r_eff, n, x_0, y_0, q, theta):
    """Sersic2D evaluation function from `astropy` rewritten for torch inputs.

    Actually, we made two changes to be more consistent with lenstronomy:

    1. Using `q` instead of `ellip`. `q` = 1 - `ellip`.

    2. Using the linear approximation of $b_n$ instead of the exact solution by
    solving the incomplete gamma function equation

    """

    bn = 1.9992 * n - 0.3271
    # (It's possible to get `bn` more accurate, but I want to keep consistent
    # with lenstronomy's implementation.)

    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta)

    # a, b = r_eff, q * r_eff
    a, b = q**-0.5 * r_eff, q**0.5 * r_eff ## Trying this to hopefully match lenstronomy
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    x_maj = (x-x_0) * cos_theta + (y-y_0) * sin_theta
    x_min = -(x-x_0) * sin_theta + (y-y_0) * cos_theta
    z = ((x_maj / a)**2 + (x_min/b)**2)**.5
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z)

    return amplitude * torch.exp(-bn * (z ** (1/n) - 1))

class DummySersic():

    def __init__(self, amplitude, r_eff, n, x_0, y_0, q, theta, alphas=None,
                 pixnum=80, pixsize=.08):
        self.sersic_params = [amplitude, r_eff, n, y_0, x_0, q, np.pi/2-theta] # have to transpose to match lenstronomy
        if alphas is not None:
            self.alphas = alphas
        else:
            self.alphas = torch.zeros((2, pixnum, pixnum))

        self.pixnum = pixnum
        self.pixsize = pixsize

        # self.sersic = Sersic2D()#amplitude, r_eff, n, x_0, y_0, ellip, theta)

    def reco_image(self, pix_offset=0):
        # The `pix_offset` variable is for debugging why the lenstronomy image
        # does not quite match the sersic image, even in the case of no noise.

        image = torch.zeros((self.pixnum, self.pixnum))

        for i in range(self.pixnum):
            for j in range(self.pixnum):
                x = (i-self.pixnum/2 + pix_offset)*self.pixsize - self.alphas[0,i,j]
                y = (j-self.pixnum/2 + pix_offset)*self.pixsize - self.alphas[1,i,j]

                # if i == 0:
                #     print('y', y)
                #     print('y (before alpha)', (j-self.pixnum/2 + pix_offset)*self.pixsize)

                image[i,j] = sersic(x, y, *self.sersic_params)

        return image

    def reco_source(self):

        image = torch.zeros((self.pixnum, self.pixnum))

        for i in range(self.pixnum):
            for j in range(self.pixnum):
                x = (i-self.pixnum/2)*self.pixsize
                y = (j-self.pixnum/2)*self.pixsize

                image[i,j] = sersic(x, y, *self.sersic_params)

        return image

    def set_amplitude(self, new_amplitude):
        self.sersic_params[0] = new_amplitude


def lenstro_spep(x_unshifted, y_unshifted, theta_E, gamma, phi, q, center_x, center_y):
    """lenstronomy-compatible spep function, rewritten in torch"""
    ## Just some type-related stuff ##
    if isinstance(phi, torch.Tensor):
        sin, cos = torch.sin, torch.cos
    else:
        sin, cos = np.sin, np.cos

    phi = np.pi/2 - phi # for compatible definitions with lenstronomy
    x, y = x_unshifted-center_y, y_unshifted-center_x # again, for compatible definitions with lenstronomy
    #E = theta_E * q / (((3-gamma)/2)**(1/(1-gamma)) * q**0.5)
    x_rot = cos(phi) * x + sin(phi) * y
    y_rot = -sin(phi)* x + cos(phi) * y
    if isinstance(phi, torch.Tensor):
        Psq = torch.maximum(x_rot**2 + y_rot**2 / q**2, torch.tensor(1e-6)) # elementwise max, kind of
    else:
        Psq = np.maximum(x_rot**2 + y_rot**2 / q**2, 1e-6)

    #fac = 2/(-gamma + 3) * (Psq / E**2)**((-gamma+1)/2)
    fac = (theta_E**2 / Psq * abs(q))**((gamma-1)/2)
    fx = cos(phi) * fac * x_rot - sin(phi) * fac * y_rot / q**2
    fy = sin(phi) * fac * x_rot + cos(phi) * fac * y_rot / q**2
    return fx, fy

def lenstro_spep_shear(x_unshifted, y_unshifted, theta_E, gamma, phi_lens, q, center_x, center_y, shear1, shear2, a3, phi3, a4, phi4):

    fx_spep, fy_spep = lenstro_spep(x_unshifted, y_unshifted, theta_E, gamma, phi_lens, q, center_x, center_y)

    fx_shear = -shear1 * x_unshifted + shear2 * y_unshifted
    fy_shear =  shear2 * x_unshifted + shear1 * y_unshifted

    if isinstance(phi3, torch.Tensor):
        sin, cos = torch.sin, torch.cos
        arctan2 = torch.atan2
    else:
        sin, cos = np.sin, np.cos
        arctan2 = np.arctan2

    ## Sorry, the following looks weird because we had to switch some x's and y's for compatibility.
    phi = arctan2(x_unshifted - center_y, y_unshifted - center_x)
    m = 3
    fy3 = cos(phi) * a3 / (1-m**2) * cos(m*(phi-phi3)) + sin(phi) * m * a3 / (1-m**2) * sin(m*(phi-phi3))
    fx3 = sin(phi) * a3 / (1-m**2) * cos(m*(phi-phi3)) - cos(phi) * m * a3 / (1-m**2) * sin(m*(phi-phi3))
    m = 4
    fy4 = cos(phi) * a4 / (1-m**2) * cos(m*(phi-phi4)) + sin(phi) * m * a4 / (1-m**2) * sin(m*(phi-phi4))
    fx4 = sin(phi) * a4 / (1-m**2) * cos(m*(phi-phi4)) - cos(phi) * m * a4 / (1-m**2) * sin(m*(phi-phi4))

    fx = fx_spep + fx_shear + fx3 + fx4
    fy = fy_spep + fy_shear + fy3 + fy4

    return fx, fy

## Convenience functions:

def spep_to_alpha(xgrid, ygrid, params):
    assert(len(params) == 6)
    malphax_flat, malphay_flat = lenstro_spep(xgrid.flatten(), ygrid.flatten(), *params)
    return malphax_flat.view(xgrid.shape), malphay_flat.view(ygrid.shape)

def spep_shear_to_alpha(xgrid, ygrid, params):
    assert(len(params) == 12)
    malphax_flat, malphay_flat = lenstro_spep_shear(xgrid.flatten(), ygrid.flatten(), *params)
    return malphax_flat.view(xgrid.shape), malphay_flat.view(ygrid.shape)
