##########################################################################
# This document contains all the functions necessary to perform the profile
# modelling for the SBF analysis. 
##########################################################################

import numpy as np

# photutils functions: required to perform the ellipse analysis
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model
from photutils.aperture import EllipticalAperture

# mgefit find_galaxy: To find the peak & orientation of a galaxy in an image
from mgefit.find_galaxy import find_galaxy

# In order to assess computation time
from time import time


# -----------------------------------------------------------------------------------------

def obtainEllipseInstance(data, sma_normfactor=1):
    """
    Obtain the ellipse instance with the required geometry, as
    necessary for the elliptical isophote fit.
    
    sma_normfactor could be required in case of non-IR data, 
    isolist will not always fit otherwise.
    """
    x_size, y_size = data.shape
    
    # use the find_galaxy procedure from mge to find the galaxy orientation
    f = find_galaxy(np.ma.masked_array(data, np.isnan(data)), plot=False, quiet=True)
    
    geometry = EllipseGeometry(x0=f.ypeak, y0=f.xpeak, 
                               sma=f.majoraxis/sma_normfactor, eps=f.eps, 
                               pa=f.pa*np.pi/180, astep=0.1)
    
    ellipse = Ellipse(data, geometry)
    
    return ellipse


def obtainIsolistInstance(ellipse, isolist_type=None, nclip_sm=0, **kwargs):
    """ 
    This function fits the ellipse instance to the data. Returned is
    an object of the isolist class.
    
    Depending on the "isolist_type" parameter, a specific parameter choice
    is made for the fit.    
    
    isolist_type can either be
    "background":   The isolist is fitted using a medium step size,
                    relatively high fflag, and includes sigma clipping
    "star_masking": Step size is relatively large here, also with large 
                    fflag in order to fit the "full" image. Sigma clipping
                    performed depending on the value of nclip_sm.    
    "full":         The full fit is made, assuming a masked instance, full
                    fflag, small step size, no sigma clipping.
    "None":         The isolist is fit following the **kwargs params
    """
    if isolist_type == "background":
        isolist = ellipse.fit_image(nclip=1, minsma=10, step=0.3, fflag=0.6)
        if len(isolist)==0:
            isolist = ellipse.fit_image(nclip=2, minsma=10, step=0.3, fflag=0.6)
        if len(isolist)==0:
            isolist = ellipse.fit_image(nclip=3, minsma=10, step=0.3, fflag=0.6)
        
    if isolist_type == "star_masking":
        while nclip_sm <= 3:
            # values for fflag, maxgerr, step have been determined experimentally
            isolist = ellipse.fit_image(nclip=nclip_sm, step=0.3, 
                                        fflag=0.35, maxgerr=0.85)
            if len(isolist)!=0:
                break
            else: 
                nclip_sm += 1
                
    if isolist_type == "full":
        isolist = ellipse.fit_image(fflag=0.35, maxgerr=0.85)
           
    if isolist_type == None:
        isolist = ellipse.fit_image(**kwargs)
    
    return isolist
   
    

def fitEllipseModel(data, model_type=None, 
                    nclip_sm=0, # indicating sigma clipping in star masking
                    range_outward=0, high_harmonics=True, 
                    gridspacing=0.1, **kwargs):
    """
    Function that fits an isophotal model to a galaxy and returns the model in 
    the same frame of the original dataset.
    
    Important here is that the model returns the actual image.
    
    model_type is a parameter that indicates which parameters should be used to
    fit the model options are:
    "star_masking": Step size is relatively large, computation time is wanted as
                    small as possible in order to run a number of iterations.
    "full":         The full fit is made, assuming a masked instance, full
                    fflag, small step size, no sigma clipping.
    None:           The isolist is fit follwing the **kwargs arguments. and the 
                    range_outward, high_harmonics and gridspacing as given when 
                    calling the funcion.
    """
    ellipse = obtainEllipseInstance(data)
    isolist = obtainIsolistInstance(ellipse, isolist_type=model_type, 
                                    nclip_sm=nclip_sm, **kwargs)
    
    if len(isolist) == 0: # if no proper model could be made try with diff. sma
        ellipse = obtainEllipseInstance(data, sma_normfactor=10)
        isolist = obtainIsolistInstance(ellipse, isolist_type=model_type, 
                                    nclip_sm=nclip_sm, **kwargs)
        
    if model_type == "star_masking":
        try:
            model_image = buildEllipseModel(data.shape, isolist, range_outward=500, 
                                        high_harmonics=False, gridspacing=0.8)
        except:
            # try ellipse model with background ellipse instances
            isolist = obtainIsolistInstance(ellipse, isolist_type="background", 
                                    nclip_sm=0, **kwargs)
            model_image = buildEllipseModel(data.shape, isolist, range_outward=500, 
                                        high_harmonics=False, gridspacing=0.8)
    if model_type == "full":
        model_image = buildEllipseModel(data.shape, isolist, range_outward=500, 
                                        high_harmonics=True)
    if model_type == None:
        model_image = buildEllipseModel(data.shape, isolist, 
                                        range_outward = range_outward, 
                                        high_harmonics = high_harmonics, 
                                        gridspacing = gridspacing)
    return model_image


# -----------------------------------------------------------------------------------------
# Functions used in the pipeline


def fitInitialEllipseModel(data, mask_cr=None):
    """
    Fits basic ellipse model, returns that model with the residual
    """
    if np.any(mask_cr == None):
        mask_cr = np.zeros(data.shape, dtype=bool)
    
    masked_data = np.ma.masked_array(data, mask_cr)
        
    model_basic = fitEllipseModel(masked_data, model_type="star_masking", nclip_sm=2)
    
    residual_basic = data - model_basic
    return residual_basic, model_basic

def fitFinalEllipseModel(data, source_mask, center_sources, mask_cr=None):
    """
    Fits final ellipse model, returns that model with the residual
    """
    if np.any(mask_cr == None):
        mask_cr = np.zeros(data.shape, dtype=bool)
    
    mask_combined = source_mask | mask_cr
    masked_data = np.ma.masked_array(data, mask_combined & ~center_sources)
    model_final = fitEllipseModel(masked_data, model_type="star_masking", nclip_sm=0)
    residual_final = data - model_final
    return residual_final, model_final

# -----------------------------------------------------------------------------------------

# The function below is taken from the photutils package and has been slightly adapted 
# in order to include the possibility to model a galaxy for which the largest elliptical
# isophote covers a larger area than the original frame size.

    
def buildEllipseModel(shape, isolist, fill=0., high_harmonics=False, 
                      range_outward=0, gridspacing=0.1):
    """
    __Self-adjusted function from the photutils package__
    __Allows to genarate elliptical apertures outward of the image shape__
    
    Build a model elliptical galaxy image from a list of isophotes.

    For each ellipse in the input isophote list the algorithm fills the
    output image array with the corresponding isophotal intensity.
    Pixels in the output array are in general only partially covered by
    the isophote "pixel".  The algorithm takes care of this partial
    pixel coverage by keeping track of how much intensity was added to
    each pixel by storing the partial area information in an auxiliary
    array.  The information in this array is then used to normalize the
    pixel intensities.

    Parameters
    ----------
    shape : 2-tuple
        The (ny, nx) shape of the array used to generate the input
        ``isolist``.

    isolist : `~photutils.isophote.IsophoteList` instance
        The isophote list created by the `~photutils.isophote.Ellipse`
        class.

    fill : float, optional
        The constant value to fill empty pixels. If an output pixel has
        no contribution from any isophote, it will be assigned this
        value.  The default is 0.

    high_harmonics : bool, optional
        Whether to add the higher-order harmonics (i.e., ``a3``, ``b3``,
        ``a4``, and ``b4``; see `~photutils.isophote.Isophote` for
        details) to the result.
        
    range_outward: int, optional
        The range outward to which the model should be generated. The 
        returned image is still of size "shape".

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The image with the model galaxy.
    """
    from scipy.interpolate import LSQUnivariateSpline

    # the target grid is spaced in 0.1 pixel intervals so as
    # to ensure no gaps will result on the output array.
    finely_spaced_sma = np.arange(isolist[0].sma, isolist[-1].sma, gridspacing)

    # interpolate ellipse parameters

    # End points must be discarded, but how many?
    # This seems to work so far
    nodes = isolist.sma[2:-2]

    intens_array = LSQUnivariateSpline(
        isolist.sma, isolist.intens, nodes)(finely_spaced_sma)
    eps_array = LSQUnivariateSpline(
        isolist.sma, isolist.eps, nodes)(finely_spaced_sma)
    pa_array = LSQUnivariateSpline(
        isolist.sma, isolist.pa, nodes)(finely_spaced_sma)
    x0_array = LSQUnivariateSpline(
        isolist.sma, isolist.x0 + range_outward, nodes)(finely_spaced_sma)
    y0_array = LSQUnivariateSpline(
        isolist.sma, isolist.y0 + range_outward, nodes)(finely_spaced_sma)
    grad_array = LSQUnivariateSpline(
        isolist.sma, isolist.grad, nodes)(finely_spaced_sma)
    a3_array = LSQUnivariateSpline(
        isolist.sma, isolist.a3, nodes)(finely_spaced_sma)
    b3_array = LSQUnivariateSpline(
        isolist.sma, isolist.b3, nodes)(finely_spaced_sma)
    a4_array = LSQUnivariateSpline(
        isolist.sma, isolist.a4, nodes)(finely_spaced_sma)
    b4_array = LSQUnivariateSpline(
        isolist.sma, isolist.b4, nodes)(finely_spaced_sma)

    # Return deviations from ellipticity to their original amplitude meaning
    a3_array = -a3_array * grad_array * finely_spaced_sma
    b3_array = -b3_array * grad_array * finely_spaced_sma
    a4_array = -a4_array * grad_array * finely_spaced_sma
    b4_array = -b4_array * grad_array * finely_spaced_sma

    # correct deviations cased by fluctuations in spline solution
    eps_array[np.where(eps_array < 0.)] = 0.
    
    shape_new = np.array(shape) + np.array([2*range_outward, 2*range_outward])
    result = np.zeros(shape=shape_new)
    weight = np.zeros(shape=shape_new)

    eps_array[np.where(eps_array < 0.)] = 0.05

    # for each interpolated isophote, generate intensity values on the
    # output image array
    # for index in range(len(finely_spaced_sma)):
    for index in range(1, len(finely_spaced_sma)):
        sma0 = finely_spaced_sma[index]
        eps = eps_array[index]
        pa = pa_array[index]
        x0 = x0_array[index]
        y0 = y0_array[index]
        geometry = EllipseGeometry(x0, y0, sma0, eps, pa)

        intens = intens_array[index]

        # scan angles. Need to go a bit beyond full circle to ensure
        # full coverage.
        r = sma0
        phi = 0.
        while phi <= 2*np.pi + geometry._phi_min:
            # we might want to add the third and fourth harmonics
            # to the basic isophotal intensity.
            harm = 0.
            if high_harmonics:
                harm = (a3_array[index] * np.sin(3.*phi) +
                        b3_array[index] * np.cos(3.*phi) +
                        a4_array[index] * np.sin(4.*phi) +
                        b4_array[index] * np.cos(4.*phi)) / 4.

            # get image coordinates of (r, phi) pixel
            x = r * np.cos(phi + pa) + x0
            y = r * np.sin(phi + pa) + y0
            i = int(x)
            j = int(y)

            if (i > 0 and i < shape_new[1] - 1 and j > 0 and j < shape_new[0] - 1):
                # get fractional deviations relative to target array
                fx = x - float(i)
                fy = y - float(j)

                # add up the isophote contribution to the overlapping pixels
                result[j, i] += (intens + harm) * (1. - fy) * (1. - fx)
                result[j, i + 1] += (intens + harm) * (1. - fy) * fx
                result[j + 1, i] += (intens + harm) * fy * (1. - fx)
                result[j + 1, i + 1] += (intens + harm) * fy * fx

                # add up the fractional area contribution to the
                # overlapping pixels
                weight[j, i] += (1. - fy) * (1. - fx)
                weight[j, i + 1] += (1. - fy) * fx
                weight[j + 1, i] += fy * (1. - fx)
                weight[j + 1, i + 1] += fy * fx

                # step towards next pixel on ellipse
                phi = max((phi + 0.75 / r), geometry._phi_min)
                r = max(geometry.radius(phi), 0.5)
            # if outside image boundaries, ignore.
            else:
                break

    # zero weight values must be set to 1.
    weight[np.where(weight <= 0.)] = 1.

    # normalize
    result /= weight

    # fill value
    result[np.where(result == 0.)] = fill
    
    # reshape to initial size
    result = result[range_outward:range_outward+shape[0],
                    range_outward:range_outward+shape[1]]
    return result
