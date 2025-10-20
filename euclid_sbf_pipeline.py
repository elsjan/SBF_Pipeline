##########################################################################
# Pipeline for SBF amplitude calculation, specifically for Euclid data, 
# based on the code of Lei Titulaer
##########################################################################

# Imports 



import os 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS 
from astroscrappy import detect_cosmics
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model
from photutils.aperture import EllipticalAperture
from photutils.aperture import EllipticalAperture
from mgefit.find_galaxy import find_galaxy
from inspect import getsource
import sep

import sys
sys.path.append("./functions")

# Own function imports 
from extractdata import maskBrightCentralStars
from ellipsemodels import fitInitialEllipseModel, fitFinalEllipseModel, buildEllipseModel
from sourcemasking import findInitialSourceMask, findFinalSourceMask
from empiricalpsf import extractPsfSources
from fourierfunctions import calculateSBF
from librarypsfhubble import calculateLibrarySBF
from sbfuncertainties import sbfMagnitudeAnnuliSigmas


def openFits(file_path):
    """
    Open a SINGLE fits file, extract the data,
    exposure time, and wcs object (required for position information)
    """
    for file in os.listdir(file_path):
        if ".fits" in file:
            with fits.open(file_path+ "/" + file) as hdu:
                data = hdu[0].data
                exptime = hdu[0].header["EXPTIME"]
    return data, exptime

def maskBadPixels(data_frame, effective_gain=3.1, readnoise=4.5):
    """
    Identify and mask remaining bad pixels in Euclid VIS ERO images.
    Euclid images are already flat-fielded and cosmic-ray cleaned,
    so thresholds are gentler than HST defaults.
    """
    
    # Optional: mask bright stars if relevant
    try:
        star_mask = maskBrightCentralStars(data_frame)
    except Exception:
        star_mask = np.zeros_like(data_frame, dtype=bool)
    
    # Gentle cosmic-ray detection
    mask_cr, clean_frame = detect_cosmics(
        data_frame,
        gain=effective_gain,
        readnoise=readnoise,
        sigclip=6.0,
        sigfrac=0.5,
        objlim=5.0,
        satlevel=65535.0,  # Euclid VIS saturation (approx)
        verbose=False
    )
    bad_pixel_mask = mask_cr | star_mask

    # optional: mask known bad pixel regions from Euclid consortium
    # mask_euclid = fits.getdata(mask_filename).astype(bool)
    # bad_pixel_mask = bad_pixel_mask | mask_euclid
    
    return bad_pixel_mask


def createRequiredVariables(data, model_final, source_mask_final, total_background):
    """
    From the data and the model, the nri, model mask, and total mask is returned.
    """
    mask_model = model_final <= 1.5 * total_background
    mask_combined = np.array(~(mask_model | source_mask_final), dtype=int)
    
    nri = (data - model_final)/np.sqrt(model_final)
    nri[np.isinf(nri)] = 0
    
    mask_combined = ~np.isnan(nri)&mask_combined
    
    nri[np.isnan(nri)] = 0

    nri *= mask_combined
    return mask_model, mask_combined, nri

def MainExtractData(file_path):
    """
    New version of extractData function
    """
    data, exptime = openFits(file_path)
    mask_cr = maskBadPixels(data)
    return data, mask_cr, exptime

def MainFitInitialEllipseModel(data, mask_cr=None, plot=False, sma_normfactor=2):
    """
    New version of fitInitialEllipseModel function
    """
    f = find_galaxy(data, plot=False, quiet=True)
    geometry = EllipseGeometry(x0=f.ypeak, y0=f.xpeak, 
                               sma=f.majoraxis/sma_normfactor, eps=f.eps, 
                               pa=f.theta*np.pi/180, astep=0.1)
    masked_data = np.ma.masked_array(data, ~mask_cr)
    ellipse = Ellipse(masked_data, geometry)
    if plot:
        aperture = EllipticalAperture((geometry.x0, geometry.y0), 2*geometry.sma, 2*geometry.sma*(1-geometry.eps), geometry.pa)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(masked_data, origin='lower', cmap='gray', norm='log')
        aperture.plot(color='red', lw=1.5)
        plt.title("Initial Ellipse Fit")
        plt.show()
    nclip_sm = 2
    while nclip_sm <= 3:
        # values for fflag, maxgerr, step have been determined experimentally for HST data
        isolist = ellipse.fit_image(nclip=nclip_sm) #, step=0.3, fflag=0.35, maxgerr=0.85
        if len(isolist)!=0:
            break
        else: 
            nclip_sm += 1
    if len(isolist) == 0:
        print("Ellipse fitting failed")
        sys.exit()

    model_basic = buildEllipseModel(masked_data.shape, isolist, range_outward=200, 
                                        high_harmonics=False, gridspacing=0.1)
    

    residual_basic = masked_data - model_basic
    return residual_basic, model_basic

def maskBackgroundSources(data, mask_cr=None, plot=False, detect_thresh=1.5, minarea=5, r=2.5):
    """
    Detect and mask background sources using SEP (SExtractor).
    Works with masked arrays or normal numpy arrays.
    """
    data_clean = np.nan_to_num(np.ma.filled(data, 0)).astype(np.float32)

    # Estimate and subtract background
    bkg = sep.Background(data_clean, mask=mask_cr, bw=64, bh=64, fw=3, fh=3)
    data_sub = data_clean - bkg.back()

    # Detect sources
    objects = sep.extract(data_sub, thresh=detect_thresh * bkg.globalrms,
                          mask=mask_cr, minarea=minarea)

    # Mask them
    mask_sources = np.zeros(data.shape, dtype=bool)
    sep.mask_ellipse(mask_sources, objects['x'], objects['y'],
                     objects['a'], objects['b'], objects['theta'], r=r)

    # Combine
    mask_combined = mask_sources | mask_cr
    if plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(data_clean, origin='lower', cmap='gray', norm='log')
        plt.title("Detected Sources Mask")
        plt.imshow(mask_combined, origin='lower', cmap='Reds', alpha=0.3)
        plt.show()

    return mask_combined




def MainPipeline(file_path, image_path=None, make_plots=True):
    """
    Combining the original with the new functions
    """
    print("\n1. Extracting the data ...")
    data, mask_cr, exptime = MainExtractData(file_path)

    print("\n2. No background estimation performed ...")
    total_bckgr = 0
    # return data, mask_cr
    print("\n3. Fitting initial ellipse model ...")
    residual_basic, model_basic = MainFitInitialEllipseModel(data, mask_cr=~mask_cr, plot=make_plots)

    print("\n4. Finding initial source mask ...")
    source_mask = maskBackgroundSources(residual_basic, mask_cr=mask_cr, plot=make_plots)
    # source_mask, center_sources = findInitialSourceMask(residual_basic, model_basic, "somefilter",  "flt", ~mask_cr, plot=make_plots, image_path=image_path)

    print("\n5. Fitting final ellipse model ...")
    residual_final, model_final = MainFitInitialEllipseModel(data, mask_cr=~source_mask, plot=make_plots)

    print("\n6. Finding final source mask ...")
    source_mask_final, residual_power, sig_res_power = findFinalSourceMask(residual_final, model_final, "somefilter",  "flt", mask0=~mask_cr, plot=make_plots, image_path=image_path)

    print("\n7. Creating required variables ...")
    mask_model, mask_combined, nri = createRequiredVariables(data, model_final, source_mask_final, total_bckgr)
    return data, mask_model, mask_combined, nri




