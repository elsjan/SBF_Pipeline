##########################################################################
# Pipeline for SBF amplitude calculation, specifically for Euclid data, 
# based on the code of Lei Titulaer
##########################################################################

# Imports 



import os 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from astropy.io import fits
from astropy.wcs import WCS 
from astroscrappy import detect_cosmics
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model
from photutils.aperture import EllipticalAperture
from mgefit.find_galaxy import find_galaxy
from inspect import getsource
import sep
from scipy.fft import fft2, fftshift
from scipy.optimize import curve_fit
from functools import partial

import sys
sys.path.append("./functions")

# Own function imports 
from extractdata import maskBrightCentralStars
from ellipsemodels import fitInitialEllipseModel, fitFinalEllipseModel, buildEllipseModel
from sourcemasking import findInitialSourceMask, findFinalSourceMask, unmaskMaxArea, centralAnnulusMask
from empiricalpsf import extractPsfSources
from fourierfunctions import calculateSBF
from librarypsfhubble import calculateLibrarySBF
from sbfuncertainties import sbfMagnitudeAnnuliSigmas

from astropy.visualization import ImageNormalize
from astropy.visualization import SinhStretch, AsymmetricPercentileInterval, LinearStretch,\
                                  LogStretch, PowerStretch, SqrtStretch, SquaredStretch,\
                                  HistEqStretch, ZScaleInterval


def imdisplay(data,
              ax,
              vmin=None, vmax=None,
              percentlow=1, percenthigh=99,
              zscale=False,
              scale='linear',
              power=1.5,
              cmap='gray', colorbar=True,
              **kwargs):
    if zscale:
        # Always overwrite vmin and vmax
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data)
    if vmin is None or vmax is None:
        interval = AsymmetricPercentileInterval(percentlow, percenthigh)
        vmin2, vmax2 = interval.get_limits(data)
        if vmin is None:
            vmin = vmin2
        if vmax is None:
            vmax = vmax2

    if scale == 'linear':
        stretch = LinearStretch(slope=0.5, intercept=0.5)
    if scale == 'sinh':
        stretch = SinhStretch()
    if scale == 'log':
        stretch = LogStretch()
    if scale == 'power':
        stretch = PowerStretch(power)
    if scale == 'sqrt':
        stretch = SqrtStretch()
    if scale == 'squared':
        stretch = SquaredStretch()
    if scale == 'hist':
        stretch = HistEqStretch(data)  # Needs argument data and data min, max for vmin, vmax
        vmin = data.min(); vmax = data.max()

    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch)
    im = ax.imshow(data, interpolation='none', origin='lower', norm=norm, cmap=cmap, **kwargs)
    if colorbar:
        return im, plt.colorbar(im)
    else:
        return im

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
                mzp = hdu[0].header['ZP_STACK']
                hdu.close()
    return data, exptime, mzp

def maskBadPixels(data_frame, effective_gain=3.1, readnoise=4.5, filter="VIS", mask_filename=None):
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
    if filter == "VIS":
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
    elif filter == "H":
        mask_cr, clean_frame = detect_cosmics(
            data_frame,
            gain=effective_gain,
            readnoise=readnoise,
            sigclip=5.0,
            sigfrac=0.4,
            objlim=4.0,
            satlevel=1.118e5,  # Euclid NIR saturation (approx)
            verbose=False
        )
    else:
        print("Reminder to set the NIR filter parameters correctly!")
    bad_pixel_mask = mask_cr | star_mask

    # optional: mask known bad pixel regions from Euclid consortium
    # mask_euclid = fits.getdata(mask_filename).astype(bool)
    # bad_pixel_mask = bad_pixel_mask | mask_euclid
    
    return bad_pixel_mask


def createRequiredVariables(data, model_final, source_mask_final, total_background, plot=False, image_path=None):
    """
    From the data and the model, the nri, model mask, and total mask is returned.
    """
    mask_model = model_final <= 0.1 * total_background   #ballsy change
    mask_combined = np.array(~(mask_model | source_mask_final), dtype=int)
    
    nri = (data - model_final)/np.sqrt(model_final)
    nri[np.isinf(nri)] = 0
    
    mask_combined = ~np.isnan(nri)&mask_combined
    
    nri[np.isnan(nri)] = 0

    nri *= mask_combined
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        imdisplay(nri, ax, percentlow=1, percenthigh=95, scale='log')
        plt.title("NRI")
        if image_path != None:
            image_title = "7.1_nri.png"
            plt.savefig(image_path + "/" + image_title)
        plt.show()
    return mask_model, mask_combined, nri

def MainExtractData(file_path, filter="VIS"):
    """
    New version of extractData function
    """
    data, exptime, mzp = openFits(file_path)
    mask_cr = maskBadPixels(data, filter=filter)
    return data, mask_cr, exptime, mzp

def MainFitEllipseModel(data, mask_cr=None, plot=False, sma_normfactor=1, final=False, image_path=None):
    """
    New version of fitInitialEllipseModel function
    """
    if final==False:
        nclip_sm = 2
        title_str = "Initial Ellipse Fit"
    elif final==True:
        nclip_sm = 0
        title_str = "Final Ellipse Fit"
    f = find_galaxy(data, plot=False, quiet=True)
    geometry = EllipseGeometry(x0=f.ypeak, y0=f.xpeak, 
                               sma=f.majoraxis/sma_normfactor, eps=f.eps, 
                               pa=(90+f.pa)*np.pi/180, astep=0.1)
    masked_data = np.ma.masked_array(data, ~mask_cr)
    # Check if central pixel is masked
    x0, y0 = int(geometry.x0), int(geometry.y0)
    if masked_data.mask[y0, x0]:
        print("Center pixel is masked — unmasking central area.")
        # minsma = 1.0  # skip sma=0.0
        masked_data.mask[y0-10:y0+10, x0-10:x0+10] = False
        # centralAnnulusMask(masked_data, inner_radius=10) #set this!

    ellipse = Ellipse(masked_data, geometry)

    if plot:
        aperture = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma, geometry.sma*(1-geometry.eps), geometry.pa)
        fig, ax = plt.subplots(figsize=(8, 8))
        imdisplay(masked_data, ax, zscale=True, scale='log')
        aperture.plot(color='red', lw=1.5)
        ax.plot(geometry.x0, geometry.y0, 'rx', markersize=7)
        plt.title(title_str) 
        plt.show()

 

    while nclip_sm <= 3:
        isolist = ellipse.fit_image(nclip=nclip_sm, fflag=0.5, step=0.1, fix_pa=True, fix_center=True) #dont fix center
                                   #  fix_center=True, 
                                    # sma0=10,
                                    # minsma=0.0,
                                    # maxsma=150.0),      # half of image size, covers galaxy
                                #     step=0.3,          # fine enough for detail
                                #     linear=True      # additive step
                                # )
        if len(isolist)!=0:
            break
        else: 
            nclip_sm += 1
    if len(isolist) == 0:
        print("Trying larger step size for ellipse fitting...")
        isolist = ellipse.fit_image(nclip=2, fix_center=True, fix_pa=True
                            , fflag=0.5, step=0.3) 
                            # sma0=10,
                            # minsma=0.0,
                            # maxsma=150.0),      # half of image size, covers galaxy
                        #     step=0.3,          # fine enough for detail
                        #     linear=True      # additive step
                        # )
    if len(isolist) == 0:
        print("Ellipse fitting failed")
        sys.exit()
    range_outward = int(geometry.sma*1.5)  #just a guess right now
    model_basic = buildEllipseModel(masked_data.shape, isolist, range_outward=range_outward, 
                                        high_harmonics=True, gridspacing=0.1)
    

    residual_basic = data - model_basic
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(model_basic, origin='lower', cmap='gray', norm='log')
        plt.title(f"{title_str} isophote model")
        if image_path != None:
            image_title = "5.1_isophote_model_fit.png"
            plt.savefig(image_path + "/" + image_title)   
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 8))
        imdisplay(residual_basic, ax, zscale=True, scale='log')
        plt.title(f"{title_str} Residuals")
        if image_path != None:
            image_title = "5.2_isophote_model_residuals.png"
            plt.savefig(image_path + "/" + image_title)   
        plt.show()
    return residual_basic, model_basic

def maskBackgroundSources(data, mask_cr=None, plot=False, detect_thresh=3, minarea=7, maxarea=None, r=2.5, image_path=None, final=False, original_image=None):
    """
    Detect and mask background sources using SEP (SExtractor).
    Works with masked arrays or normal numpy arrays.
    """
    data_clean = np.nan_to_num(np.ma.filled(data, 0)).astype(np.float32)

    # Estimate and subtract background
    bkg = sep.Background(data_clean, mask=mask_cr, bw=64, bh=64, fw=3, fh=3)
    data_sub = data_clean - bkg.back()

    # Detect sources
    objects, segmap = sep.extract(data_sub, thresh=detect_thresh * bkg.globalrms,
                          mask=mask_cr, minarea=minarea, segmentation_map=True)
    if maxarea != None:
        objects, segmap = unmaskMaxArea(objects, segmap, maxarea)

    # Mask them
    mask_sources = np.zeros(data.shape, dtype=bool)
    sep.mask_ellipse(mask_sources, objects['x'], objects['y'],
                     objects['a'], objects['b'], objects['theta'], r=r)

    # Combine
    mask_combined = mask_sources | mask_cr
    if final:
        if original_image.all() == None:
            mask_combined = mask_combined | centralAnnulusMask(data, inner_radius=25)
        else:
            mask_combined = mask_combined | centralAnnulusMask(original_image, inner_radius=25)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        imdisplay(data_clean, ax, zscale=True, scale='log')
        plt.title("Detected Sources Mask")
        plt.imshow(mask_combined, origin='lower', cmap='Reds', alpha=0.3)
        if image_path != None:
            image_title = "6.1_source_mask.png"
            plt.savefig(image_path + "/" + image_title)
        plt.show()
    
    return mask_combined

###############
# PSF and SBF calculation
###############

#compute the 2D fourier transform and the power spectrum, now in a function
def ps_compute(image, plot_ft=False, plot_ps=False):
    fft_nonshift = fft2(image)
    fft_shift = fftshift(fft_nonshift)  # shift puts the 0,0 point in the middel of the image
    fourier_power = np.abs(fft_shift)**2  
    
    npix = fourier_power.shape[0]
    freq_1d = np.fft.fftfreq(npix) * npix
    freq_2d = np.array(np.meshgrid(freq_1d, freq_1d))+0.5  #why the +0.5?
    freq_normed = np.sqrt(freq_2d[0]**2 + freq_2d[1]**2)
    frequencies = np.fft.fftshift(freq_normed)  #second shift
    bin_corners = np.arange(0, npix//2+1)
    power_spectrum, _, _ = sc.stats.binned_statistic(frequencies.flatten(), 
                                               fourier_power.flatten(),
                                               statistic = "mean",
                                               bins = bin_corners)
    if plot_ft:
        fig = plt.figure()
        plt.imshow(fourier_power, cmap='grey', norm='log')
        plt.colorbar()
        plt.show()
    if plot_ps:
        fig = plt.figure()
        plt.plot(range(len(power_spectrum)), power_spectrum)
        plt.yscale('log')
        plt.xlabel(r"Wavenumber k  [px$^{-1}$]")
        plt.ylabel(r"P(k) [e$^-$ s$^{-1}$ px$^-1$]")
        plt.show()
    
    return fourier_power, power_spectrum

def ps_compute_psf(psf, image):
    """Does the exact same thing as computing the ps of the image, but finds the ps of the psf we convoluted with
    The image is used to match up the shapes"""
    
    fft_nonshift = fft2(psf, s=image.shape) #padded to the image!
    fft_shift = fftshift(fft_nonshift)  # shift puts the 0,0 point in the middel of the image
    fourier_power = np.abs(fft_shift)**2  
    
    npix = fourier_power.shape[0]
    freq_1d = np.fft.fftfreq(npix) * npix
    freq_2d = np.array(np.meshgrid(freq_1d, freq_1d))+0.5  #why the +0.5?
    freq_normed = np.sqrt(freq_2d[0]**2 + freq_2d[1]**2)
    frequencies = np.fft.fftshift(freq_normed)  #second shift
    bin_corners = np.arange(0, npix//2+1)
    power_spectrum, _, _ = sc.stats.binned_statistic(frequencies.flatten(), 
                                               fourier_power.flatten(),
                                               statistic = "mean",
                                               bins = bin_corners)
    
    return power_spectrum

def sbfToFit(k, P0, P1, Ek):
    Pk = P0 * Ek + P1
    return Pk


def fitSbfComponents(image_ps, expected_ps, fit_range_i=0.2, fit_range_f=0.6):
    """Taken directly from code Lei"""
    kfit_i = int(fit_range_i*len(image_ps)) # to define the range of the ps to fit
    kfit_f = int(fit_range_f*len(image_ps))
    
    sbfToFitAdjusted = partial(sbfToFit, Ek=expected_ps[kfit_i:kfit_f])
    popt, pcov = curve_fit(f=sbfToFitAdjusted, 
                              xdata=np.linspace(0,1,len(image_ps[kfit_i:kfit_f])), 
                              ydata=image_ps[kfit_i:kfit_f], 
                              bounds=[[0, 0],[np.inf, np.inf]])   
    sbf, noise = popt
    return sbf, noise, pcov, kfit_i, kfit_f
    
def appSBFmagnitude(sbf, mzp):
    m = -2.5 * np.log10(sbf) + mzp
    return m


###############
# Main SBF calculation pipeline function
###############


def MainPipeline(file_path, image_path=None, make_plots=True, psf=None, maxarea_sourcemask=None, filter="VIS", background_estimation=False):
    """
    Combining the original with the new functions
    """
    print("\n1. Extracting the data ...")
    data, mask_cr, exptime, mzp = MainExtractData(file_path, filter=filter)
    
    if background_estimation:
        print("\n2. Performing background estimation ...")
        mask_nan = ~np.isfinite(data)
        background = sep.Background(data.astype(np.float32), mask=(mask_cr | mask_nan), bw=64, bh=64, fw=3, fh=3)
        total_bckgr = background.globalback
        print("\n Total background is", total_bckgr)
    else:
        print("\n2. No background estimation performed ...")
        total_bckgr = 0
    data -= total_bckgr
    print("\n3. Fitting initial ellipse model ...")
    residual_basic, model_basic = MainFitEllipseModel(data, mask_cr=~mask_cr, plot=make_plots, final=False)

    print("\n4. Finding initial source mask ...")
    source_mask = maskBackgroundSources(residual_basic, mask_cr=mask_cr, plot=make_plots, maxarea=maxarea_sourcemask)
    # source_mask, center_sources = findInitialSourceMask(residual_basic, model_basic, "VIS",  "flt", ~mask_cr, plot=make_plots, image_path=image_path)

    print("\n5. Fitting final ellipse model ...")
    residual_final, model_final = MainFitEllipseModel(data, mask_cr=~source_mask, plot=make_plots, final=True, image_path=image_path)

    print("\n6. Finding final source mask ...")
    source_mask_final = maskBackgroundSources(residual_final, mask_cr=mask_cr, plot=make_plots, maxarea=maxarea_sourcemask, image_path=image_path, final=True, original_image=data)
    # source_mask_final, residual_power, sig_res_power = findFinalSourceMask(residual_final, model_final, "VIS",  "flt", mask0=~mask_cr, plot=make_plots, image_path=image_path)

    print("\n7. Creating required variables ...")
    mask_model, mask_combined, nri = createRequiredVariables(data, model_final, source_mask_final, total_bckgr, plot=make_plots, image_path=image_path)
    
    print("\n8 Calculate power spectra  ...")
    # fp, ps = ps_compute(nri, plot_ps=make_plots, plot_ft=make_plots)
    # psf_ps = ps_compute_psf(psf, nri)
    # sbf, noise, pcov, kfit_i, kfit_f = fitSbfComponents(ps, psf_ps, fit_range_i=0.2, fit_range_f=0.6)
    # psf_frames, log_power_spectra  = extractPsfSources(obs_filter, data, model_final, data_path, cutout_mask, total_bckgr,
    #                                                    plot=make_plots, 
    #                                                    image_path=image_path)

    image_ps, expected_ps, sbf, noise = calculateSBF(nri, mask_combined, psf,
                                                       norm_type = "MaskedPixels",
                                                       fit_range_i=0.2, fit_range_f=0.6,  
                                                       plot=make_plots,
                                                       image_path=image_path)
    
    sbfmag = appSBFmagnitude(sbf, mzp)
    print("\nSBF amplitude:", sbf)
    print("\nApparent SBF magnitude:", sbfmag)

    return data, mask_model, mask_combined, nri, image_ps, expected_ps, sbf, noise, sbfmag




