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
from photutils.isophote import Ellipse, EllipseGeometry, Isophote, IsophoteList, build_ellipse_model
from photutils.aperture import EllipticalAperture
from mgefit.find_galaxy import find_galaxy
from inspect import getsource
import sep
from scipy.fft import fft2, fftshift
from scipy.optimize import curve_fit
from functools import partial
from astropy.visualization import ImageNormalize
from astropy.visualization import SinhStretch, AsymmetricPercentileInterval, LinearStretch,\
                                  LogStretch, PowerStretch, SqrtStretch, SquaredStretch,\
                                  HistEqStretch, ZScaleInterval, AsinhStretch

import sys
sys.path.append("./functions")

# Own function imports 
from extractdata import maskBrightCentralStars
from backgroundmodel import backgroundLevelAnalysis
from ellipsemodels import fitInitialEllipseModel, fitFinalEllipseModel, buildEllipseModel
from sourcemasking import findInitialSourceMask, findFinalSourceMask, unmaskMaxArea, centralAnnulusMask, maskCircle
from empiricalpsf import extractPsfSources
from fourierfunctions import calculateSBF
from librarypsfhubble import calculateLibrarySBF
from sbfuncertainties import sbfMagnitudeAnnuliSigmas




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
    if scale == 'asinh':
        stretch = AsinhStretch()
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


def createRequiredVariables(data, model_final, source_mask_final, total_background, geometry, plot=False, image_path=None):
    """
    From the data and the model, the nri, model mask, and total mask is returned.
    """

    # geometry.sma *= 3
    aperture = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma, geometry.sma*(1-geometry.eps), geometry.pa)
    aperture_mask_obj = aperture.to_mask(method='center',subpixels=1)
    aperture_mask = aperture_mask_obj.to_image(data.shape).astype(bool)

    inner_radius = geometry.sma*0.15

    mask_center = maskCircle(data, geometry.x0, geometry.y0, rout=inner_radius, rin=0)


    # mask_model = model_final <= 1.5 #* total_background   #ballsy change
    # mask_combined = np.array(~(mask_model | source_mask_final), dtype=int)

    mask_combined = ~(mask_center | source_mask_final)
    mask_combined &= aperture_mask

    nri = (data - model_final) / np.sqrt(model_final)
    nri[~np.isfinite(nri)] = 0

    nri *= mask_combined

    # mask_combined = np.array(~(mask_center | source_mask_final), dtype=int)
    # mask_combined = aperture_mask&mask_combined
    # nri = (data - model_final)/np.sqrt(model_final)
    # nri[np.isinf(nri)] = 0
    
    # mask_combined = ~np.isnan(nri)&mask_combined
    
    # nri[np.isnan(nri)] = 0

    # nri *= mask_combined
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        imdisplay(nri, ax, percentlow=1, percenthigh=99, scale='linear')
        plt.title("NRI")
        if image_path != None:
            image_title = "7.1_nri.png"
            plt.savefig(image_path + "/" + image_title)
        plt.show()
    return aperture_mask, mask_combined, nri

def MainExtractData(file_path, filter="VIS"):
    """
    New version of extractData function
    """
    data, exptime, mzp = openFits(file_path)
    mask_nan = np.isnan(data)
    mask_cr = maskBadPixels(data, filter=filter)
    mask_cr = mask_cr | mask_nan
    return data, mask_cr, exptime, mzp

def qtable_to_isophote_list(tbl):
    iso_list = IsophoteList()

    for row in tbl:
        iso = Isophote(sma=row['sma'],
            intens=row['intens'],
            int_err=row['intens_err'],
            grad=row['grad'],
            grad_error=row['grad_error'],
            grad_r_error=row['grad_rerror'],
            pa=row['pa'],
            pa_err=row['pa_err'],
            eps=row['ellipticity'],
            ellip_err=row['ellipticity_err'],
            x0=row['x0'],
            y0=row['y0'],
            x0_err=row['x0_err'],
            y0_err=row['y0_err'],
            # rms=row['rms'],
            ndata=row['ndata'],
            nflag=['nflag'],
            niter=row['niter'],
            stop_code=row['stop_code']
        )
        iso_list.append(iso)

    return IsophoteList(iso_list)
# 'sma', 'intens', 'int_err', 'eps', 'ellip_err',
#                            'pa', 'pa_err', 'grad', 'grad_error',
#                            'grad_r_error', 'x0', 'x0_err', 'y0', 'y0_err',
#                            'ndata', 'nflag', 'niter', 'stop_code

# 'intens_err', 'ellipticity',
#                                        'ellipticity_err', 'grad_rerror',
#                                        'nflag

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
    nonandata = np.where(np.isnan(data), np.nanmedian(data), data)

    f = find_galaxy(nonandata, quiet=True) # np.ma.masked_array(data, np.isnan(data)),
    if f.pa < 90:
        gpa = (f.pa+90)*np.pi/180
    else:
        gpa = (f.pa-90)*np.pi/180
    geometry = EllipseGeometry(x0=f.ypeak, y0=f.xpeak, 
                               sma=f.majoraxis/sma_normfactor, eps=f.eps, 
                               pa=gpa, astep=0.1)#, fix_center=True, fix_pa=True, fix_eps=True)
    masked_data = np.ma.masked_array(data, ~mask_cr)
    # Check if central pixel is masked
    x0, y0 = int(geometry.x0), int(geometry.y0)
    if masked_data.mask[y0, x0]:
        print("Center pixel is masked — unmasking central area.")
        masked_data.mask = ~(~masked_data.mask | centralAnnulusMask(nonandata, inner_radius=10))

    ellipse = Ellipse(masked_data, geometry)
    aperture = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma, geometry.sma*(1-geometry.eps), geometry.pa)
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        imdisplay(masked_data, ax, percentlow=1, percenthigh=95, scale='log')
        aperture.plot(color='red', lw=1.5)
        ax.plot(geometry.x0, geometry.y0, 'rx', markersize=7)
        plt.title(title_str) 
        plt.show()

 

    while nclip_sm <= 3:
        isolist = ellipse.fit_image(nclip=nclip_sm, fflag=0.3, step=0.3, fix_pa=True, fix_center=True, maxgerr=0.4)
                                    # , maxgerr=0.6)
                                    # , maxsma=1.5*geometry.sma) #dont fix center
                                   #  fix_center=True, 
                                    # sma0=10,
                                    # minsma=0.0,
                                    # maxsma=150.0),      # half of image size, covers galaxy
                                #     step=0.3,          # fine enough for detail
                                #     linear=True      # additive step=
                                # )
        if len(isolist)!=0:
            break
        else: 
            nclip_sm += 1
    # if len(isolist) == 0:
    #     print("Trying fixed ellipticity for ellipse fitting...")
    #     isolist = ellipse.fit_image(nclip=2, fflag=0.5, step=0.1)
    #                                 #, fix_center=True, fix_pa=True, fix_eps=True
                            
    #                         # maxsma=1.5*geometry.sma) 
    #                         # sma0=10,
    #                         # minsma=0.0,
    #                         # maxsma=150.0),      # half of image size, covers galaxy
    #                     #     step=0.3,          # fine enough for detail
    #                     #     linear=True      # additive step
    #                     # )
    if len(isolist) == 0:
        print("Trying larger step size for ellipse fitting...")
        isolist = ellipse.fit_image(nclip=2, fix_center=True, fix_pa=True
                            , fflag=0.5, step=0.3, maxgerr=0.6)
                            # maxsma=1.5*geometry.sma) 
                            # sma0=10,
                            # minsma=0.0,
                            # maxsma=150.0),      # half of image size, covers galaxy
                        #     step=0.3,          # fine enough for detail
                        #     linear=True      # additive step
                        # )
    if len(isolist) == 0:
        print("Ellipse fitting failed")
        sys.exit()
    range_outward = int(geometry.sma*1.3)  #just a guess right now
    isotable = isolist.to_table()
    # median_eps = np.median(isotable['ellipticity'][1:])
    # isotable['ellipticity'][1:] = median_eps
    print(isotable)
    # isolist = qtable_to_isophote_list(isotable)
    # for isophote in isolist:
    #     isophote.sample.geometry.eps = median_eps

    model_basic = buildEllipseModel(masked_data.shape, isolist, range_outward=range_outward, 
                                        high_harmonics=True, gridspacing=0.1)
    

    residual_basic = data - model_basic
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(model_basic, origin='lower', cmap='gray', norm='asinh')
        plt.title(f"{title_str} Isophote Model")
        if image_path != None:
            image_title = "5.1_isophote_model_fit.png"
            plt.savefig(image_path + "/" + image_title)   
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 8))
        imdisplay(residual_basic, ax, percentlow=1, percenthigh=99, scale='asinh')
        plt.title(f"{title_str} Residuals")
        if image_path != None:
            image_title = "5.2_isophote_model_residuals.png"
            plt.savefig(image_path + "/" + image_title)   
        plt.show()
    return residual_basic, model_basic, geometry

def maskBackgroundSources(data, mask_cr=None, plot=False, detect_thresh=3, minarea=7, maxarea=None, r=2.5, image_path=None, final=False, original_image=None):
    """
    Detect and mask background sources using SEP (SExtractor).
    Works with masked arrays or normal numpy arrays.
    """
    # data_clean = np.nan_to_num(np.ma.filled(data, 0)).astype(np.float32)
 
    mask_nan = ~np.isfinite(data)
    # Estimate and subtract background
    bkg = sep.Background(data.astype(np.float32), mask=(mask_cr | mask_nan), bw=64, bh=64, fw=3, fh=3)
    data_sub = data - bkg.back()

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

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        imdisplay(data, ax, percentlow=1, percenthigh=99, scale='asinh')
        plt.title("Detected Sources Mask")
        plt.imshow(mask_combined, origin='lower', cmap='Reds', alpha=0.5)
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


def MainPipeline(data_path, file_path=None, image_path=None, make_plots=True, psf=None, maxarea_sourcemask=None, filter="VIS", background_estimation=False, background=None):
    """
    Combining the original with the new functions
    """
    print("version 1")
    print("\n1. Extracting the data ...")
    data, mask_cr, exptime, mzp = MainExtractData(data_path, filter=filter)
    fig, ax = plt.subplots(figsize=(8, 8))
    imdisplay(data, ax, percentlow=1, percenthigh=99, scale='asinh')
    plt.title("Raw data")
    image_title = "1.1_raw_data.png"
    plt.savefig(image_path + "/" + image_title)

    if background_estimation:
        print("\n2. Performing background estimation ...")
        # mask_nan = ~np.isfinite(data)
        background = sep.Background(data.astype(np.float32), mask=mask_cr, bw=64, bh=64, fw=3, fh=3)
        total_bckgr = background.globalback - background.globalrms
        data -= total_bckgr
        bckgr_evol, data, total_bckgr, std_bckgr= backgroundLevelAnalysis(data, total_bckgr, make_plots, image_path=image_path)
        print(total_bckgr)
        print("\n Total background is", background.globalback, background.globalrms)
    else:
        print("\n2. No background estimation performed ...")
        if background != None:
            total_bckgr = background
            data -= total_bckgr
        else:
            total_bckgr = 0

    # plt.imshow(background.back())
    # plt.show()
    print("\n3. Fitting initial ellipse model ...")
    residual_basic, model_basic, geometry = MainFitEllipseModel(data, mask_cr=~mask_cr, plot=make_plots, final=False)

    print("\n4. Finding initial source mask ...")
    source_mask = maskBackgroundSources(residual_basic, mask_cr=mask_cr, plot=make_plots, maxarea=maxarea_sourcemask, r=3)
    # source_mask, center_sources = findInitialSourceMask(residual_basic, model_basic, "VIS",  "flt", ~mask_cr, plot=make_plots, image_path=image_path)

    print("\n5. Fitting final ellipse model ...")
    residual_final, model_final, geometry = MainFitEllipseModel(data, mask_cr=~source_mask, plot=make_plots, final=True, image_path=image_path)

    print("\n6. Finding final source mask ...")
    source_mask_final = maskBackgroundSources(residual_final, mask_cr=mask_cr, plot=make_plots, maxarea=maxarea_sourcemask, image_path=image_path, final=True, original_image=data, r=3)
    # source_mask_final, residual_power, sig_res_power = findFinalSourceMask(residual_final, model_final, "VIS",  "flt", mask0=~mask_cr, plot=make_plots, image_path=image_path)

    print("\n7. Creating required variables ...")
    mask_model, mask_combined, nri = createRequiredVariables(data, model_final, source_mask_final, total_bckgr, geometry, plot=make_plots, image_path=image_path)
    
    print("\n8 Calculate power spectra  ...")
    # fp, ps = ps_compute(nri, plot_ps=make_plots, plot_ft=make_plots)
    # psf_ps = ps_compute_psf(psf, nri)
    # sbf, noise, pcov, kfit_i, kfit_f = fitSbfComponents(ps, psf_ps, fit_range_i=0.2, fit_range_f=0.6)
    # psf_frames, log_power_spectra  = extractPsfSources(obs_filter, data, model_final, data_path, cutout_mask, total_bckgr,
    #                                                    plot=make_plots, 
    #                                                    image_path=image_path)

    image_ps, expected_ps, sbf, noise = calculateSBF(nri, mask_combined, psf,
                                                       norm_type = "MaskedPixels",
                                                       fit_range_i=0.1, fit_range_f=0.6,  
                                                       plot=make_plots,
                                                       image_path=image_path)
    print("\n9. Calculate sbf magnitude")
    sbfmag = appSBFmagnitude(sbf, mzp)
    print("\nSBF amplitude:", sbf)
    print("\nApparent SBF magnitude:", sbfmag)

    # print("\n10. ")
    if file_path != None:
        np.savetxt(file_path + "/data_background_subtracted", data)
        np.savetxt(file_path + "/combined_final_mask", mask_combined)

    return data, mask_model, mask_combined, nri, image_ps, expected_ps, sbf, noise, sbfmag, geometry




