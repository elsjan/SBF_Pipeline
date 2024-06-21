import numpy as np
import matplotlib.pyplot as plt

import sys
# path where the individual components are stored.
sys.path.append("C:/Users/Lei/Documents/Courses/MSc Astronomy/Thesis/MAIN/pipeline/functions")

from mgefit.find_galaxy import find_galaxy
from tqdm.notebook import tqdm

# from sbf_pipeline import sbfPipeline

from fourierfunctions import getNormFactor, imageToPowerSpectrum, findExpectedPS
from fourierfunctions import fitSbfComponents, plotSbfAnalysis



###########################################################################################

def obtainZeroPoint(obs_filter):
    """
    Return zero-point magnitude given an observation filter
    
    sources: 
    https://iopscience.iop.org/article/10.1086/444553/pdf
    """
    if obs_filter == "F110W":
        return 26.8223
    elif obs_filter == "F160W":
        return 25.9463
    elif obs_filter == "F475W":
        return  26.068
    elif obs_filter == "F850LP":
        return  24.862
    else:
        print("Please enter a valid filter (F110W; F160W, F475W or F850LP).")
        return 0    
    
def obtainApparentMag(P0, Pr, obs_filter):
    
    if P0 == 0:
        print("No amplitude present.")
        return 0
    
    zero_point = obtainZeroPoint(obs_filter)
    
    return -2.5 * np.log10(P0 - Pr) + zero_point

obtainApparentMag = np.vectorize(obtainApparentMag)

def maskCircle(data, x0, y0, rout, rin=0):
    """
    Mask a circular annulus with given center, inner and outer radius. 
    
    The area between the inner and outer radius is returned as True values.
    """
    x0 = int(np.round(x0))
    y0 = int(np.round(y0))
    
    shape = data.shape
    
    xrange = np.arange(max(0,x0-rout),min(shape[1],x0+rout), dtype=int)
    yrange = np.arange(max(0,y0-rout),min(shape[0],y0+rout), dtype=int)

    mask = np.zeros(shape, dtype=bool)
    
    for x in xrange:
        for y in yrange:
            r = np.sqrt((x-x0)**2 + (y-y0)**2)
            if (r <= rout) & (r >= rin):
                mask[y,x] = True
                
    return mask

###########################################################################################
# Uncertainty calculators
###########################################################################################

def psfSigmaAndPS(obs_filter, nri, mask_combined, psf_frames, lib_psf, 
                 residual_power=0, fit_range_i=0.2, fit_range_f=0.6):
    """
    Function fits the SBF power spectrum for each psf frame in the list.
    
    An mean sbf magnitude and uncertainty is returned given the data.
    """
    if (obs_filter == "F160W") & (len(np.shape(psf_frames)) == 3):
        psf_list = np.vstack([psf_frames, lib_psf])
    elif (obs_filter == "F110W") & (len(np.shape(psf_frames)) == 3):
        psf_list = np.vstack([psf_frames, lib_psf])
    else:
        psf_list = lib_psf
        
    norm_factor = getNormFactor("MaskedPixels", mask_combined)
    
    # Calculate the azimutally averaged Power Spectrum of the image
    image_ps = imageToPowerSpectrum(nri)/norm_factor
    
    sbf_list = []
    power_spectra = []
    
    kfit_i = int(fit_range_i*len(image_ps))
    kfit_f = int(fit_range_f*len(image_ps))
    
    print("finding the different psf power spectra ...")
    for psf in tqdm(psf_list):
        expected_ps = findExpectedPS(psf, mask_combined, norm_factor)
        
        if np.any(expected_ps == None):
            print("No psf sources found, no SBF amplitude fit")
            return image_ps, [0], 0, 0
    
        sbf_i, _ = fitSbfComponents(image_ps, expected_ps, kfit_i, kfit_f, 
                                      plot=False, image_path=None)
        
        sbf_list.append(sbf_i)
        power_spectra.append(expected_ps)

    mags = obtainApparentMag(sbf_list, residual_power, obs_filter)
    mean_mag = np.mean(mags)
    sig_psf = np.std(mags)
    
    mean_expected_ps = np.mean(power_spectra, axis=0)
    
    return mean_mag, sig_psf, image_ps, mean_expected_ps 

          
def kfitSigma(obs_filter, image_ps, expected_ps, fit_i_range, fit_f_range, 
              residual_power=0):
    """
    Function estimates the uncertainty on different kfit_i and kfit_f,
    by fitting the sbf power spectrum on a grid of kfit_i and kfit_f.
    
    The final returned mag and sigma is the mean and std of the full grid range.
    """
    fit_i = np.linspace(fit_i_range[0], fit_i_range[1], 80)
    fit_f = np.linspace(fit_f_range[0], fit_f_range[1], 80)
    
    grid = np.zeros([len(fit_i), len(fit_f)])
    grid_noise = np.zeros([len(fit_i), len(fit_f)])

    for idx_i in range(len(fit_i)):
        for idx_f in range(len(fit_f)):
            kfit_i = int(fit_i[idx_i]*len(image_ps))
            kfit_f = int(fit_f[idx_f]*len(image_ps))

            sbf, noise = fitSbfComponents(image_ps, expected_ps, 
                                      kfit_i, kfit_f, plot=False)
            grid[idx_i, idx_f] = sbf
            grid_noise[idx_i, idx_f] = noise

       
    mags = obtainApparentMag(grid, residual_power, obs_filter)
    mean_mag = np.mean(mags)
    std_mag = np.std(mags)
    return mean_mag, std_mag, [np.mean(grid), np.mean(grid_noise)]

          
def backgroundSigma(nri, image_model, mask_combined, sigma_background, obs_filter,
                    expected_ps, residual_power=0, fit_range_i=0.2, fit_range_f=0.6):
    """
    Calculate the uncertainty given the sigma on the background.
    """
    mask_new = image_model<=0
    mask_combined = mask_combined | mask_new
    
    image_model[mask_combined] = 1
    
    nri_top = nri + (sigma_background/np.sqrt(image_model)) * mask_combined
    nri_bot = nri - (sigma_background/np.sqrt(image_model)) * mask_combined
    
    nri_top[np.isnan(nri_top) | np.isinf(nri_top)] = 0
    nri_bot[np.isnan(nri_bot) | np.isinf(nri_bot)] = 0
    
    norm_factor = getNormFactor("MaskedPixels", mask_combined)
          
    image_ps_top = imageToPowerSpectrum(nri_top)/norm_factor
    image_ps_bot = imageToPowerSpectrum(nri_bot)/norm_factor
    
    kfit_i = int(fit_range_i*len(image_ps_top))
    kfit_f = int(fit_range_f*len(image_ps_top))

    sbf_top, _ = fitSbfComponents(image_ps_top, expected_ps, 
                                  kfit_i, kfit_f, plot=False)
    sbf_bot, _ = fitSbfComponents(image_ps_bot, expected_ps, 
                                  kfit_i, kfit_f, plot=False)
    
    mag_top = obtainApparentMag(sbf_top, residual_power, obs_filter)
    mag_bot = obtainApparentMag(sbf_bot, residual_power, obs_filter)
    
    sigma_mag = np.abs(mag_top - mag_bot)/2
    return sigma_mag


def meanMagPerPixel(data, mask_combined, sigma_background, obs_filter):
    """
    Calculate the magnitude per pixel given a data set and an observation.
    
    The uncerainty is determined by taking into account the uncertainty on the 
    subtracted background level.
    """
    
    zero_point = obtainZeroPoint(obs_filter)
          
    mean_pix_flux = np.mean(data[mask_combined])
    
    mag_pix = -2.5 * np.log10(mean_pix_flux) + zero_point
    sig_mag = (2.5/np.log(10)) * (sigma_background / mean_pix_flux)
    
    return mag_pix, sig_mag
          


def calculateMagsWithUncertainties(obs_filter, nri, mask_combined, 
                                  psf_frames, lib_psf, sigma_background, data, 
                                  image_model, residual_power=0, 
                                  kfit_i=0.2, kfit_f=0.6, 
                                  fit_i_range=[0.1, 0.4], fit_f_range=[0.6, 0.8],
                                  plot=False, image_path=None, image_title=None):
    """
    Calculate the final sbg signal together with the uncertainties
    
    Uncertainties are calculated through the PSF and PS fit range.
    """
    mean_mag_psf, sig_psf, image_ps, expected_ps = psfSigmaAndPS(obs_filter, 
                                    nri, mask_combined, psf_frames, lib_psf, 
                                    residual_power, kfit_i, kfit_f)

    print("finding the kfit sigma ... ")
    mean_mag_kfit, sig_kfit, powers = kfitSigma(obs_filter, image_ps, expected_ps, 
                                    fit_i_range, fit_f_range, residual_power)
          
    sig_bckgr = backgroundSigma(nri, image_model, mask_combined, sigma_background, 
                                    obs_filter, expected_ps, residual_power, 
                                    kfit_i, kfit_f)
    
    sigma_total = np.sqrt(sig_psf**2 + sig_kfit**2 + sig_bckgr**2)
    
    if plot == True:
        plotSbfAnalysis(image_ps, expected_ps, powers[0], powers[1], 
                        kfit_i*len(image_ps), kfit_f*len(image_ps), 
                        image_path, image_title)
          
    # mean mag
    mag_pix, sig_mag = meanMagPerPixel(data, mask_combined, sigma_background, obs_filter)
          
    return mean_mag_kfit, sigma_total, mag_pix, sig_mag, sig_psf, sig_kfit, sig_bckgr
    
          
###########################################################################################
# Main function
###########################################################################################

def sbfMagnitudeAnnuliSigmas(obs_filter, nri, mask_combined, 
                                  psf_frames, lib_psf, sigma_background, data, 
                                  image_model, residual_power=0, 
                                  kfit_i=0.2, kfit_f=0.6, 
                                  fit_i_range=[0.1, 0.4], fit_f_range=[0.6, 0.8],
                                  radii_arcsec = [[4, 9], [9, 16], [16, 30], [30,60]],
                                  plot=True, image_path=None):
    """
    Find the sbf magnitude with uncertainty in different annuli of the same image.
    Also the magnitude for the total image is returned, with those uncertainties.
    """
    outputs = []
    full_output = calculateMagsWithUncertainties(obs_filter, nri, 
                                  mask_combined, psf_frames, lib_psf, 
                                  sigma_background, data, image_model, residual_power, 
                                  kfit_i, kfit_f, fit_i_range, fit_f_range,
                                  plot=plot, image_path=image_path, 
                                  image_title="11.1_sbf_full_image.png")
    outputs.append(full_output)
         
    if (obs_filter == "F160W") | (obs_filter == "F110W"):
        plate_scale = 0.13 # arcsec / pixel
    else:
        plate_scale = 0.05 # arcsec / pixel
          
    radii_pix = np.array(radii_arcsec) / plate_scale
    
    f = find_galaxy(image_model, quiet=True)
    
    idx_plot = 1
    for radii in radii_pix:
        mask_annuli = maskCircle(mask_combined, f.ypeak, f.xpeak, 
                                 rout=radii[1], rin=radii[0])
        mask_new = mask_combined & mask_annuli
        
        nri_new = nri * mask_new
        
        if len(mask_new[mask_new]) <= 0.5* len(mask_annuli[mask_annuli]):
            outputs.append([0,0,0,0,0,0,0])
        else:
            output_i = calculateMagsWithUncertainties(obs_filter, nri_new, 
                                  mask_new, psf_frames, lib_psf, 
                                  sigma_background, data, image_model, residual_power, 
                                  kfit_i, kfit_f, fit_i_range, fit_f_range,
                                  plot=plot, image_path=image_path, 
                                  image_title="11.{}_sbf_radius.png".format(idx_plot))
            outputs.append(output_i)
        idx_plot += 1
        
    return np.array(outputs)

